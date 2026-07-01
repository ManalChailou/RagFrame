from contextlib import ExitStack
from pathlib import Path
import os
import re
import shutil

from dotenv import load_dotenv, set_key
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
OLD_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")
ENV_FILE = Path(".env")

# Readable source file created from the original JSONL.
SOURCE_FILE = Path("rag_docs/cosmic_rag_update_structured.txt")

# Temporary files uploaded to the Vector Store.
# Each file contains only useful COSMIC content.
TEMP_DIR = Path("rag_docs/generated_cosmic_contexts")


def _safe_name(value: str) -> str:
    value = (value or "general").strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "", value)
    return value or "general"


def _clean_content(lines: list[str]) -> str:
    """
    Keep only useful readable content.

    Removed:
    - COSMIC KNOWLEDGE BASE
    - APPLICATION DOMAIN
    - COSMIC COMPONENT
    - Type
    - section separators
    - decorative characters
    """
    cleaned = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        if re.fullmatch(r"[-=#]{5,}", stripped):
            continue

        if stripped.startswith("### "):
            continue

        if stripped.startswith("## COSMIC COMPONENT:"):
            continue

        if stripped.startswith("APPLICATION DOMAIN:"):
            continue

        if stripped == "COSMIC KNOWLEDGE BASE":
            continue

        if stripped.startswith(
            "This document contains COSMIC definitions"
        ):
            continue

        cleaned.append(line)

    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned).strip()


def build_upload_files() -> list[Path]:
    """
    Split the structured source into small focused TXT files.

    Metadata is encoded only in filenames so it can be used by the backend
    for filtering. It is not included in the indexed passage text.
    """
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(
            f"COSMIC source file not found: {SOURCE_FILE.resolve()}"
        )

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    text = SOURCE_FILE.read_text(encoding="utf-8")
    lines = text.splitlines()

    current_domain = "general"
    current_component = "general"
    current_type = "knowledge"
    current_record_lines: list[str] = []
    generated_files: list[Path] = []
    record_number = 0

    def flush_record() -> None:
        nonlocal current_record_lines, record_number

        content = _clean_content(current_record_lines)
        current_record_lines = []

        if not content:
            return

        record_number += 1
        filename = (
            f"cosmic"
            f"__domain-{_safe_name(current_domain)}"
            f"__component-{_safe_name(current_component)}"
            f"__type-{_safe_name(current_type)}"
            f"__record-{record_number:03d}.txt"
        )

        output_path = TEMP_DIR / filename
        output_path.write_text(content + "\n", encoding="utf-8")
        generated_files.append(output_path)

    for raw_line in lines:
        stripped = raw_line.strip()

        domain_match = re.match(
            r"APPLICATION DOMAIN:\s*(.+)",
            stripped,
            flags=re.IGNORECASE,
        )
        if domain_match:
            flush_record()
            current_domain = domain_match.group(1).strip().lower().replace(" ", "_")
            continue

        component_match = re.match(
            r"##\s*COSMIC COMPONENT:\s*(.+)",
            stripped,
            flags=re.IGNORECASE,
        )
        if component_match:
            flush_record()
            current_component = (
                component_match.group(1).strip().lower().replace(" ", "_")
            )
            continue

        section_match = re.match(
            r"###\s*(.*?)(?:\s*\|\s*Type:\s*(.+))?$",
            stripped,
            flags=re.IGNORECASE,
        )
        if section_match:
            flush_record()
            current_type = (
                section_match.group(2) or "knowledge"
            ).strip().lower().replace(" ", "_")
            # Do not add the section title or Type to the indexed content.
            continue

        # A row of hyphens closes the current record.
        if re.fullmatch(r"-{20,}", stripped):
            flush_record()
            continue

        current_record_lines.append(raw_line)

    flush_record()

    if not generated_files:
        raise RuntimeError(
            "No usable COSMIC passages were generated from the source file."
        )

    print(
        f"Generated {len(generated_files)} clean COSMIC context files."
    )
    return generated_files


if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from the .env file.")

client = OpenAI(api_key=API_KEY)
new_vector_store = None

try:
    upload_files = build_upload_files()

    if OLD_VECTOR_STORE_ID:
        print(f"Deleting old vector store: {OLD_VECTOR_STORE_ID}")
        try:
            deleted = client.vector_stores.delete(
                vector_store_id=OLD_VECTOR_STORE_ID
            )
            if getattr(deleted, "deleted", False):
                print("Old vector store deleted successfully.")
            else:
                print("Warning: deletion was not confirmed by the API.")
        except Exception as exc:
            print(
                "Warning: old vector store could not be deleted "
                f"or no longer exists: {exc}"
            )

    new_vector_store = client.vector_stores.create(
        name="COSMIC Clean Knowledge Base"
    )
    print(f"New vector store created: {new_vector_store.id}")

    with ExitStack() as stack:
        streams = [
            stack.enter_context(path.open("rb"))
            for path in upload_files
        ]

        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=new_vector_store.id,
            files=streams,
        )

    print(f"Batch status: {batch.status}")
    print(f"Files completed: {batch.file_counts.completed}")
    print(f"Files failed: {batch.file_counts.failed}")

    if batch.status != "completed" or batch.file_counts.failed > 0:
        raise RuntimeError(
            "One or more COSMIC context files failed during indexing."
        )

    if not ENV_FILE.exists():
        ENV_FILE.touch()

    set_key(
        dotenv_path=str(ENV_FILE),
        key_to_set="OPENAI_VECTOR_STORE_ID",
        value_to_set=new_vector_store.id,
    )

    print("\nCOSMIC knowledge base replaced successfully.")
    print(f"OPENAI_VECTOR_STORE_ID={new_vector_store.id}")
    print("Restart the FastAPI application before testing retrieval.")

except Exception:
    if new_vector_store is not None:
        try:
            client.vector_stores.delete(
                vector_store_id=new_vector_store.id
            )
            print(
                f"Failed vector store removed: {new_vector_store.id}"
            )
        except Exception as cleanup_error:
            print(
                "Warning: failed vector store could not be removed: "
                f"{cleanup_error}"
            )
    raise

finally:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        print(f"Temporary files removed: {TEMP_DIR}")
