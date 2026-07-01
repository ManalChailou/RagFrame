# setup_vector_store.py

from contextlib import ExitStack
from pathlib import Path
import os

from dotenv import load_dotenv, set_key
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
OLD_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")
ENV_FILE = Path(".env")

# Original project filenames are preserved.
FILE_PATHS = [
    Path("rag_docs/COSMIC-Guidelines.pdf"),
    Path("rag_docs/COSMIC-Principles-Definitions-Rules.pdf"),
    Path("rag_docs/COSMIC-Examples-Business.txt"),
    Path("rag_docs/COSMIC-Examples-Real-Time.txt"),
]

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from the .env file.")

missing_files = [str(path) for path in FILE_PATHS if not path.is_file()]
if missing_files:
    raise FileNotFoundError(
        "The following files were not found:\n- "
        + "\n- ".join(missing_files)
    )

client = OpenAI(api_key=API_KEY)

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
        print(f"Warning: old vector store could not be deleted: {exc}")

vector_store = client.vector_stores.create(
    name="COSMIC Knowledge Base"
)
print(f"New vector store created: {vector_store.id}")

with ExitStack() as stack:
    file_streams = [
        stack.enter_context(path.open("rb"))
        for path in FILE_PATHS
    ]

    batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=file_streams,
    )

print(f"Batch status: {batch.status}")
print(f"Files completed: {batch.file_counts.completed}")
print(f"Files failed: {batch.file_counts.failed}")

if batch.status != "completed" or batch.file_counts.failed > 0:
    raise RuntimeError(
        "The new vector store was created, but one or more files failed "
        "during upload or indexing."
    )

if not ENV_FILE.exists():
    ENV_FILE.touch()

set_key(
    dotenv_path=str(ENV_FILE),
    key_to_set="OPENAI_VECTOR_STORE_ID",
    value_to_set=vector_store.id,
)

print("\nKnowledge base replaced successfully.")
print(f"OPENAI_VECTOR_STORE_ID={vector_store.id}")
print("Restart the FastAPI application before testing retrieval.")
