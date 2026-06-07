import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Put the JSONL file here, or change this path.
EXAMPLES_JSONL_PATH = Path("rag_docs/cosmic_examples_chunks.jsonl")

# If False, definitions/rules/knowledge chunks are also stored as ExampleKnowledge.
LOAD_ONLY_EXAMPLES = os.getenv("LOAD_ONLY_EXAMPLES", "true").lower() == "true"

if not NEO4J_PASSWORD:
    raise RuntimeError("Missing NEO4J_PASSWORD in .env")


class CosmicExamplesGraphLoader:
    """
    Loads COSMIC example chunks into Neo4j.

    Expected input: JSONL records like:
    {
      "id": "chunk_011",
      "app_domain": "business",
      "section": "Examples — Functional Processes",
      "type": "example",
      "domain": "functional_processes",
      "keywords": [...],
      "content": "Input: ... Expected Output: ..."
    }

    The loader:
    - Stores the raw chunk.
    - Extracts the FUR input text.
    - Creates example nodes for FU / FP / DG / SP depending on the chunk domain.
    - Links SP examples to MovementType nodes from the core ontology.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def run(self, query: str, params: Optional[dict] = None):
        with self.driver.session() as session:
            session.run(query, params or {})

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def create_constraints(self):
        logger.info("Creating example graph constraints...")

        constraints = [
            """
            CREATE CONSTRAINT example_id IF NOT EXISTS
            FOR (c:ExampleKnowledge)
            REQUIRE c.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT fur_example_id IF NOT EXISTS
            FOR (f:FURExample)
            REQUIRE f.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT domain_name IF NOT EXISTS
            FOR (d:Domain)
            REQUIRE d.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT example_fu_id IF NOT EXISTS
            FOR (fu:ExampleFunctionalUser)
            REQUIRE fu.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT example_fp_id IF NOT EXISTS
            FOR (fp:ExampleFunctionalProcess)
            REQUIRE fp.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT example_dg_id IF NOT EXISTS
            FOR (dg:ExampleDataGroup)
            REQUIRE dg.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT example_ooi_id IF NOT EXISTS
            FOR (o:ExampleObjectOfInterest)
            REQUIRE o.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT example_attr_id IF NOT EXISTS
            FOR (a:ExampleDataAttribute)
            REQUIRE a.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT example_sp_id IF NOT EXISTS
            FOR (sp:ExampleSubProcess)
            REQUIRE sp.id IS UNIQUE
            """
        ]

        for q in constraints:
            self.run(q)

    # ------------------------------------------------------------------
    # Generic parsing helpers
    # ------------------------------------------------------------------

    def clean(self, value: Any) -> str:
        return str(value or "").strip()

    def normalize_list(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [self.clean(v) for v in value if self.clean(v)]
        return [self.clean(value)]

    def extract_input_text(self, content: str) -> str:
        content = content or ""
        match = re.search(r"Input:\s*(.*?)(?:\nExpected Output:|$)", content, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return content.strip()

    def extract_expected_output(self, content: str) -> str:
        content = content or ""
        match = re.search(r"Expected Output:\s*(.*)$", content, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_movement_type(self, value: str) -> str:
        v = self.clean(value).lower()
        if v == "entry":
            return "Entry"
        if v == "exit":
            return "Exit"
        if v == "read":
            return "Read"
        if v == "write":
            return "Write"
        return self.clean(value)

    # ------------------------------------------------------------------
    # Load file
    # ------------------------------------------------------------------

    def load_jsonl(self):
        if not EXAMPLES_JSONL_PATH.exists():
            raise FileNotFoundError(
                f"JSONL file not found: {EXAMPLES_JSONL_PATH}. "
                "Create data/cosmic_examples_chunks.jsonl or set COSMIC_EXAMPLES_JSONL."
            )

        logger.info("Loading examples from %s", EXAMPLES_JSONL_PATH)

        loaded = 0
        skipped = 0

        with open(EXAMPLES_JSONL_PATH, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping invalid JSONL line %s: %s", line_no, e)
                    skipped += 1
                    continue

                if LOAD_ONLY_EXAMPLES and self.clean(item.get("type")).lower() != "example":
                    skipped += 1
                    continue

                self.load_chunk(item)
                loaded += 1

        logger.info("Loaded %s record(s), skipped %s record(s).", loaded, skipped)

    # ------------------------------------------------------------------
    # Main chunk loading
    # ------------------------------------------------------------------

    def load_chunk(self, item: Dict[str, Any]):
        chunk_id = self.clean(item.get("id"))
        if not chunk_id:
            return

        app_domain = self.clean(item.get("app_domain")) or "general"
        component_domain = self.clean(item.get("domain")) or "general"
        section = self.clean(item.get("section"))
        record_type = self.clean(item.get("type"))
        content = self.clean(item.get("content"))
        keywords = self.normalize_list(item.get("keywords"))

        fur_text = self.extract_input_text(content)

        self.run(
            """
            MERGE (chunk:ExampleKnowledge {id: $chunk_id})
            SET chunk.section = $section,
                chunk.type = $record_type,
                chunk.component_domain = $component_domain,
                chunk.app_domain = $app_domain,
                chunk.content = $content,
                chunk.keywords = $keywords

            MERGE (domain:Domain {name: $app_domain})
            MERGE (chunk)-[:BELONGS_TO_DOMAIN]->(domain)

            MERGE (component:ComponentType {name: $component_domain})
            MERGE (chunk)-[:APPLIES_TO]->(component)

            MERGE (fur:FURExample {id: $chunk_id})
            SET fur.text = $fur_text,
                fur.app_domain = $app_domain,
                fur.component_domain = $component_domain

            MERGE (fur)-[:DERIVED_FROM_CHUNK]->(chunk)
            MERGE (fur)-[:BELONGS_TO_DOMAIN]->(domain)
            """,
            {
                "chunk_id": chunk_id,
                "section": section,
                "record_type": record_type,
                "component_domain": component_domain,
                "app_domain": app_domain,
                "content": content,
                "keywords": keywords,
                "fur_text": fur_text,
            }
        )

        if component_domain == "functional_users":
            self.load_functional_users_example(chunk_id, content)

        elif component_domain == "functional_processes":
            self.load_functional_process_example(chunk_id, content)

        elif component_domain == "data_groups":
            self.load_data_groups_example(chunk_id, content)

        elif component_domain in {"sub_processes", "data_movements"}:
            self.load_sub_processes_example(chunk_id, content)

    # ------------------------------------------------------------------
    # Functional Users examples
    # ------------------------------------------------------------------

    def load_functional_users_example(self, chunk_id: str, content: str):
        expected = self.extract_expected_output(content)
        match = re.search(r"Functional Users:\s*(.*)$", expected, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            return

        users = re.findall(r"^\s*-\s*(.+?)\s*$", match.group(1), flags=re.MULTILINE)

        for fu in users:
            fu = self.clean(fu)
            if not fu:
                continue

            self.run(
                """
                MATCH (fur:FURExample {id: $chunk_id})
                MERGE (fu:ExampleFunctionalUser {id: $fu_id})
                SET fu.name = $fu_name
                MERGE (fur)-[:HAS_FUNCTIONAL_USER]->(fu)
                """,
                {
                    "chunk_id": chunk_id,
                    "fu_id": f"{chunk_id}::FU::{fu}",
                    "fu_name": fu
                }
            )

    # ------------------------------------------------------------------
    # Functional Process examples
    # ------------------------------------------------------------------

    def load_functional_process_example(self, chunk_id: str, content: str):
        expected = self.extract_expected_output(content)

        fp_name = self.extract_field(expected, "Functional Process")
        triggering_event = self.extract_field(expected, "Triggering Event")
        functional_user = self.extract_field(expected, "Functional User")
        triggering_entry = self.extract_field(expected, "Triggering Entry")

        if not fp_name:
            return

        self.run(
            """
            MATCH (fur:FURExample {id: $chunk_id})

            MERGE (fp:ExampleFunctionalProcess {id: $fp_id})
            SET fp.name = $fp_name,
                fp.triggering_event = $triggering_event,
                fp.functional_user = $functional_user,
                fp.triggering_entry = $triggering_entry

            MERGE (fur)-[:HAS_FUNCTIONAL_PROCESS]->(fp)

            MERGE (fu:ExampleFunctionalUser {id: $fu_id})
            SET fu.name = $functional_user
            MERGE (fur)-[:HAS_FUNCTIONAL_USER]->(fu)
            MERGE (fp)-[:PERFORMED_BY]->(fu)

            MERGE (te:ExampleTriggeringEntry {id: $te_id})
            SET te.text = $triggering_entry
            MERGE (fp)-[:INITIATED_BY]->(te)
            """,
            {
                "chunk_id": chunk_id,
                "fp_id": f"{chunk_id}::FP::{fp_name}",
                "fp_name": fp_name,
                "triggering_event": triggering_event,
                "functional_user": functional_user,
                "triggering_entry": triggering_entry,
                "fu_id": f"{chunk_id}::FU::{functional_user}",
                "te_id": f"{chunk_id}::TE::{triggering_entry or fp_name}",
            }
        )

    def extract_field(self, text: str, field_name: str) -> str:
        pattern = rf"{re.escape(field_name)}:\s*(.+)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    # ------------------------------------------------------------------
    # Data Group examples
    # ------------------------------------------------------------------

    def load_data_groups_example(self, chunk_id: str, content: str):
        expected = self.extract_expected_output(content)

        # Matches lines like:
        # - Professor: professor_id, name, address
        # - Heater: Heater On/Off command
        lines = re.findall(r"^\s*-\s*([^:\n]+):\s*(.+?)\s*$", expected, flags=re.MULTILINE)

        for name, attrs_text in lines:
            dg_name = self.clean(name)
            attributes = [self.clean(x) for x in re.split(r",\s*", attrs_text) if self.clean(x)]

            if not dg_name:
                continue

            dg_id = f"{chunk_id}::DG::{dg_name}"
            ooi_id = f"{chunk_id}::OOI::{dg_name}"

            self.run(
                """
                MATCH (fur:FURExample {id: $chunk_id})

                MERGE (dg:ExampleDataGroup {id: $dg_id})
                SET dg.name = $dg_name

                MERGE (ooi:ExampleObjectOfInterest {id: $ooi_id})
                SET ooi.name = $ooi_name

                MERGE (fur)-[:HAS_DATA_GROUP]->(dg)
                MERGE (dg)-[:DESCRIBES]->(ooi)
                """,
                {
                    "chunk_id": chunk_id,
                    "dg_id": dg_id,
                    "dg_name": dg_name,
                    "ooi_id": ooi_id,
                    "ooi_name": dg_name
                }
            )

            for attr in attributes:
                self.run(
                    """
                    MATCH (dg:ExampleDataGroup {id: $dg_id})
                    MERGE (a:ExampleDataAttribute {id: $attr_id})
                    SET a.name = $attr
                    MERGE (dg)-[:HAS_ATTRIBUTE]->(a)
                    """,
                    {
                        "dg_id": dg_id,
                        "attr_id": f"{dg_id}::ATTR::{attr}",
                        "attr": attr
                    }
                )

    # ------------------------------------------------------------------
    # Sub-process examples
    # ------------------------------------------------------------------

    def load_sub_processes_example(self, chunk_id: str, content: str):
        expected = self.extract_expected_output(content)

        # Split on "- Sub-process:"
        blocks = re.split(r"(?=^\s*-\s*Sub-process:\s*)", expected, flags=re.MULTILINE)

        for block in blocks:
            if "Sub-process:" not in block:
                continue

            step_name = self.extract_subprocess_step_name(block)
            functional_user = self.extract_field(block, "Functional User")
            moved_data_group = self.extract_field(block, "Moved Data Group")
            object_of_interest = self.extract_field(block, "Object of Interest")
            source = self.extract_field(block, "Source")
            destination = self.extract_field(block, "Destination")
            movement_type = self.normalize_movement_type(self.extract_field(block, "Type"))

            if not step_name:
                continue

            sp_id = f"{chunk_id}::SP::{step_name}"

            self.run(
                """
                MATCH (fur:FURExample {id: $chunk_id})

                MERGE (sp:ExampleSubProcess {id: $sp_id})
                SET sp.step_name = $step_name,
                    sp.functional_user = $functional_user,
                    sp.moved_data_group = $moved_data_group,
                    sp.object_of_interest = $object_of_interest,
                    sp.source = $source,
                    sp.destination = $destination,
                    sp.movement_type = $movement_type

                MERGE (fur)-[:HAS_SUB_PROCESS]->(sp)

                MERGE (dg:ExampleDataGroup {id: $dg_id})
                SET dg.name = $moved_data_group
                MERGE (sp)-[:MOVES_DATA_GROUP]->(dg)

                MERGE (ooi:ExampleObjectOfInterest {id: $ooi_id})
                SET ooi.name = $object_of_interest
                MERGE (dg)-[:DESCRIBES]->(ooi)

                MERGE (fu:ExampleFunctionalUser {id: $fu_id})
                SET fu.name = $functional_user
                MERGE (fur)-[:HAS_FUNCTIONAL_USER]->(fu)
                """,
                {
                    "chunk_id": chunk_id,
                    "sp_id": sp_id,
                    "step_name": step_name,
                    "functional_user": functional_user,
                    "moved_data_group": moved_data_group,
                    "object_of_interest": object_of_interest,
                    "source": source,
                    "destination": destination,
                    "movement_type": movement_type,
                    "dg_id": f"{chunk_id}::DG::{moved_data_group}",
                    "ooi_id": f"{chunk_id}::OOI::{object_of_interest}",
                    "fu_id": f"{chunk_id}::FU::{functional_user}",
                }
            )

            # Link to core MovementType if it exists
            if movement_type:
                self.run(
                    """
                    MATCH (sp:ExampleSubProcess {id: $sp_id})
                    MATCH (mt:MovementType {name: $movement_type})
                    MERGE (sp)-[:CLASSIFIED_AS]->(mt)
                    """,
                    {
                        "sp_id": sp_id,
                        "movement_type": movement_type
                    }
                )

    def extract_subprocess_step_name(self, block: str) -> str:
        match = re.search(r"Sub-process:\s*(.+)", block, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self):
        queries = {
            "Example chunks": "MATCH (n:ExampleKnowledge) RETURN count(n) AS count",
            "FUR examples": "MATCH (n:FURExample) RETURN count(n) AS count",
            "Example functional users": "MATCH (n:ExampleFunctionalUser) RETURN count(n) AS count",
            "Example functional processes": "MATCH (n:ExampleFunctionalProcess) RETURN count(n) AS count",
            "Example data groups": "MATCH (n:ExampleDataGroup) RETURN count(n) AS count",
            "Example sub-processes": "MATCH (n:ExampleSubProcess) RETURN count(n) AS count",
            "SP classified as movements": "MATCH (:ExampleSubProcess)-[:CLASSIFIED_AS]->(:MovementType) RETURN count(*) AS count",
        }

        with self.driver.session() as session:
            print("\n=== COSMIC example graph verification ===")
            for label, query in queries.items():
                result = session.run(query).single()
                print(f"{label}: {result['count']}")

    def load_all(self):
        self.create_constraints()
        self.load_jsonl()
        self.verify()


if __name__ == "__main__":
    loader = CosmicExamplesGraphLoader()

    try:
        loader.load_all()
        print("\nCOSMIC examples loaded successfully.")

    finally:
        loader.close()
