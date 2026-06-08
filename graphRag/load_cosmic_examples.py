import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
EXAMPLES_JSONL_PATH = Path(
    os.getenv("COSMIC_EXAMPLES_JSONL", "rag_docs/cosmic_examples_chunks.jsonl")
)

LOAD_ONLY_EXAMPLES = os.getenv("LOAD_ONLY_EXAMPLES", "true").lower() == "true"
RESET_EXAMPLES = os.getenv("RESET_COSMIC_EXAMPLES", "true").lower() == "true"

if not NEO4J_PASSWORD:
    raise RuntimeError("Missing NEO4J_PASSWORD in .env")


class CosmicExamplesGraphLoader:
    """Load COSMIC examples into Neo4j and connect them to the core ontology.

    The loader supports the current text-oriented JSONL format and groups chunks
    containing the same requirement text under one FURExample. Example instances
    remain separate from ComponentType ontology nodes and are connected through
    INSTANCE_OF relationships.
    """

    COMPONENT_TYPE_MAP = {
        "functional_users": "FunctionalUser",
        "functional_processes": "FunctionalProcess",
        "data_groups": "DataGroup",
        "sub_processes": "SubProcess",
        "data_movements": "DataMovement",
    }

    MISSING_VALUES = {
        "",
        "-",
        "n/a",
        "na",
        "none",
        "null",
        "not applicable",
        "unknown",
    }

    MOVEMENT_MAP = {
        "entry": "Entry",
        "e": "Entry",
        "exit": "Exit",
        "x": "Exit",
        "read": "Read",
        "r": "Read",
        "write": "Write",
        "w": "Write",
    }

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, params: Optional[dict] = None) -> None:
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run(query, params or {}).consume()

    def fetch_one(self, query: str, params: Optional[dict] = None) -> Optional[dict]:
        with self.driver.session(database=NEO4J_DATABASE) as session:
            record = session.run(query, params or {}).single()
            return dict(record) if record else None

    # ------------------------------------------------------------------
    # DATABASE PREPARATION
    # ------------------------------------------------------------------

    def reset_example_graph(self) -> None:
        """Delete only example data and preserve the COSMIC core ontology."""
        if not RESET_EXAMPLES:
            return

        logger.warning("Deleting existing COSMIC example graph...")
        self.run(
            """
            MATCH (n)
            WHERE n:ExampleKnowledge
               OR n:FURExample
               OR n:Domain
               OR n:ExampleFunctionalUser
               OR n:ExampleFunctionalProcess
               OR n:ExampleTriggeringEvent
               OR n:ExampleTriggeringEntry
               OR n:ExampleDataGroup
               OR n:ExampleObjectOfInterest
               OR n:ExampleDataAttribute
               OR n:ExampleSubProcess
               OR n:ExampleEndpoint
            DETACH DELETE n
            """
        )

    def create_constraints(self) -> None:
        logger.info("Creating example graph constraints...")

        constraints = [
            "CREATE CONSTRAINT example_knowledge_id IF NOT EXISTS FOR (n:ExampleKnowledge) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT fur_example_id IF NOT EXISTS FOR (n:FURExample) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT example_domain_name IF NOT EXISTS FOR (n:Domain) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT example_fu_id IF NOT EXISTS FOR (n:ExampleFunctionalUser) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT example_fp_id IF NOT EXISTS FOR (n:ExampleFunctionalProcess) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT example_event_id IF NOT EXISTS FOR (n:ExampleTriggeringEvent) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT example_entry_id IF NOT EXISTS FOR (n:ExampleTriggeringEntry) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT example_dg_id IF NOT EXISTS FOR (n:ExampleDataGroup) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT example_ooi_id IF NOT EXISTS FOR (n:ExampleObjectOfInterest) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT example_attr_id IF NOT EXISTS FOR (n:ExampleDataAttribute) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT example_sp_id IF NOT EXISTS FOR (n:ExampleSubProcess) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT example_endpoint_id IF NOT EXISTS FOR (n:ExampleEndpoint) REQUIRE n.id IS UNIQUE",
        ]

        for query in constraints:
            self.run(query)

    def validate_core_ontology(self) -> None:
        required_component_types = sorted(set(self.COMPONENT_TYPE_MAP.values()))
        required_component_types.extend(
            ["FUR", "TriggeringEvent", "TriggeringEntry", "ObjectOfInterest", "DataAttribute"]
        )
        required_movements = ["Entry", "Exit", "Read", "Write"]

        component_result = self.fetch_one(
            """
            UNWIND $names AS name
            OPTIONAL MATCH (component:ComponentType {name: name})
            WITH name, count(component) AS matches
            WHERE matches = 0
            RETURN collect(name) AS missing
            """,
            {"names": required_component_types},
        )
        movement_result = self.fetch_one(
            """
            UNWIND $names AS name
            OPTIONAL MATCH (movement:MovementType {name: name})
            WITH name, count(movement) AS matches
            WHERE matches = 0
            RETURN collect(name) AS missing
            """,
            {"names": required_movements},
        )

        missing_components = (component_result or {}).get("missing", [])
        missing_movements = (movement_result or {}).get("missing", [])
        if missing_components or missing_movements:
            raise RuntimeError(
                "COSMIC core ontology is incomplete. "
                f"Missing ComponentType nodes: {missing_components}; "
                f"missing MovementType nodes: {missing_movements}. "
                "Run load_cosmic_core_ontology.py first."
            )

    # ------------------------------------------------------------------
    # NORMALIZATION AND PARSING
    # ------------------------------------------------------------------

    @staticmethod
    def clean(value: Any) -> str:
        return str(value or "").strip()

    def normalize_list(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [self.clean(item) for item in value if self.clean(item)]
        cleaned = self.clean(value)
        return [cleaned] if cleaned else []

    def is_missing(self, value: Any) -> bool:
        return self.clean(value).casefold() in self.MISSING_VALUES

    @staticmethod
    def normalize_text(value: str) -> str:
        value = value.replace("’", "'").replace("‘", "'")
        return re.sub(r"\s+", " ", value).strip().casefold()

    def make_fur_id(self, item: Dict[str, Any], fur_text: str) -> str:
        """Group chunks belonging to the same Functional Process."""
        app_domain = self.clean(item.get("app_domain"))
        functional_process = self.clean(item.get("functional_process"))

        identity = "::".join(
            [
                self.normalize_text(app_domain),
                self.normalize_text(functional_process),
            ]
        )

        if not identity.replace(":", ""):
            identity = self.normalize_text(fur_text)

        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return f"fur::{digest}"

    def extract_input_text(self, content: str) -> str:
        match = re.search(
            r"Input:\s*(.*?)(?:\n\s*Expected Output:|$)",
            content or "",
            flags=re.DOTALL | re.IGNORECASE,
        )
        return match.group(1).strip() if match else self.clean(content)

    def extract_expected_output(self, content: str) -> str:
        match = re.search(
            r"Expected Output:\s*(.*)$",
            content or "",
            flags=re.DOTALL | re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def extract_field(text: str, field_name: str) -> str:
        match = re.search(
            rf"^\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    def normalize_movement_type(self, value: str) -> Optional[str]:
        raw = self.clean(value)
        normalized = self.MOVEMENT_MAP.get(raw.casefold())
        if not normalized:
            logger.warning("Unknown COSMIC movement type: %r", value)
        return normalized

    def read_records(self) -> List[Dict[str, Any]]:
        if not EXAMPLES_JSONL_PATH.exists():
            raise FileNotFoundError(
                f"JSONL file not found: {EXAMPLES_JSONL_PATH}. "
                "Set COSMIC_EXAMPLES_JSONL or place the file under rag_docs/."
            )

        records: List[Dict[str, Any]] = []
        with EXAMPLES_JSONL_PATH.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSONL line %s: %s", line_no, exc)
                    continue

                record_type = self.clean(item.get("type")).casefold()
                if LOAD_ONLY_EXAMPLES and record_type and record_type != "example":
                    continue

                if not self.clean(item.get("id")):
                    logger.warning("Skipping line %s because id is missing", line_no)
                    continue

                records.append(item)

        return records

    # ------------------------------------------------------------------
    # BASE CHUNKS AND SHARED FUR NODES
    # ------------------------------------------------------------------

    def load_base_chunk(self, item: Dict[str, Any]) -> str:
        chunk_id = self.clean(item["id"])
        app_domain = self.clean(item.get("app_domain")) or "general"
        cosmic_component = self.clean(item.get("cosmic_component")) or "general"
        functional_process = self.clean(item.get("functional_process"))
        content = self.clean(item.get("content"))
        fur_text = self.extract_input_text(content)
        fur_id = self.make_fur_id(item, fur_text)

        ontology_component = self.COMPONENT_TYPE_MAP.get(cosmic_component)

        params = {
            "chunk_id": chunk_id,
            "fur_id": fur_id,
            "cosmic_component": cosmic_component,
            "functional_process": functional_process,
            "app_domain": app_domain,
            "content": content,
            "fur_text": fur_text,
            "ontology_component": ontology_component,
        }

        self.run(
            """
            MERGE (chunk:ExampleKnowledge {id: $chunk_id})
            SET chunk.cosmic_component = $cosmic_component,
                chunk.functional_process = $functional_process,
                chunk.app_domain = $app_domain,
                chunk.content = $content

            MERGE (domain:Domain {name: $app_domain})
            MERGE (chunk)-[:BELONGS_TO_DOMAIN]->(domain)

            MERGE (fur:FURExample {id: $fur_id})
            ON CREATE SET fur.text = $fur_text
            SET fur.app_domain = $app_domain,
                fur.functional_process = $functional_process

            MERGE (fur)-[:DERIVED_FROM_CHUNK]->(chunk)
            MERGE (fur)-[:BELONGS_TO_DOMAIN]->(domain)

            WITH chunk, fur
            MATCH (fur_type:ComponentType {name: 'FUR'})
            MERGE (fur)-[:INSTANCE_OF]->(fur_type)
            """,
            params,
        )

        if ontology_component:
            self.run(
                """
                MATCH (chunk:ExampleKnowledge {id: $chunk_id})
                MATCH (component:ComponentType {name: $ontology_component})
                MERGE (chunk)-[:APPLIES_TO]->(component)
                """,
                params,
            )
        else:
            logger.warning(
                "Chunk %s has unsupported cosmic_component %r",
                chunk_id,
                cosmic_component,
            )

        return fur_id

    # ------------------------------------------------------------------
    # FUNCTIONAL USERS
    # ------------------------------------------------------------------

    def parse_functional_users(self, content: str) -> List[str]:
        expected = self.extract_expected_output(content)
        match = re.search(
            r"Functional Users\s*:\s*(.*?)(?=\n\s*\n|\n\s*[A-Z][^\n:]*:|$)",
            expected,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not match:
            return []
        return [
            self.clean(user)
            for user in re.findall(r"^\s*-\s*(.+?)\s*$", match.group(1), flags=re.MULTILINE)
            if not self.is_missing(user)
        ]

    def ensure_functional_user(self, fur_id: str, name: str) -> Optional[str]:
        if self.is_missing(name):
            return None

        user_id = f"{fur_id}::FU::{self.normalize_text(name)}"
        self.run(
            """
            MATCH (fur:FURExample {id: $fur_id})
            MATCH (type:ComponentType {name: 'FunctionalUser'})
            MERGE (fu:ExampleFunctionalUser {id: $user_id})
            SET fu.name = $name
            MERGE (fur)-[:HAS_FUNCTIONAL_USER]->(fu)
            MERGE (fu)-[:INSTANCE_OF]->(type)
            """,
            {"fur_id": fur_id, "user_id": user_id, "name": name},
        )
        return user_id

    def load_functional_users_example(self, fur_id: str, content: str) -> None:
        for user in self.parse_functional_users(content):
            self.ensure_functional_user(fur_id, user)

    # ------------------------------------------------------------------
    # FUNCTIONAL PROCESSES
    # ------------------------------------------------------------------

    def load_functional_process_example(self, fur_id: str, content: str) -> None:
        expected = self.extract_expected_output(content)
        fp_name = self.extract_field(expected, "Functional Process")
        triggering_event = self.extract_field(expected, "Triggering Event")
        functional_user = self.extract_field(expected, "Functional User")
        triggering_entry = self.extract_field(expected, "Triggering Entry")

        if not fp_name:
            logger.warning("Functional-process chunk has no Functional Process field")
            return

        fp_id = f"{fur_id}::FP::{self.normalize_text(fp_name)}"
        user_id = self.ensure_functional_user(fur_id, functional_user)
        event_id = f"{fp_id}::EVENT"
        entry_id = f"{fp_id}::TRIGGERING_ENTRY"

        self.run(
            """
            MATCH (fur:FURExample {id: $fur_id})
            MATCH (fp_type:ComponentType {name: 'FunctionalProcess'})
            MERGE (fp:ExampleFunctionalProcess {id: $fp_id})
            SET fp.name = $fp_name
            MERGE (fur)-[:HAS_FUNCTIONAL_PROCESS]->(fp)
            MERGE (fp)-[:INSTANCE_OF]->(fp_type)

            WITH fur, fp
            MATCH (event_type:ComponentType {name: 'TriggeringEvent'})
            MERGE (event:ExampleTriggeringEvent {id: $event_id})
            SET event.description = $triggering_event
            MERGE (event)-[:TRIGGERS]->(fp)
            MERGE (event)-[:INSTANCE_OF]->(event_type)

            WITH fur, fp, event
            MATCH (entry_type:ComponentType {name: 'TriggeringEntry'})
            MERGE (entry:ExampleTriggeringEntry {id: $entry_id})
            SET entry.text = $triggering_entry
            MERGE (fp)-[:INITIATED_BY]->(entry)
            MERGE (entry)-[:INSTANCE_OF]->(entry_type)
            """,
            {
                "fur_id": fur_id,
                "fp_id": fp_id,
                "fp_name": fp_name,
                "event_id": event_id,
                "triggering_event": triggering_event,
                "entry_id": entry_id,
                "triggering_entry": triggering_entry,
            },
        )

        if user_id:
            self.run(
                """
                MATCH (fp:ExampleFunctionalProcess {id: $fp_id})
                MATCH (event:ExampleTriggeringEvent {id: $event_id})
                MATCH (fu:ExampleFunctionalUser {id: $user_id})
                MERGE (fp)-[:TRIGGERED_BY_USER]->(fu)
                MERGE (event)-[:CAUSES_RESPONSE_FROM]->(fu)
                """,
                {"fp_id": fp_id, "event_id": event_id, "user_id": user_id},
            )

    # ------------------------------------------------------------------
    # DATA GROUPS
    # ------------------------------------------------------------------

    def parse_data_groups(self, content: str) -> Iterable[Dict[str, Any]]:
        expected = self.extract_expected_output(content)
        for name, attrs_text in re.findall(
            r"^\s*-\s*([^:\n]+):\s*([^\n]+)",
            expected,
            flags=re.MULTILINE,
        ):
            ooi_name = self.clean(name)
            if "objects of interest" in ooi_name.casefold():
                continue
            attributes = [
                self.clean(item)
                for item in re.split(r",\s*", attrs_text)
                if self.clean(item)
            ]
            yield {
                "name": f"{ooi_name} data",
                "object_of_interest": ooi_name,
                "attributes": attributes,
            }

    def ensure_data_group(
        self,
        fur_id: str,
        dg_name: str,
        ooi_name: str,
        attributes: Optional[List[str]] = None,
    ) -> str:
        normalized_dg = self.normalize_text(dg_name or ooi_name)
        normalized_ooi = self.normalize_text(ooi_name or dg_name)
        dg_id = f"{fur_id}::DG::{normalized_dg}"
        ooi_id = f"{fur_id}::OOI::{normalized_ooi}"

        self.run(
            """
            MATCH (fur:FURExample {id: $fur_id})
            MATCH (dg_type:ComponentType {name: 'DataGroup'})
            MATCH (ooi_type:ComponentType {name: 'ObjectOfInterest'})
            MERGE (dg:ExampleDataGroup {id: $dg_id})
            SET dg.name = $dg_name
            MERGE (ooi:ExampleObjectOfInterest {id: $ooi_id})
            SET ooi.name = $ooi_name
            MERGE (fur)-[:HAS_DATA_GROUP]->(dg)
            MERGE (dg)-[:DESCRIBES]->(ooi)
            MERGE (dg)-[:INSTANCE_OF]->(dg_type)
            MERGE (ooi)-[:INSTANCE_OF]->(ooi_type)
            """,
            {
                "fur_id": fur_id,
                "dg_id": dg_id,
                "dg_name": dg_name,
                "ooi_id": ooi_id,
                "ooi_name": ooi_name,
            },
        )

        for attribute in attributes or []:
            attr_id = f"{dg_id}::ATTR::{self.normalize_text(attribute)}"
            self.run(
                """
                MATCH (dg:ExampleDataGroup {id: $dg_id})
                MATCH (attr_type:ComponentType {name: 'DataAttribute'})
                MERGE (attr:ExampleDataAttribute {id: $attr_id})
                SET attr.name = $attribute
                MERGE (dg)-[:HAS_ATTRIBUTE]->(attr)
                MERGE (attr)-[:INSTANCE_OF]->(attr_type)
                """,
                {"dg_id": dg_id, "attr_id": attr_id, "attribute": attribute},
            )

        return dg_id

    def load_data_groups_example(self, fur_id: str, content: str) -> None:
        for data_group in self.parse_data_groups(content):
            self.ensure_data_group(
                fur_id,
                data_group["name"],
                data_group["object_of_interest"],
                data_group["attributes"],
            )

    # ------------------------------------------------------------------
    # SUB-PROCESSES AND DATA MOVEMENTS
    # ------------------------------------------------------------------

    def parse_sub_processes(self, content: str) -> List[Dict[str, str]]:
        expected = self.extract_expected_output(content)
        parsed: List[Dict[str, str]] = []
        current: Dict[str, str] = {}

        for raw_line in expected.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if re.match(r"^-?\s*Sub-process\s*:", line, flags=re.IGNORECASE):
                if current.get("step_name"):
                    parsed.append(current)
                current = {
                    "step_name": re.sub(
                        r"^-?\s*Sub-process\s*:\s*",
                        "",
                        line,
                        flags=re.IGNORECASE,
                    ).strip()
                }
                continue

            if current:
                match = re.match(r"^([^:]+):\s*(.*)$", line)
                if match:
                    key = re.sub(r"\s+", "_", match.group(1).strip().casefold())
                    current[key] = match.group(2).strip()

        if current.get("step_name"):
            parsed.append(current)

        return parsed

    def resolve_functional_process_id(
        self,
        fur_id: str,
        explicit_name: str = "",
    ) -> Optional[str]:
        """Resolve the FP for a sub-process chunk.

        Priority:
        1. Match the explicit functional_process name from the JSONL.
        2. Create the FP when the JSONL explicitly names it but no FP chunk exists.
        3. Fall back to the only FP attached to the FUR.
        """
        explicit_name = self.clean(explicit_name)

        if explicit_name:
            fp_id = f"{fur_id}::FP::{self.normalize_text(explicit_name)}"
            self.run(
                """
                MATCH (fur:FURExample {id: $fur_id})
                MATCH (fp_type:ComponentType {name: 'FunctionalProcess'})
                MERGE (fp:ExampleFunctionalProcess {id: $fp_id})
                ON CREATE SET fp.name = $fp_name,
                              fp.inferred_from_subprocess_chunk = true
                ON MATCH SET fp.name = CASE
                    WHEN fp.name IS NULL OR trim(fp.name) = '' THEN $fp_name
                    ELSE fp.name
                END
                MERGE (fur)-[:HAS_FUNCTIONAL_PROCESS]->(fp)
                MERGE (fp)-[:INSTANCE_OF]->(fp_type)
                """,
                {
                    "fur_id": fur_id,
                    "fp_id": fp_id,
                    "fp_name": explicit_name,
                },
            )
            return fp_id

        result = self.fetch_one(
            """
            MATCH (:FURExample {id: $fur_id})
                  -[:HAS_FUNCTIONAL_PROCESS]->
                  (fp:ExampleFunctionalProcess)
            RETURN collect(fp.id) AS ids
            """,
            {"fur_id": fur_id},
        )
        ids = (result or {}).get("ids", [])

        if len(ids) == 1:
            return ids[0]

        if len(ids) > 1:
            logger.warning(
                "FUR %s has multiple functional processes; "
                "add a functional_process field to the JSONL chunk",
                fur_id,
            )
        else:
            logger.warning(
                "FUR %s has no functional process; "
                "add a functional_process field to the JSONL chunk",
                fur_id,
            )

        return None

    def endpoint_kind(self, movement_type: str, is_source: bool) -> str:
        if movement_type == "Entry":
            return "FunctionalUser" if is_source else "FunctionalProcess"
        if movement_type == "Exit":
            return "FunctionalProcess" if is_source else "FunctionalUser"
        if movement_type == "Read":
            return "PersistentStorage" if is_source else "FunctionalProcess"
        if movement_type == "Write":
            return "FunctionalProcess" if is_source else "PersistentStorage"
        return "Unknown"

    def ensure_endpoint(
        self,
        fur_id: str,
        name: str,
        kind: str,
        functional_user_id: Optional[str] = None,
    ) -> str:
        endpoint_id = f"{fur_id}::ENDPOINT::{kind}::{self.normalize_text(name)}"
        self.run(
            """
            MERGE (endpoint:ExampleEndpoint {id: $endpoint_id})
            SET endpoint.name = $name,
                endpoint.kind = $kind
            """,
            {"endpoint_id": endpoint_id, "name": name, "kind": kind},
        )

        if functional_user_id and kind == "FunctionalUser":
            self.run(
                """
                MATCH (endpoint:ExampleEndpoint {id: $endpoint_id})
                MATCH (fu:ExampleFunctionalUser {id: $functional_user_id})
                MERGE (endpoint)-[:REPRESENTS]->(fu)
                """,
                {
                    "endpoint_id": endpoint_id,
                    "functional_user_id": functional_user_id,
                },
            )

        return endpoint_id

    def load_sub_processes_example(
        self,
        fur_id: str,
        content: str,
        functional_process_name: str = "",
    ) -> None:
        fp_id = self.resolve_functional_process_id(
            fur_id,
            explicit_name=functional_process_name,
        )
        for sp_data in self.parse_sub_processes(content):
            self.save_subprocess(fur_id, fp_id, sp_data)

    def save_subprocess(
        self,
        fur_id: str,
        fp_id: Optional[str],
        sp_data: Dict[str, str],
    ) -> None:
        step_name = self.clean(sp_data.get("step_name"))
        functional_user = self.clean(sp_data.get("functional_user"))
        moved_data_group = self.clean(sp_data.get("moved_data_group"))
        object_of_interest = self.clean(sp_data.get("object_of_interest"))
        source = self.clean(sp_data.get("source"))
        destination = self.clean(sp_data.get("destination"))
        movement_type = self.normalize_movement_type(sp_data.get("type", ""))

        if not step_name or not movement_type:
            logger.warning("Skipping invalid sub-process: %s", sp_data)
            return
        if self.is_missing(moved_data_group) or self.is_missing(object_of_interest):
            logger.warning("Skipping SP %r because DG or OOI is missing", step_name)
            return

        # Correct a frequent dataset mistake: for Exit, the external destination
        # is the functional user, not the measured system itself.
        if movement_type == "Exit" and functional_user.casefold() == "system":
            if not self.is_missing(destination) and destination.casefold() != "system":
                logger.warning(
                    "Correcting Functional User 'System' to Exit destination %r for SP %r",
                    destination,
                    step_name,
                )
                functional_user = destination

        fu_id = self.ensure_functional_user(fur_id, functional_user)
        dg_id = self.ensure_data_group(
            fur_id,
            moved_data_group,
            object_of_interest,
        )
        sp_id = f"{fur_id}::SP::{self.normalize_text(step_name)}"

        self.run(
            """
            MATCH (fur:FURExample {id: $fur_id})
            MATCH (sp_type:ComponentType {name: 'SubProcess'})
            MATCH (dg:ExampleDataGroup {id: $dg_id})
            MATCH (mt:MovementType {name: $movement_type})

            MERGE (sp:ExampleSubProcess {id: $sp_id})
            SET sp.name = $step_name,
                sp.movement_type = $movement_type,
                sp.source_text = $source,
                sp.destination_text = $destination

            MERGE (fur)-[:HAS_SUB_PROCESS]->(sp)
            MERGE (sp)-[:INSTANCE_OF]->(sp_type)
            MERGE (sp)-[:MOVES_DATA_GROUP]->(dg)

            WITH sp, mt
            OPTIONAL MATCH (sp)-[old:CLASSIFIED_AS]->(:MovementType)
            DELETE old
            MERGE (sp)-[:CLASSIFIED_AS]->(mt)
            """,
            {
                "fur_id": fur_id,
                "sp_id": sp_id,
                "step_name": step_name,
                "movement_type": movement_type,
                "source": source,
                "destination": destination,
                "dg_id": dg_id,
            },
        )

        if fp_id:
            self.run(
                """
                MATCH (fp:ExampleFunctionalProcess {id: $fp_id})
                MATCH (sp:ExampleSubProcess {id: $sp_id})
                MATCH (dg:ExampleDataGroup {id: $dg_id})
                MERGE (fp)-[:HAS_SUB_PROCESS]->(sp)
                MERGE (fp)-[:USES_DATA_GROUP]->(dg)
                """,
                {"fp_id": fp_id, "sp_id": sp_id, "dg_id": dg_id},
            )

        if fu_id:
            self.run(
                """
                MATCH (sp:ExampleSubProcess {id: $sp_id})
                MATCH (fu:ExampleFunctionalUser {id: $fu_id})
                MERGE (sp)-[:INTERACTS_WITH_USER]->(fu)
                """,
                {"sp_id": sp_id, "fu_id": fu_id},
            )

        source_name = source or self.endpoint_kind(movement_type, True)
        destination_name = destination or self.endpoint_kind(movement_type, False)
        source_kind = self.endpoint_kind(movement_type, True)
        destination_kind = self.endpoint_kind(movement_type, False)

        source_fu_id = fu_id if source_kind == "FunctionalUser" else None
        destination_fu_id = fu_id if destination_kind == "FunctionalUser" else None
        source_id = self.ensure_endpoint(fur_id, source_name, source_kind, source_fu_id)
        destination_id = self.ensure_endpoint(
            fur_id,
            destination_name,
            destination_kind,
            destination_fu_id,
        )

        self.run(
            """
            MATCH (sp:ExampleSubProcess {id: $sp_id})
            MATCH (source:ExampleEndpoint {id: $source_id})
            MATCH (destination:ExampleEndpoint {id: $destination_id})
            MERGE (sp)-[:HAS_SOURCE]->(source)
            MERGE (sp)-[:HAS_DESTINATION]->(destination)
            """,
            {
                "sp_id": sp_id,
                "source_id": source_id,
                "destination_id": destination_id,
            },
        )

    # ------------------------------------------------------------------
    # LOADING ORCHESTRATION
    # ------------------------------------------------------------------

    def load_records(self, records: List[Dict[str, Any]]) -> None:
        fur_ids: Dict[str, str] = {}

        # Pass 1: chunks and shared FURs.
        for item in records:
            fur_ids[self.clean(item["id"])] = self.load_base_chunk(item)

        # Pass 2: FU, FP and DG nodes. FP nodes must exist before SP linking.
        for cosmic_component in (
            "functional_users",
            "functional_processes",
            "data_groups",
        ):
            for item in records:
                if self.clean(item.get("cosmic_component")) != cosmic_component:
                    continue

                fur_id = fur_ids[self.clean(item["id"])]
                content = self.clean(item.get("content"))

                if cosmic_component == "functional_users":
                    self.load_functional_users_example(fur_id, content)
                elif cosmic_component == "functional_processes":
                    self.load_functional_process_example(fur_id, content)
                else:
                    self.load_data_groups_example(fur_id, content)

        # Pass 3: SP/data movements after FP and DG nodes have been loaded.
        for item in records:
            cosmic_component = self.clean(item.get("cosmic_component"))

            if cosmic_component not in {"sub_processes", "data_movements"}:
                continue

            fur_id = fur_ids[self.clean(item["id"])]

            self.load_sub_processes_example(
                fur_id,
                self.clean(item.get("content")),
                functional_process_name=self.clean(
                    item.get("functional_process")
                ),
            )

    # ------------------------------------------------------------------
    # VERIFICATION
    # ------------------------------------------------------------------

    def verify(self) -> None:
        queries = {
            "Example chunks": "MATCH (n:ExampleKnowledge) RETURN count(n) AS count",
            "Unique FUR examples": "MATCH (n:FURExample) RETURN count(n) AS count",
            "Example functional users": "MATCH (n:ExampleFunctionalUser) RETURN count(n) AS count",
            "Example functional processes": "MATCH (n:ExampleFunctionalProcess) RETURN count(n) AS count",
            "Example triggering events": "MATCH (n:ExampleTriggeringEvent) RETURN count(n) AS count",
            "Example data groups": "MATCH (n:ExampleDataGroup) RETURN count(n) AS count",
            "Example sub-processes": "MATCH (n:ExampleSubProcess) RETURN count(n) AS count",
            "SP classified as movements": "MATCH (:ExampleSubProcess)-[:CLASSIFIED_AS]->(:MovementType) RETURN count(*) AS count",
            "SP linked to FP": "MATCH (:ExampleFunctionalProcess)-[:HAS_SUB_PROCESS]->(:ExampleSubProcess) RETURN count(*) AS count",
            "Example ontology links": "MATCH (n)-[:INSTANCE_OF]->(:ComponentType) WHERE n:FURExample OR n:ExampleFunctionalUser OR n:ExampleFunctionalProcess OR n:ExampleTriggeringEvent OR n:ExampleTriggeringEntry OR n:ExampleDataGroup OR n:ExampleObjectOfInterest OR n:ExampleDataAttribute OR n:ExampleSubProcess RETURN count(*) AS count",
        }

        with self.driver.session(database=NEO4J_DATABASE) as session:
            print("\n=== COSMIC example graph verification ===")
            for label, query in queries.items():
                result = session.run(query).single()
                print(f"{label}: {result['count']}")

            invalid_movements = session.run(
                """
                MATCH (sp:ExampleSubProcess)
                WHERE NOT (sp)-[:CLASSIFIED_AS]->(:MovementType)
                RETURN collect(sp.name) AS names
                """
            ).single()["names"]
            if invalid_movements:
                logger.warning("Unclassified sub-processes: %s", invalid_movements)

            orphan_subprocesses = session.run(
                """
                MATCH (sp:ExampleSubProcess)
                WHERE NOT (:ExampleFunctionalProcess)-[:HAS_SUB_PROCESS]->(sp)
                RETURN collect(sp.name) AS names
                """
            ).single()["names"]
            if orphan_subprocesses:
                logger.warning(
                    "Sub-processes not linked to a Functional Process: %s",
                    orphan_subprocesses,
                )

    def load_all(self) -> None:
        logger.info("Using Neo4j database: %s", NEO4J_DATABASE)
        logger.info("Using COSMIC examples file: %s", EXAMPLES_JSONL_PATH.resolve())

        self.validate_core_ontology()
        self.reset_example_graph()
        self.create_constraints()
        records = self.read_records()
        logger.info("Loading %s COSMIC example chunk(s)...", len(records))
        self.load_records(records)
        self.verify()


if __name__ == "__main__":
    loader = CosmicExamplesGraphLoader()
    try:
        loader.load_all()
        print("\nCOSMIC examples loaded successfully.")
    finally:
        loader.close()
