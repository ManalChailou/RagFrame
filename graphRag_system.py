import os
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)
load_dotenv()


class CosmicGraphRAGSystem:
    """ GraphRAG retrieval layer for the COSMIC ontology stored in Neo4j. """

    def __init__(self,uri: Optional[str] = None,user: Optional[str] = None,password: Optional[str] = None,):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")

        if not self.password:
            raise RuntimeError("Missing NEO4J_PASSWORD in .env")

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )

    # Basic Neo4j helpers

    def close(self):
        if self.driver:
            self.driver.close()

    def _run_query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict]:
        params = params or {}
        with self.driver.session() as session:
            result = session.run(cypher, params)
            return [record.data() for record in result]

    def test_connection(self) -> bool:
        try:
            self._run_query("RETURN 1 AS ok")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False

    def get_graph_stats(self) -> Dict:
        query = """
        MATCH (n)
        RETURN labels(n) AS labels, count(n) AS count
        ORDER BY labels
        """

        rows = self._run_query(query)

        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS relationship, count(r) AS count
        ORDER BY relationship
        """

        rel_rows = self._run_query(rel_query)

        return {
            "node_counts": {
                ",".join(row["labels"]): row["count"]
                for row in rows
            },
            "relationship_counts": {
                row["relationship"]: row["count"]
                for row in rel_rows
            }
        }

    # Formatting helpers

    def _format_rule_context(self, records: List[Dict], title: str = "") -> str:
        if not records:
            return ""

        context = "COSMIC knowledge:\n"
        seen = set()

        for r in records:
            rule_title = r.get("rule_title") or ""
            content = r.get("content") or ""

            key = (rule_title, content)
            if key in seen:
                continue
            seen.add(key)

            if rule_title and content:
                context += f"- {rule_title}: {content}\n"
            elif content:
                context += f"- {content}\n"

        return context.strip() + "\n"

    def _format_guideline_context(self, records: List[Dict], title: str = "") -> str:
        if not records:
            return ""

        context = "\nCOSMIC guidance:\n"

        seen = set()

        for r in records:
            guide_id = r.get("guideline_id") or r.get("guideline_title")
            guide_content = r.get("guideline_content") or ""

            if not guide_content or guide_id in seen:
                continue

            seen.add(guide_id)
            context += f"- {guide_content}\n"

        return context.strip() + "\n"

    def _format_movement_context(self, records: List[Dict]) -> str:
        if not records:
            return ""

        context = "\nCOSMIC movement directions:\n"

        for r in records:
            movement = r.get("movement")
            abbr = r.get("abbreviation")
            source = r.get("source")
            destination = r.get("destination")
            size = r.get("size_cfp")

            context += (
                f"- {movement} ({abbr}): "
                f"{source} -> {destination}; "
                f"{size} CFP\n"
            )

        return context.strip() + "\n"

    def _format_operation_patterns(self, records: List[Dict]) -> str:
        if not records:
            return ""

        grouped = {}

        for r in records:
            op = r.get("operation")
            desc = r.get("description")
            movement = r.get("movement")
            order = r.get("sequence_order")

            if op not in grouped:
                grouped[op] = {
                    "description": desc,
                    "movements": []
                }

            if movement:
                grouped[op]["movements"].append((order, movement))

        context = "\nTypical COSMIC operation patterns:\n"

        for op, info in grouped.items():
            movements = sorted(info["movements"], key=lambda x: x[0] or 999)
            movement_names = [m for _, m in movements]

            context += f"- {op}: {movement_names}\n"

        return context.strip() + "\n"

    # Core retrieval methods used by EnhancedPromptDispatcher

    def get_context_for_functional_users(self, requirements: List[str], app_domain: Optional[str] = None) -> str:

        rules_query = """
        MATCH (r:Rule)
        WHERE r.id IN ["RULE_07"]
        RETURN DISTINCT
            r.number AS rule_number,
            r.title AS rule_title,
            r.content AS content,
            r.source AS source
        ORDER BY r.number
        """

        constraints_query = """
        MATCH (cst:Constraint)
        WHERE cst.id IN ["CONSTRAINT_STORAGE_NOT_FUNCTIONAL_USER"]
        RETURN DISTINCT
            cst.name AS constraint_name,
            cst.content AS constraint_content
        """

        guidelines_query = """
        MATCH (g:Guideline)
        WHERE g.id IN ["GUIDE_CLOCK_ENTRY"]
        OPTIONAL MATCH (g)-[:CLARIFIES]->(r:Rule)
        RETURN DISTINCT
            g.id AS guideline_id,
            g.title AS guideline_title,
            g.content AS guideline_content,
            r.number + " - " + r.title AS clarified_rule
        ORDER BY g.title
        """

        rules = self._run_query(rules_query)
        constraints = self._run_query(constraints_query)
        guidelines = self._run_query(guidelines_query)

        context = self._format_rule_context(
            rules,
            ""
        )

        context += "\nCOSMIC constraints:\n"
        for c in constraints:
            context += f"- {c.get('constraint_content')}\n"

        context += self._format_guideline_context(
            guidelines,
            "Functional User Guidelines"
        )

        return context

    def get_context_for_functional_processes(self, requirements: List[str], app_domain: Optional[str] = None) -> str:

        rules_query = """
        MATCH (r:Rule)
        WHERE r.id IN ["RULE_10"]
        RETURN DISTINCT
            r.number AS rule_number,
            r.title AS rule_title,
            r.content AS content,
            r.source AS source
        ORDER BY r.number
        """

        fp_structure_query = """
        MATCH (fp:ComponentType {name: "FunctionalProcess"})
        OPTIONAL MATCH (fp)-[:MUST_BE_INITIATED_BY]->(te)
        OPTIONAL MATCH (fp)-[:MUST_BE_PARTITIONED_INTO]->(dm)
        OPTIONAL MATCH (fp)-[:MUST_HAVE]->(entry)
        OPTIONAL MATCH (fp)-[:MUST_HAVE_AT_LEAST_ONE_OF]->(required)
        RETURN
            fp.name AS component,
            collect(DISTINCT te.name) AS initiated_by,
            collect(DISTINCT dm.name) AS partitioned_into,
            collect(DISTINCT entry.name) AS must_have,
            collect(DISTINCT required.name) AS must_have_at_least_one_of
        """

        guidelines_query = """
        MATCH (g:Guideline)
        WHERE g.id IN ["GUIDE_FP_IDENTIFICATION_CHAIN"]
        OPTIONAL MATCH (g)-[:CLARIFIES]->(r:Rule)
        RETURN DISTINCT
            g.id AS guideline_id,
            g.title AS guideline_title,
            g.content AS guideline_content,
            r.number + " - " + r.title AS clarified_rule
        ORDER BY g.title
        """

        rules = self._run_query(rules_query)
        fp_structure = self._run_query(fp_structure_query)
        guidelines = self._run_query(guidelines_query)

        context = self._format_rule_context(
            rules,
            "GraphRAG Functional Processes Context"
        )

        if fp_structure:
            s = fp_structure[0]
            context += "\nFunctional Process Structural Constraints:\n"
            context += f"- Must be initiated by: {s.get('initiated_by', [])}\n"
            context += f"- Must be partitioned into: {s.get('partitioned_into', [])}\n"
            context += f"- Must have: {s.get('must_have', [])}\n"
            context += f"- Must have at least one of: {s.get('must_have_at_least_one_of', [])}\n"

        context += self._format_guideline_context(
            guidelines,
            "Functional Process Guidelines"
        )

        return context

    def get_context_for_data_groups(self, requirements: List[str], functional_processes: List[Dict], app_domain: Optional[str] = None) -> str:

        rules_query = """
        MATCH (r:Rule)
        WHERE r.id IN ["RULE_11"]
        RETURN DISTINCT
            r.number AS rule_number,
            r.title AS rule_title,
            r.content AS content,
            r.source AS source
        ORDER BY r.number
        """

        dg_structure_query = """
        MATCH (dg:ComponentType {name: "DataGroup"})
        OPTIONAL MATCH (dg)-[:DESCRIBES_EXACTLY_ONE]->(ooi)
        OPTIONAL MATCH (dg)-[:COMPOSED_OF]->(attr)
        RETURN
            dg.name AS component,
            collect(DISTINCT ooi.name) AS describes,
            collect(DISTINCT attr.name) AS composed_of
        """

        guidelines_query = """
        MATCH (g:Guideline)
        WHERE g.id IN [
            "GUIDE_DG_DIFFERENT_FREQUENCY",
            "GUIDE_DG_DIFFERENT_KEYS"
        ]
        OPTIONAL MATCH (g)-[:CLARIFIES]->(r:Rule)
        RETURN DISTINCT
            g.id AS guideline_id,
            g.title AS guideline_title,
            g.content AS guideline_content,
            r.number + " - " + r.title AS clarified_rule
        ORDER BY g.title
        """

        rules = self._run_query(rules_query)
        dg_structure = self._run_query(dg_structure_query)
        guidelines = self._run_query(guidelines_query)

        context = self._format_rule_context(
            rules,
            "GraphRAG Data Groups Context"
        )

        if dg_structure:
            s = dg_structure[0]
            context += "\nData Group Structural Constraints:\n"
            context += f"- DataGroup describes exactly one: {s.get('describes', [])}\n"
            context += f"- DataGroup is composed of: {s.get('composed_of', [])}\n"

        context += self._format_guideline_context(
            guidelines,
            "Data Group Guidelines"
        )

        return context

    def get_context_for_sub_processes(self,requirements: List[str], functional_processes: List[Dict], data_groups: List[Dict], app_domain: Optional[str] = None ) -> str:

        rules_query = """
        MATCH (r:Rule)
        WHERE r.id IN [
            "RULE_12",
            "RULE_13",
            "RULE_14",
            "RULE_15",
            "RULE_16",
            "RULE_17",
            "RULE_18",
            "RULE_19",
            "RULE_20",
            "RULE_21"
        ]
        RETURN DISTINCT
            r.number AS rule_number,
            r.title AS rule_title,
            r.content AS content,
            r.source AS source
        ORDER BY r.number
        """

        movements_query = """
        MATCH (m:MovementType)-[:HAS_SOURCE]->(src:EntityType),
            (m)-[:HAS_DESTINATION]->(dst:EntityType)
        RETURN
            m.name AS movement,
            m.abbreviation AS abbreviation,
            src.name AS source,
            dst.name AS destination,
            m.size_cfp AS size_cfp
        ORDER BY m.name
        """

        constraints_query = """
        MATCH (cst:Constraint)
        WHERE cst.id IN [
            "CONSTRAINT_NO_SYSTEM_TO_SYSTEM",
            "CONSTRAINT_ONE_DATA_GROUP_PER_MOVEMENT",
            "CONSTRAINT_ONE_OBJECT_PER_DATA_GROUP",
            "CONSTRAINT_CONTROL_COMMAND_IGNORED"
        ]
        RETURN DISTINCT
            cst.name AS constraint_name,
            cst.content AS constraint_content
        ORDER BY cst.name
        """

        guidelines_query = """
        MATCH (g:Guideline)
        WHERE g.id IN [
            "GUIDE_CLOCK_ENTRY",
            "GUIDE_CONTROL_COMMANDS",
            "GUIDE_CONFIRMATION_ERROR_EXIT",
            "GUIDE_NO_READ_FROM_FUNCTIONAL_USER",
            "GUIDE_NO_WRITE_TO_FUNCTIONAL_USER"
        ]
        OPTIONAL MATCH (g)-[:CLARIFIES]->(r:Rule)
        RETURN DISTINCT
            g.id AS guideline_id,
            g.title AS guideline_title,
            g.content AS guideline_content,
            r.number + " - " + r.title AS clarified_rule
        ORDER BY g.title
        """

        operation_query = """
        MATCH (op:OperationPattern)-[rel:REQUIRES_MOVEMENT]->(m:MovementType)
        RETURN
            op.name AS operation,
            op.description AS description,
            m.name AS movement,
            rel.sequence_order AS sequence_order
        ORDER BY op.name, rel.sequence_order
        """

        rules = self._run_query(rules_query)
        movements = self._run_query(movements_query)
        constraints = self._run_query(constraints_query)
        guidelines = self._run_query(guidelines_query)
        operations = self._run_query(operation_query)

        context = self._format_rule_context(
            rules,
            "GraphRAG Sub-Processes and Data Movements Context"
        )

        context += self._format_movement_context(movements)

        context += "\nCOSMIC constraints:\n"
        for c in constraints:
            context += f"- {c.get('constraint_content')}\n"

        context += self._format_guideline_context(
            guidelines,
            "Data Movement Guidelines"
        )

        context += self._format_operation_patterns(operations)

        return context

    def get_validation_context(self) -> str:
        """Retrieve rules and constraints for validation and consistency checks."""

        rules_query = """
        MATCH (r:Rule)-[:APPLIES_TO]->(c:ComponentType)
        WHERE c.name IN [
            "FunctionalProcess",
            "DataMovement",
            "Entry",
            "Exit",
            "Read",
            "Write",
            "DataGroup",
            "FunctionalSize"
        ]
        RETURN DISTINCT
            r.number AS rule_number,
            r.title AS rule_title,
            r.content AS content,
            r.source AS source
        ORDER BY r.number
        LIMIT 30
        """

        constraints_query = """
        MATCH (cst:Constraint)
        RETURN
            cst.name AS constraint_name,
            cst.content AS constraint_content
        ORDER BY cst.name
        """

        movements_query = """
        MATCH (m:MovementType)-[:HAS_SOURCE]->(src:EntityType),
              (m)-[:HAS_DESTINATION]->(dst:EntityType)
        RETURN
            m.name AS movement,
            m.abbreviation AS abbreviation,
            src.name AS source,
            dst.name AS destination,
            m.size_cfp AS size_cfp
        ORDER BY m.name
        """

        rules = self._run_query(rules_query)
        constraints = self._run_query(constraints_query)
        movements = self._run_query(movements_query)

        context = self._format_rule_context(
            rules,
            "GraphRAG Validation Context"
        )

        context += self._format_movement_context(movements)

        if constraints:
            context += "\nValidation Constraints:\n\n"
            for c in constraints:
                context += f"- {c.get('constraint_name')}: {c.get('constraint_content')}\n"

        return context

    # Compatibility method for older rule-engine calls

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5, domain_filter: Optional[str] = None ) -> List[Dict]:
        """
        Compatibility method used by older code.
        Since this GraphRAG version is ontology-based, it returns relevant
        rules, guidelines, constraints, and operation patterns instead of vector chunks.
        """

        q = (query or "").lower()

        component_names = []

        if any(x in q for x in ["functional user", "actor", "user"]):
            component_names.extend(["FunctionalUser", "FunctionalProcess"])

        if any(x in q for x in ["functional process", "trigger", "triggering"]):
            component_names.extend(["FunctionalProcess", "TriggeringEvent", "TriggeringEntry"])

        if any(x in q for x in ["data group", "object of interest", "attribute"]):
            component_names.extend(["DataGroup", "ObjectOfInterest", "DataAttribute"])

        if any(x in q for x in ["movement", "entry", "exit", "read", "write", "validation", "quality"]):
            component_names.extend(["DataMovement", "Entry", "Exit", "Read", "Write"])

        if not component_names:
            component_names = [
                "FunctionalProcess",
                "DataMovement",
                "DataGroup",
                "Entry",
                "Exit",
                "Read",
                "Write"
            ]

        component_names = list(dict.fromkeys(component_names))

        rule_query = """
        MATCH (r:Rule)-[:APPLIES_TO]->(c:ComponentType)
        WHERE c.name IN $component_names
        RETURN DISTINCT
            r.id AS id,
            r.number AS section,
            r.title AS title,
            r.content AS content,
            "Rule" AS type
        ORDER BY r.number
        LIMIT $top_k
        """

        rows = self._run_query(
            rule_query,
            {
                "component_names": component_names,
                "top_k": top_k
            }
        )

        guideline_query = """
        MATCH (g:Guideline)-[:APPLIES_TO]->(c:ComponentType)
        WHERE c.name IN $component_names
        RETURN DISTINCT
            g.id AS id,
            g.title AS section,
            g.title AS title,
            g.content AS content,
            "Guideline" AS type
        ORDER BY g.title
        LIMIT $top_k
        """

        guide_rows = self._run_query(
            guideline_query,
            {
                "component_names": component_names,
                "top_k": top_k
            }
        )

        return rows + guide_rows