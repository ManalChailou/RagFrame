import os
import logging
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)
load_dotenv()


class CosmicGraphRAGSystem:
    """
    GraphRAG retrieval layer aligned with the current strict COSMIC graph schema.

    Core ontology labels:
    - Rule
    - Guideline
    - ValidationRule
    - ComponentType
    - Concept
    - EntityType
    - MovementType

    Example layer labels:
    - ExampleKnowledge
    - FURExample
    - ExampleFunctionalUser
    - ExampleFunctionalProcess
    - ExampleDataGroup
    - ExampleSubProcess
    - Domain

    This version retrieves BOTH:
    1) COSMIC rules/guidelines/validation rules
    2) Similar examples loaded by load_cosmic_examples.py
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")

        if not self.password:
            raise RuntimeError("Missing NEO4J_PASSWORD in .env")

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )

    # ------------------------------------------------------------------
    # Basic Neo4j helpers
    # ------------------------------------------------------------------

    def close(self):
        if self.driver:
            self.driver.close()

    def _run_query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
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
        node_query = """
        MATCH (n)
        RETURN labels(n) AS labels, count(n) AS count
        ORDER BY labels
        """

        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS relationship, count(r) AS count
        ORDER BY relationship
        """

        nodes = self._run_query(node_query)
        rels = self._run_query(rel_query)

        return {
            "node_counts": {
                ",".join(row["labels"]): row["count"]
                for row in nodes
            },
            "relationship_counts": {
                row["relationship"]: row["count"]
                for row in rels
            },
            "example_counts": self.get_example_stats()
        }

    def get_example_stats(self) -> Dict[str, int]:
        query = """
        MATCH (n)
        WHERE n:ExampleKnowledge
           OR n:ExampleKnowledgeChunk
           OR n:FURExample
           OR n:ExampleFunctionalUser
           OR n:ExampleFunctionalProcess
           OR n:ExampleDataGroup
           OR n:ExampleSubProcess
        RETURN labels(n) AS labels, count(n) AS count
        ORDER BY labels
        """

        rows = self._run_query(query)
        return {
            ",".join(row["labels"]): row["count"]
            for row in rows
        }

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str, limit: int = 1200) -> str:
        text = " ".join(str(text or "").split())
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    def _format_rule_context(self, records: List[Dict]) -> str:
        if not records:
            return ""

        context = "COSMIC knowledge:\n"
        seen = set()

        for row in records:
            title = row.get("title") or row.get("rule_title") or ""
            content = row.get("content") or ""

            key = (title, content)
            if not content or key in seen:
                continue

            seen.add(key)

            if title:
                context += f"- {title}: {content}\n"
            else:
                context += f"- {content}\n"

        return context.strip() + "\n"

    def _format_guideline_context(self, records: List[Dict]) -> str:
        if not records:
            return ""

        context = "\nCOSMIC guidance:\n"
        seen = set()

        for row in records:
            gid = row.get("id") or row.get("guideline_id") or row.get("title")
            content = row.get("content") or row.get("guideline_content") or ""

            if not content or gid in seen:
                continue

            seen.add(gid)
            context += f"- {content}\n"

        return context.strip() + "\n"

    def _format_validation_context(self, records: List[Dict]) -> str:
        if not records:
            return ""

        context = "\nCOSMIC validation rules:\n"
        seen = set()

        for row in records:
            vid = row.get("id") or row.get("title")
            content = row.get("content") or ""

            if not content or vid in seen:
                continue

            seen.add(vid)
            context += f"- {content}\n"

        return context.strip() + "\n"

    def _format_movement_context(self, records: List[Dict]) -> str:
        if not records:
            return ""

        context = "\nCOSMIC movement directions:\n"
        seen = set()

        for row in records:
            movement = row.get("movement")
            abbr = row.get("abbreviation")
            source = row.get("source")
            destination = row.get("destination")
            size = row.get("size_cfp")

            key = (movement, source, destination)
            if key in seen:
                continue
            seen.add(key)

            context += (
                f"- {movement} ({abbr}): "
                f"{source} -> {destination}; "
                f"{size} CFP\n"
            )

        return context.strip() + "\n"

    def _format_examples_context(self, records: List[Dict]) -> str:
        if not records:
            return ""

        context = "\nCOSMIC examples:\n"
        seen = set()

        for row in records:
            eid = row.get("id")
            content = row.get("content") or ""

            if not eid or not content or eid in seen:
                continue

            seen.add(eid)

            component_domain = (
                row.get("component_domain")
                or row.get("domain")
                or "example"
            )

            context += f"- [{component_domain}] {self._truncate(content)}\n"

        return context.strip() + "\n"

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _get_rules_by_ids(self, rule_ids: List[str]) -> List[Dict]:
        query = """
        MATCH (r:Rule)
        WHERE r.id IN $rule_ids
        RETURN DISTINCT
            r.id AS id,
            r.title AS title,
            r.content AS content
        ORDER BY r.id
        """
        return self._run_query(query, {"rule_ids": rule_ids})

    def _get_guidelines_by_ids(self, guideline_ids: List[str]) -> List[Dict]:
        query = """
        MATCH (g:Guideline)
        WHERE g.id IN $guideline_ids
        RETURN DISTINCT
            g.id AS id,
            g.title AS title,
            g.content AS content
        ORDER BY g.id
        """
        return self._run_query(query, {"guideline_ids": guideline_ids})

    def _get_validation_rules_by_ids(self, validation_ids: List[str]) -> List[Dict]:
        query = """
        MATCH (v:ValidationRule)
        WHERE v.id IN $validation_ids
        RETURN DISTINCT
            v.id AS id,
            v.title AS title,
            v.content AS content
        ORDER BY v.id
        """
        return self._run_query(query, {"validation_ids": validation_ids})

    def _get_movement_directions(self) -> List[Dict]:
        query = """
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
        return self._run_query(query)

    def _get_structural_context_for_fp(self) -> List[Dict]:
        query = """
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
        return self._run_query(query)

    def _get_structural_context_for_dg(self) -> List[Dict]:
        query = """
        MATCH (dg:ComponentType {name: "DataGroup"})
        OPTIONAL MATCH (dg)-[:DESCRIBES_EXACTLY_ONE]->(ooi)
        OPTIONAL MATCH (dg)-[:COMPOSED_OF]->(attr)
        OPTIONAL MATCH (dg)-[:DISTINGUISHED_BY]->(criterion)
        RETURN
            dg.name AS component,
            collect(DISTINCT ooi.name) AS describes,
            collect(DISTINCT attr.name) AS composed_of,
            collect(DISTINCT criterion.name) AS distinguished_by
        """
        return self._run_query(query)

    # ------------------------------------------------------------------
    # Example retrieval
    # ------------------------------------------------------------------

    def _extract_query_terms(self, requirements: List[str]) -> List[str]:
        raw = " ".join(requirements or []).lower()
        raw = raw.replace("-", " ").replace("_", " ")

        tokens = []
        current = []
        for ch in raw:
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))

        stop_words = {
            "when", "then", "with", "from", "that", "this", "data",
            "system", "user", "users", "name", "shall", "will", "must",
            "display", "displays", "show", "shows", "error", "message",
            "messages", "confirmation", "select", "selects", "enter",
            "enters", "provided", "provides", "record", "records",
            "validates", "validate", "updates", "update", "edits", "edit",
            "using", "into", "onto", "about", "where", "which"
        }

        terms = []
        for token in tokens:
            if len(token) >= 4 and token not in stop_words and token not in terms:
                terms.append(token)

        return terms[:25]

    def _get_examples(
        self,
        requirements: List[str],
        component_domains: List[str],
        app_domain: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict]:
        """
        Retrieve examples from the examples layer.

        Important:
        - Supports both labels used during your iterations:
          ExampleKnowledge and ExampleKnowledgeChunk.
        - Supports both property names:
          component_domain and domain.
        - Does not strictly filter by app_domain. It ranks same-domain examples
          first but still returns examples if the stored domain differs.
        """
        normalized_domains = [
            str(d).strip().lower()
            for d in (component_domains or [])
            if str(d).strip()
        ]

        terms = self._extract_query_terms(requirements)
        normalized_app_domain = (app_domain or "").strip().lower()

        query = """
        MATCH (ex)
        WHERE (ex:ExampleKnowledge OR ex:ExampleKnowledgeChunk)
          AND toLower(coalesce(ex.component_domain, ex.domain, "")) IN $component_domains

        WITH ex,
             toLower(coalesce(ex.app_domain, "")) AS ex_domain,
             toLower(coalesce(ex.component_domain, ex.domain, "")) AS component_domain,
             toLower(coalesce(ex.content, "")) AS content,
             coalesce(ex.keywords, []) AS keywords

        WITH ex, ex_domain, component_domain, content, keywords,
             CASE
                WHEN $app_domain = "" THEN 0
                WHEN ex_domain = $app_domain THEN 3
                WHEN ex_domain = "general" THEN 2
                ELSE 1
             END AS domain_score,
             size([t IN $terms WHERE content CONTAINS t]) AS content_score,
             size([kw IN keywords
                   WHERE toLower(toString(kw)) IN $terms
                      OR content CONTAINS toLower(toString(kw))]) AS keyword_score

        RETURN
            ex.id AS id,
            component_domain AS component_domain,
            ex_domain AS app_domain,
            ex.section AS section,
            ex.content AS content,
            domain_score + content_score + keyword_score AS score
        ORDER BY score DESC, ex.id
        LIMIT $limit
        """

        rows = self._run_query(
            query,
            {
                "component_domains": normalized_domains,
                "app_domain": normalized_app_domain,
                "terms": terms,
                "limit": limit
            }
        )

        # Fallback to structured FURExample nodes if raw ExampleKnowledge is absent.
        if not rows:
            fallback_query = """
            MATCH (fur:FURExample)
            WHERE toLower(coalesce(fur.component_domain, "")) IN $component_domains
            RETURN
                fur.id AS id,
                toLower(coalesce(fur.component_domain, "example")) AS component_domain,
                toLower(coalesce(fur.app_domain, "")) AS app_domain,
                "FURExample" AS section,
                fur.text AS content,
                0 AS score
            ORDER BY fur.id
            LIMIT $limit
            """

            rows = self._run_query(
                fallback_query,
                {
                    "component_domains": normalized_domains,
                    "limit": limit
                }
            )

        return rows

    # ------------------------------------------------------------------
    # Context methods used by EnhancedPromptDispatcher
    # ------------------------------------------------------------------

    def get_context_for_functional_users(
        self,
        requirements: List[str],
        app_domain: Optional[str] = None
    ) -> str:
        rules = self._get_rules_by_ids([
            "RULE_FU_IDENTIFICATION",
            "RULE_STORAGE_NOT_FU",
            "RULE_FU_PER_FP"
        ])

        examples = self._get_examples(
            requirements=requirements,
            component_domains=["functional_users"],
            app_domain=app_domain,
            limit=2
        )

        context = self._format_rule_context(rules)
        context += self._format_examples_context(examples)
        return context

    def get_context_for_functional_processes(
        self,
        requirements: List[str],
        app_domain: Optional[str] = None
    ) -> str:
        rules = self._get_rules_by_ids([
            "RULE_FP_IDENTIFICATION",
            "RULE_TRIGGERING_CHAIN"
        ])

        guidelines = self._get_guidelines_by_ids([
            "GUIDE_FP_IDENTIFICATION_STEPS",
            "GUIDE_TRIGGER_EVENTS_FROM_MODELS"
        ])

        structure = self._get_structural_context_for_fp()

        examples = self._get_examples(
            requirements=requirements,
            component_domains=["functional_processes"],
            app_domain=app_domain,
            limit=2
        )

        context = self._format_rule_context(rules)

        if structure:
            s = structure[0]
            context += "\nCOSMIC structural constraints:\n"
            context += f"- FunctionalProcess must be initiated by: {s.get('initiated_by', [])}\n"
            context += f"- FunctionalProcess must be partitioned into: {s.get('partitioned_into', [])}\n"
            context += f"- FunctionalProcess must have: {s.get('must_have', [])}\n"
            context += f"- FunctionalProcess must have at least one of: {s.get('must_have_at_least_one_of', [])}\n"

        context += self._format_guideline_context(guidelines)
        context += self._format_examples_context(examples)
        return context

    def get_context_for_data_groups(
        self,
        requirements: List[str],
        functional_processes: List[Dict],
        app_domain: Optional[str] = None
    ) -> str:
        rules = self._get_rules_by_ids([
            "RULE_DG_IDENTIFICATION",
            "RULE_DG_NOT_INTERNAL",
            "RULE_DG_SCREEN_REPORT_FILTER"
        ])

        guidelines = self._get_guidelines_by_ids([
            "GUIDE_DG_FREQUENCY",
            "GUIDE_DG_IDENTIFYING_KEYS",
            "GUIDE_DG_SAME_GROUP_AFTER_CRITERIA",
            "GUIDE_FU_MAY_BE_OOI",
            "GUIDE_SINGLE_ATTRIBUTE_REAL_TIME_DG",
            "GUIDE_OOI_CONTEXT_DEPENDENT",
            "GUIDE_DG_ORIGINS",
            "GUIDE_MULTIPLE_OOI_IN_INPUT_OUTPUT",
            "GUIDE_COMPLEX_OUTPUTS",
            "GUIDE_NO_DG_FROM_DATA_MANIPULATION"
        ])

        structure = self._get_structural_context_for_dg()

        examples = self._get_examples(
            requirements=requirements,
            component_domains=["data_groups"],
            app_domain=app_domain,
            limit=3
        )

        context = self._format_rule_context(rules)

        if structure:
            s = structure[0]
            context += "\nCOSMIC structural constraints:\n"
            context += f"- DataGroup describes exactly one: {s.get('describes', [])}\n"
            context += f"- DataGroup is composed of: {s.get('composed_of', [])}\n"
            context += f"- DataGroup can be distinguished by: {s.get('distinguished_by', [])}\n"

        context += self._format_guideline_context(guidelines)
        context += self._format_examples_context(examples)
        return context

    def get_context_for_sub_processes(
        self,
        requirements: List[str],
        functional_processes: List[Dict],
        data_groups: List[Dict],
        app_domain: Optional[str] = None
    ) -> str:
        rules = self._get_rules_by_ids([
            "RULE_SUBPROCESS",
            "RULE_DATA_MOVEMENT_IDENTIFICATION",
            "RULE_ENTRY",
            "RULE_SINGLE_ENTRY",
            "RULE_EXIT",
            "RULE_READ",
            "RULE_WRITE",
            "RULE_SINGLE_EXIT_READ_WRITE",
            "RULE_OCCURRENCES",
            "RULE_ERROR_CONFIRMATION_EXIT",
            "RULE_ONE_EXIT_FOR_ERROR_CONFIRMATION",
            "RULE_ERROR_ADDITIONAL_DATA",
            "RULE_READ_WRITE_ERROR_CONDITIONS",
            "RULE_CFP_UNIT"
        ])

        guidelines = self._get_guidelines_by_ids([
            "GUIDE_ERROR_CONFIRMATION_MESSAGES",
            "GUIDE_OTHER_ERROR_DATA_NORMAL_ENTRY_EXIT",
            "GUIDE_OS_ERROR_NOT_COUNTED",
            "GUIDE_NO_DG_FROM_DATA_MANIPULATION",
            "GUIDE_MULTIPLE_OOI_IN_INPUT_OUTPUT"
        ])

        validations = self._get_validation_rules_by_ids([
            "VALIDATE_FP_MINIMUM_MOVEMENTS",
            "VALIDATE_ONE_DG_PER_MOVEMENT",
            "VALIDATE_IGNORE_INTERNAL_MANIPULATION",
            "VALIDATE_ERROR_CONFIRMATION_DEDUP"
        ])

        movements = self._get_movement_directions()

        examples = self._get_examples(
            requirements=requirements,
            component_domains=["sub_processes", "data_movements"],
            app_domain=app_domain,
            limit=4
        )

        context = self._format_rule_context(rules)
        context += self._format_movement_context(movements)
        context += self._format_validation_context(validations)
        context += self._format_guideline_context(guidelines)
        context += self._format_examples_context(examples)
        return context

    def get_validation_context(self) -> str:
        rules = self._get_rules_by_ids([
            "RULE_FP_IDENTIFICATION",
            "RULE_DG_IDENTIFICATION",
            "RULE_DATA_MOVEMENT_IDENTIFICATION",
            "RULE_ENTRY",
            "RULE_EXIT",
            "RULE_READ",
            "RULE_WRITE",
            "RULE_SINGLE_ENTRY",
            "RULE_SINGLE_EXIT_READ_WRITE",
            "RULE_OCCURRENCES",
            "RULE_ERROR_CONFIRMATION_EXIT",
            "RULE_ONE_EXIT_FOR_ERROR_CONFIRMATION",
            "RULE_CFP_UNIT"
        ])

        validations = self._run_query(
            """
            MATCH (v:ValidationRule)
            RETURN DISTINCT
                v.id AS id,
                v.title AS title,
                v.content AS content
            ORDER BY v.id
            """
        )

        movements = self._get_movement_directions()

        context = self._format_rule_context(rules)
        context += self._format_movement_context(movements)
        context += self._format_validation_context(validations)

        return context

    # ------------------------------------------------------------------
    # Compatibility method for older rule-engine calls
    # ------------------------------------------------------------------

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Compatibility method used by the existing rule engine.

        Returns Rule, Guideline, ValidationRule and ExampleKnowledge records
        formatted like the old vector chunks.
        """
        q = (query or "").lower()
        component_names = []

        if any(x in q for x in ["functional user", "actor", "user"]):
            component_names.extend(["FunctionalUser", "FunctionalProcess"])

        if any(x in q for x in ["functional process", "trigger", "triggering"]):
            component_names.extend(["FunctionalProcess", "TriggeringEvent", "TriggeringEntry"])

        if any(x in q for x in ["data group", "object of interest", "attribute"]):
            component_names.extend(["DataGroup", "ObjectOfInterest", "DataAttribute"])

        if any(x in q for x in ["sub-process", "sub process", "movement", "entry", "exit", "read", "write", "validation", "quality"]):
            component_names.extend(["SubProcess", "DataMovement", "Entry", "Exit", "Read", "Write"])

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
            r.title AS section,
            r.title AS title,
            r.content AS content,
            "Rule" AS type
        ORDER BY r.id
        LIMIT $top_k
        """

        guideline_query = """
        MATCH (g:Guideline)-[:APPLIES_TO]->(c:ComponentType)
        WHERE c.name IN $component_names
        RETURN DISTINCT
            g.id AS id,
            g.title AS section,
            g.title AS title,
            g.content AS content,
            "Guideline" AS type
        ORDER BY g.id
        LIMIT $top_k
        """

        validation_query = """
        MATCH (v:ValidationRule)-[:APPLIES_TO]->(c:ComponentType)
        WHERE c.name IN $component_names
        RETURN DISTINCT
            v.id AS id,
            v.title AS section,
            v.title AS title,
            v.content AS content,
            "ValidationRule" AS type
        ORDER BY v.id
        LIMIT $top_k
        """

        example_query = """
        MATCH (ex)
        WHERE ex:ExampleKnowledge OR ex:ExampleKnowledgeChunk
        WITH ex,
             toLower(coalesce(ex.content, "")) AS content,
             toLower(coalesce(ex.component_domain, ex.domain, "")) AS component_domain,
             coalesce(ex.keywords, []) AS keywords
        WITH ex, content, component_domain, keywords,
             size([kw IN keywords
                   WHERE toLower($query_text) CONTAINS toLower(toString(kw))
                      OR content CONTAINS toLower(toString(kw))]) AS keyword_score
        RETURN
            ex.id AS id,
            ex.section AS section,
            component_domain AS title,
            ex.content AS content,
            "ExampleKnowledge" AS type
        ORDER BY keyword_score DESC, ex.id
        LIMIT $top_k
        """

        params = {
            "component_names": component_names,
            "top_k": top_k,
            "query_text": q
        }

        rows = []
        rows.extend(self._run_query(rule_query, params))
        rows.extend(self._run_query(guideline_query, params))
        rows.extend(self._run_query(validation_query, params))
        rows.extend(self._run_query(example_query, params))
        return rows

