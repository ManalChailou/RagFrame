import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise RuntimeError("Missing NEO4J_PASSWORD in .env")


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


def run_query(title, query):
    print(f"\n=== {title} ===")

    with driver.session() as session:
        result = session.run(query)

        records = list(result)

        if not records:
            print("No results found.")
            return

        for record in records:
            print(dict(record))


def test_graph_counts():
    query = """
    MATCH (n)
    RETURN labels(n) AS labels, count(n) AS count
    ORDER BY labels
    """

    run_query("Node counts", query)


def test_relationship_counts():
    query = """
    MATCH ()-[r]->()
    RETURN type(r) AS relationship, count(r) AS count
    ORDER BY relationship
    """

    run_query("Relationship counts", query)


def test_movement_directions():
    query = """
    MATCH (m:MovementType)-[:HAS_SOURCE]->(src:EntityType),
          (m)-[:HAS_DESTINATION]->(dst:EntityType)
    RETURN m.name AS movement,
           m.abbreviation AS abbreviation,
           src.name AS source,
           dst.name AS destination,
           m.size_cfp AS size_cfp
    ORDER BY movement
    """

    run_query("Movement directions", query)


def test_functional_process_rules():
    query = """
    MATCH (fp:ComponentType {name: "FunctionalProcess"})
    OPTIONAL MATCH (fp)-[:MUST_BE_INITIATED_BY]->(te)
    OPTIONAL MATCH (fp)-[:MUST_BE_PARTITIONED_INTO]->(dm)
    OPTIONAL MATCH (fp)-[:MUST_HAVE]->(entry)
    OPTIONAL MATCH (fp)-[:MUST_HAVE_AT_LEAST_ONE_OF]->(required)
    RETURN fp.name AS component,
           collect(DISTINCT te.name) AS initiated_by,
           collect(DISTINCT dm.name) AS partitioned_into,
           collect(DISTINCT entry.name) AS must_have,
           collect(DISTINCT required.name) AS must_have_at_least_one_of
    """

    run_query("Functional process rules", query)


def test_data_group_rules():
    query = """
    MATCH (dg:ComponentType {name: "DataGroup"})
    OPTIONAL MATCH (dg)-[:DESCRIBES_EXACTLY_ONE]->(ooi)
    OPTIONAL MATCH (dg)-[:COMPOSED_OF]->(attr)
    RETURN dg.name AS component,
           collect(DISTINCT ooi.name) AS describes,
           collect(DISTINCT attr.name) AS composed_of
    """

    run_query("Data group rules", query)


def test_rules_by_component():
    query = """
    MATCH (r:Rule)-[:APPLIES_TO]->(c:ComponentType)
    RETURN c.name AS component,
           collect(DISTINCT r.number + " - " + r.title) AS rules
    ORDER BY component
    """

    run_query("Rules by component", query)


def test_guidelines():
    query = """
    MATCH (g:Guideline)-[:CLARIFIES]->(r:Rule)
    RETURN g.title AS guideline,
           r.number AS clarified_rule,
           r.title AS rule_title
    ORDER BY clarified_rule
    """

    run_query("Guidelines linked to rules", query)


if __name__ == "__main__":
    try:
        test_graph_counts()
        test_relationship_counts()
        test_movement_directions()
        test_functional_process_rules()
        test_data_group_rules()
        test_rules_by_component()
        test_guidelines()

    finally:
        driver.close()