import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise RuntimeError("Missing NEO4J_PASSWORD in .env")


class CosmicCoreOntologyLoader:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def run(self, query: str, params: dict | None = None):
        with self.driver.session() as session:
            session.run(query, params or {})

    def create_constraints(self):
        logger.info("Creating constraints...")

        constraints = [
            """
            CREATE CONSTRAINT concept_name IF NOT EXISTS
            FOR (c:Concept)
            REQUIRE c.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT rule_id IF NOT EXISTS
            FOR (r:Rule)
            REQUIRE r.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT principle_id IF NOT EXISTS
            FOR (p:Principle)
            REQUIRE p.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT movement_type_name IF NOT EXISTS
            FOR (m:MovementType)
            REQUIRE m.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT entity_type_name IF NOT EXISTS
            FOR (e:EntityType)
            REQUIRE e.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT component_type_name IF NOT EXISTS
            FOR (c:ComponentType)
            REQUIRE c.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT constraint_id IF NOT EXISTS
            FOR (c:Constraint)
            REQUIRE c.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT source_id IF NOT EXISTS
            FOR (s:SourceDocument)
            REQUIRE s.id IS UNIQUE
            """
        ]

        for q in constraints:
            self.run(q)

    def load_source_documents(self):
        logger.info("Loading source documents...")

        documents = [
            {
                "id": "cosmic_part_1_v5_2021",
                "title": "COSMIC Measurement Manual Part 1: Principles, Definitions & Rules",
                "version": "5.0",
                "date": "August 2021",
                "type": "manual_part_1"
            },
            {
                "id": "cosmic_part_2_v5_2024",
                "title": "COSMIC Measurement Manual Part 2: Guidelines",
                "version": "5.0",
                "date": "September 2024",
                "type": "manual_part_2"
            }
        ]

        for doc in documents:
            self.run(
                """
                MERGE (s:SourceDocument {id: $id})
                SET s.title = $title,
                    s.version = $version,
                    s.date = $date,
                    s.type = $type
                """,
                doc
            )

    def load_component_types(self):
        logger.info("Loading component types...")

        component_types = [
            {
                "name": "FunctionalUser",
                "label": "Functional User",
                "description": "A sender and/or intended recipient of data in the FUR of the software being measured."
            },
            {
                "name": "FunctionalProcess",
                "label": "Functional Process",
                "description": "A unique, cohesive and independently executable set of data movements."
            },
            {
                "name": "TriggeringEvent",
                "label": "Triggering Event",
                "description": "An event that causes a functional user to initiate one or more functional processes."
            },
            {
                "name": "TriggeringEntry",
                "label": "Triggering Entry",
                "description": "The Entry data movement that starts a functional process."
            },
            {
                "name": "DataGroup",
                "label": "Data Group",
                "description": "A set of data attributes describing one object of interest."
            },
            {
                "name": "ObjectOfInterest",
                "label": "Object of Interest",
                "description": "A thing from the FUR perspective about which the software processes or stores data."
            },
            {
                "name": "DataAttribute",
                "label": "Data Attribute",
                "description": "Smallest parcel of information within a data group."
            },
            {
                "name": "DataMovement",
                "label": "Data Movement",
                "description": "A sub-process that moves one data group."
            },
            {
                "name": "Boundary",
                "label": "Boundary",
                "description": "Conceptual interface between the software being measured and its functional users."
            },
            {
                "name": "PersistentStorage",
                "label": "Persistent Storage",
                "description": "Storage used to store or retrieve data beyond the life of a functional process."
            },
            {
                "name": "Layer",
                "label": "Layer",
                "description": "Partition resulting from the functional division of a software system."
            },
            {
                "name": "FUR",
                "label": "Functional User Requirements",
                "description": "Subset of user requirements describing what the software shall do in terms of tasks and services."
            }
        ]

        for item in component_types:
            self.run(
                """
                MERGE (c:ComponentType {name: $name})
                SET c.label = $label,
                    c.description = $description
                """,
                item
            )

    def load_concepts(self):
        logger.info("Loading concepts...")

        concepts = [
            {
                "name": "COSMIC Method",
                "type": "method",
                "description": "Functional Size Measurement method based on the COSMIC Software Context Model and the Generic Software Model."
            },
            {
                "name": "COSMIC Software Context Model",
                "type": "model",
                "description": "Model used to identify the nature and structure of the software to be measured."
            },
            {
                "name": "Generic Software Model",
                "type": "model",
                "description": "Model applied to FUR to extract and measure functional size elements."
            },
            {
                "name": "Functional Size",
                "type": "measurement",
                "description": "Size of the software derived by quantifying Functional User Requirements."
            },
            {
                "name": "COSMIC Function Point",
                "type": "unit",
                "description": "Unit of measurement. One data movement has size 1 CFP."
            },
            {
                "name": "Data Manipulation",
                "type": "concept",
                "description": "Processing of data other than a movement of data into or out of a functional process or to/from persistent storage."
            },
            {
                "name": "Control Command",
                "type": "concept",
                "description": "Command that controls use of software but does not move data about an object of interest."
            },
            {
                "name": "Error/Confirmation Message",
                "type": "concept",
                "description": "Exit issued by a functional process to a human user confirming acceptance or reporting an error."
            }
        ]

        for item in concepts:
            self.run(
                """
                MERGE (c:Concept {name: $name})
                SET c.type = $type,
                    c.description = $description
                """,
                item
            )

    def load_entity_types(self):
        logger.info("Loading entity types...")

        entity_types = [
            {
                "name": "Functional User",
                "description": "External sender or recipient of data across the software boundary."
            },
            {
                "name": "Functional Process",
                "description": "Process inside the measured software boundary."
            },
            {
                "name": "Persistent Storage",
                "description": "Storage inside the software side of the boundary."
            },
            {
                "name": "Boundary",
                "description": "Conceptual interface separating functional users from the measured software."
            },
            {
                "name": "Measured Software",
                "description": "Piece of software within the scope of the functional size measurement."
            },
            {
                "name": "Clock",
                "description": "Functional user for clock-tick triggering events."
            },
            {
                "name": "External Application",
                "description": "External software interacting with the measured software."
            },
            {
                "name": "Hardware Device",
                "description": "Hardware device interacting with the measured software."
            },
            {
                "name": "Human User",
                "description": "Human functional user interacting with the measured software."
            }
        ]

        for item in entity_types:
            self.run(
                """
                MERGE (e:EntityType {name: $name})
                SET e.description = $description
                """,
                item
            )

    def load_movement_types(self):
        logger.info("Loading movement types...")

        movements = [
            {
                "name": "Entry",
                "abbreviation": "E",
                "description": "Moves one data group from a functional user across the boundary into the functional process.",
                "source": "Functional User",
                "destination": "Functional Process"
            },
            {
                "name": "Exit",
                "abbreviation": "X",
                "description": "Moves one data group from the functional process across the boundary to the functional user.",
                "source": "Functional Process",
                "destination": "Functional User"
            },
            {
                "name": "Read",
                "abbreviation": "R",
                "description": "Moves one data group from persistent storage to the functional process.",
                "source": "Persistent Storage",
                "destination": "Functional Process"
            },
            {
                "name": "Write",
                "abbreviation": "W",
                "description": "Moves one data group from the functional process to persistent storage.",
                "source": "Functional Process",
                "destination": "Persistent Storage"
            }
        ]

        for m in movements:
            self.run(
                """
                MERGE (mt:MovementType {name: $name})
                SET mt.abbreviation = $abbreviation,
                    mt.description = $description,
                    mt.size_cfp = 1

                MERGE (src:EntityType {name: $source})
                MERGE (dst:EntityType {name: $destination})

                MERGE (mt)-[:HAS_SOURCE]->(src)
                MERGE (mt)-[:HAS_DESTINATION]->(dst)

                MERGE (dm:ComponentType {name: "DataMovement"})
                MERGE (mt)-[:IS_TYPE_OF]->(dm)

                MERGE (unit:Concept {name: "COSMIC Function Point"})
                MERGE (mt)-[:HAS_SIZE_UNIT]->(unit)
                """,
                m
            )

    def load_principles(self):
        logger.info("Loading COSMIC principles...")

        principles = [
            {
                "id": "GSM_P1",
                "model": "Generic Software Model",
                "content": "A piece of software interacts with its functional users across a boundary, and with persistent storage within this boundary.",
                "applies_to": ["Boundary", "FunctionalUser", "PersistentStorage"]
            },
            {
                "id": "GSM_P2",
                "model": "Generic Software Model",
                "content": "A functional process consists of sub-processes called data movements.",
                "applies_to": ["FunctionalProcess", "DataMovement"]
            },
            {
                "id": "GSM_P3",
                "model": "Generic Software Model",
                "content": "There are four data movement sub-types: Entry, Exit, Write and Read. A data movement sub-type includes any associated data manipulation.",
                "applies_to": ["DataMovement"]
            },
            {
                "id": "GSM_P4",
                "model": "Generic Software Model",
                "content": "A data movement moves a single data group.",
                "applies_to": ["DataMovement", "DataGroup"]
            },
            {
                "id": "GSM_P5",
                "model": "Generic Software Model",
                "content": "A data group consists of a unique set of data attributes that describe a single object of interest.",
                "applies_to": ["DataGroup", "ObjectOfInterest", "DataAttribute"]
            },
            {
                "id": "GSM_P6",
                "model": "Generic Software Model",
                "content": "Each functional process is initiated by a triggering event, detected by a functional user, which initiates a triggering Entry.",
                "applies_to": ["FunctionalProcess", "TriggeringEvent", "FunctionalUser", "TriggeringEntry"]
            },
            {
                "id": "GSM_P7",
                "model": "Generic Software Model",
                "content": "Functional size is based on the types of elements used for measurement, not on the number of occurrences.",
                "applies_to": ["FunctionalSize", "DataMovement"]
            },
            {
                "id": "GSM_P8",
                "model": "Generic Software Model",
                "content": "The size of a functional process is equal to the number of its data movements where one data movement has size 1 CFP.",
                "applies_to": ["FunctionalProcess", "DataMovement"]
            },
            {
                "id": "GSM_P9",
                "model": "Generic Software Model",
                "content": "The size of a piece of software is the sum of the sizes of the functional processes within the scope of the FSM.",
                "applies_to": ["FunctionalProcess", "FunctionalSize"]
            }
        ]

        for p in principles:
            self.run(
                """
                MERGE (principle:Principle {id: $id})
                SET principle.model = $model,
                    principle.content = $content

                MERGE (source:SourceDocument {id: "cosmic_part_1_v5_2021"})
                MERGE (principle)-[:FROM_SOURCE]->(source)

                MERGE (model:Concept {name: $model})
                MERGE (principle)-[:BELONGS_TO_MODEL]->(model)
                """,
                {
                    "id": p["id"],
                    "model": p["model"],
                    "content": p["content"]
                }
            )

            for component in p["applies_to"]:
                self.run(
                    """
                    MATCH (principle:Principle {id: $principle_id})
                    MERGE (c:ComponentType {name: $component})
                    MERGE (principle)-[:APPLIES_TO]->(c)
                    """,
                    {
                        "principle_id": p["id"],
                        "component": component
                    }
                )

    def load_rules(self):
        logger.info("Loading COSMIC rules...")

        rules = [
            {
                "id": "RULE_03",
                "number": "Rule 3",
                "title": "Identification of the FUR",
                "content": "The FUR identified to be within the scope of the FSM shall be used as the exclusive source from which the functional size of the software is to be measured.",
                "applies_to": ["FUR", "FunctionalSize"]
            },
            {
                "id": "RULE_07",
                "number": "Rule 7",
                "title": "Functional Users",
                "content": "All functional users that trigger, provide information to, or receive information from functional processes in the FUR of the software within the scope of the FSM shall be identified.",
                "applies_to": ["FunctionalUser", "FunctionalProcess", "FUR"]
            },
            {
                "id": "RULE_10",
                "number": "Rule 10",
                "title": "Identification of Functional Processes",
                "content": "Each functional process shall be derived from at least one identifiable FUR, initiated by an Entry from a functional user, comprise at least two data movements, belong to one layer, and be complete for all responses to its triggering Entry.",
                "applies_to": ["FunctionalProcess", "TriggeringEntry", "Entry", "Layer"]
            },
            {
                "id": "RULE_11",
                "number": "Rule 11",
                "title": "Identification of Objects of Interest and Data Groups",
                "content": "Each data group shall be unique and distinguishable through its unique collection of data attributes and directly related to one object of interest described in the FUR.",
                "applies_to": ["DataGroup", "ObjectOfInterest", "DataAttribute"]
            },
            {
                "id": "RULE_12",
                "number": "Rule 12",
                "title": "Identification of Data Movements",
                "content": "Each functional process shall be partitioned into its component data movements.",
                "applies_to": ["FunctionalProcess", "DataMovement"]
            },
            {
                "id": "RULE_13",
                "number": "Rule 13",
                "title": "Functional Process - Single Entry",
                "content": "For any one functional process, a single Entry shall be identified and counted for the entry of all data describing a single object of interest, unless the FUR explicitly require otherwise.",
                "applies_to": ["Entry", "FunctionalProcess", "DataGroup", "ObjectOfInterest"]
            },
            {
                "id": "RULE_14",
                "number": "Rule 14",
                "title": "Functional Process - Single Exit, Read or Write",
                "content": "A single Exit, Read or Write shall be identified and counted for the movement of all data describing a single object of interest, unless the FUR explicitly require otherwise.",
                "applies_to": ["Exit", "Read", "Write", "FunctionalProcess", "DataGroup", "ObjectOfInterest"]
            },
            {
                "id": "RULE_15",
                "number": "Rule 15",
                "title": "Functional Process - Occurrences",
                "content": "If a data movement of a particular type occurs multiple times with different values when a functional process is executed, only one data movement of that type shall be counted.",
                "applies_to": ["DataMovement", "FunctionalProcess"]
            },
            {
                "id": "RULE_16",
                "number": "Rule 16",
                "title": "Entry",
                "content": "An Entry receives a single data group which originates from the functional user side of the boundary.",
                "applies_to": ["Entry", "FunctionalUser", "Boundary", "DataGroup"]
            },
            {
                "id": "RULE_17",
                "number": "Rule 17",
                "title": "Exit",
                "content": "An Exit sends data attributes from a single data group to the functional user side of the boundary.",
                "applies_to": ["Exit", "FunctionalUser", "Boundary", "DataGroup"]
            },
            {
                "id": "RULE_18",
                "number": "Rule 18",
                "title": "Read",
                "content": "A Read retrieves a single data group from persistent storage.",
                "applies_to": ["Read", "PersistentStorage", "DataGroup"]
            },
            {
                "id": "RULE_19",
                "number": "Rule 19",
                "title": "Write",
                "content": "A Write moves data attributes from a single data group to persistent storage.",
                "applies_to": ["Write", "PersistentStorage", "DataGroup"]
            },
            {
                "id": "RULE_20",
                "number": "Rule 20",
                "title": "Write - Delete",
                "content": "A requirement to delete a data group from persistent storage shall be a single Write data movement.",
                "applies_to": ["Write", "PersistentStorage", "DataGroup"]
            },
            {
                "id": "RULE_21",
                "number": "Rule 21",
                "title": "Size of a Data Movement",
                "content": "A unit of measurement, 1 CFP, shall be assigned to each Entry, Exit, Read or Write identified in each functional process.",
                "applies_to": ["DataMovement", "Entry", "Exit", "Read", "Write", "COSMIC Function Point"]
            },
            {
                "id": "RULE_22",
                "number": "Rule 22",
                "title": "Size of a Functional Process",
                "content": "The size of a functional process is the sum of the sizes of its Entries, Exits, Reads and Writes.",
                "applies_to": ["FunctionalProcess", "FunctionalSize", "Entry", "Exit", "Read", "Write"]
            },
            {
                "id": "RULE_23",
                "number": "Rule 23",
                "title": "Functional Size of the Identified FUR",
                "content": "The size of each piece of software within a layer shall be obtained by aggregating the size of the functional processes within the identified FUR.",
                "applies_to": ["FunctionalSize", "FunctionalProcess", "FUR", "Layer"]
            }
        ]

        for r in rules:
            self.run(
                """
                MERGE (rule:Rule {id: $id})
                SET rule.number = $number,
                    rule.title = $title,
                    rule.content = $content,
                    rule.source = "COSMIC Measurement Manual Part 1 v5.0"

                MERGE (source:SourceDocument {id: "cosmic_part_1_v5_2021"})
                MERGE (rule)-[:FROM_SOURCE]->(source)
                """,
                {
                    "id": r["id"],
                    "number": r["number"],
                    "title": r["title"],
                    "content": r["content"]
                }
            )

            for component in r["applies_to"]:
                self.run(
                    """
                    MATCH (rule:Rule {id: $rule_id})
                    MERGE (c:ComponentType {name: $component})
                    MERGE (rule)-[:APPLIES_TO]->(c)
                    """,
                    {
                        "rule_id": r["id"],
                        "component": component
                    }
                )

    def load_rule_relations(self):
        logger.info("Loading semantic rule relations...")

        queries = [
            """
            MATCH (fp:ComponentType {name: "FunctionalProcess"})
            MATCH (te:ComponentType {name: "TriggeringEntry"})
            MATCH (dm:ComponentType {name: "DataMovement"})
            MERGE (fp)-[:MUST_BE_INITIATED_BY]->(te)
            MERGE (fp)-[:MUST_BE_PARTITIONED_INTO]->(dm)
            """,
            """
            MATCH (fp:ComponentType {name: "FunctionalProcess"})
            MATCH (entry:MovementType {name: "Entry"})
            MERGE (fp)-[:MUST_HAVE]->(entry)
            """,
            """
            MATCH (fp:ComponentType {name: "FunctionalProcess"})
            MATCH (exit:MovementType {name: "Exit"})
            MATCH (write:MovementType {name: "Write"})
            MERGE (fp)-[:MUST_HAVE_AT_LEAST_ONE_OF]->(exit)
            MERGE (fp)-[:MUST_HAVE_AT_LEAST_ONE_OF]->(write)
            """,
            """
            MATCH (dm:ComponentType {name: "DataMovement"})
            MATCH (dg:ComponentType {name: "DataGroup"})
            MERGE (dm)-[:MOVES_EXACTLY_ONE]->(dg)
            """,
            """
            MATCH (dg:ComponentType {name: "DataGroup"})
            MATCH (ooi:ComponentType {name: "ObjectOfInterest"})
            MATCH (attr:ComponentType {name: "DataAttribute"})
            MERGE (dg)-[:DESCRIBES_EXACTLY_ONE]->(ooi)
            MERGE (dg)-[:COMPOSED_OF]->(attr)
            """,
            """
            MATCH (te:ComponentType {name: "TriggeringEntry"})
            MATCH (entry:MovementType {name: "Entry"})
            MERGE (te)-[:IS_A]->(entry)
            """,
            """
            MATCH (delete_rule:Rule {id: "RULE_20"})
            MATCH (write:MovementType {name: "Write"})
            MERGE (delete_rule)-[:CLASSIFIES_DELETE_AS]->(write)
            """,
            """
            MATCH (rule21:Rule {id: "RULE_21"})
            MATCH (entry:MovementType {name: "Entry"})
            MATCH (exit:MovementType {name: "Exit"})
            MATCH (read:MovementType {name: "Read"})
            MATCH (write:MovementType {name: "Write"})
            MERGE (rule21)-[:ASSIGNS_ONE_CFP_TO]->(entry)
            MERGE (rule21)-[:ASSIGNS_ONE_CFP_TO]->(exit)
            MERGE (rule21)-[:ASSIGNS_ONE_CFP_TO]->(read)
            MERGE (rule21)-[:ASSIGNS_ONE_CFP_TO]->(write)
            """
        ]

        for q in queries:
            self.run(q)

    def load_constraints(self):
        logger.info("Loading COSMIC constraints...")

        constraints = [
            {
                "id": "CONSTRAINT_STORAGE_NOT_FUNCTIONAL_USER",
                "name": "Persistent storage is not a functional user",
                "content": "Persistent storage is on the software side of the boundary and is not considered a functional user.",
                "applies_to": ["PersistentStorage", "FunctionalUser"]
            },
            {
                "id": "CONSTRAINT_NO_SYSTEM_TO_SYSTEM",
                "name": "Internal System-to-System movement is not counted",
                "content": "Internal data manipulation inside a functional process is not counted as a separate data movement.",
                "applies_to": ["DataMovement", "DataManipulation"]
            },
            {
                "id": "CONSTRAINT_ONE_DATA_GROUP_PER_MOVEMENT",
                "name": "One data group per data movement",
                "content": "A data movement moves one and only one data group.",
                "applies_to": ["DataMovement", "DataGroup"]
            },
            {
                "id": "CONSTRAINT_ONE_OBJECT_PER_DATA_GROUP",
                "name": "One object of interest per data group",
                "content": "A data group describes one object of interest through a unique set of data attributes.",
                "applies_to": ["DataGroup", "ObjectOfInterest"]
            },
            {
                "id": "CONSTRAINT_FUNCTIONAL_PROCESS_MINIMUM_SIZE",
                "name": "Functional process minimum movement structure",
                "content": "A functional process must have at least one Entry and at least one Exit or Write.",
                "applies_to": ["FunctionalProcess", "Entry", "Exit", "Write"]
            },
            {
                "id": "CONSTRAINT_CONTROL_COMMAND_IGNORED",
                "name": "Control commands are ignored",
                "content": "Control commands that do not move data about an object of interest are not counted as data movements.",
                "applies_to": ["ControlCommand", "DataMovement"]
            }
        ]

        for c in constraints:
            self.run(
                """
                MERGE (constraint:Constraint {id: $id})
                SET constraint.name = $name,
                    constraint.content = $content

                MERGE (source:SourceDocument {id: "cosmic_part_1_v5_2021"})
                MERGE (constraint)-[:FROM_SOURCE]->(source)
                """,
                {
                    "id": c["id"],
                    "name": c["name"],
                    "content": c["content"]
                }
            )

            for component in c["applies_to"]:
                self.run(
                    """
                    MATCH (constraint:Constraint {id: $constraint_id})
                    MERGE (c:ComponentType {name: $component})
                    MERGE (constraint)-[:APPLIES_TO]->(c)
                    """,
                    {
                        "constraint_id": c["id"],
                        "component": component
                    }
                )

    def load_operation_patterns(self):
        logger.info("Loading common operation patterns...")

        patterns = [
            {
                "name": "Create",
                "description": "Create or add a new object of interest.",
                "required_movements": ["Entry", "Write", "Exit"]
            },
            {
                "name": "Modify",
                "description": "Modify or update an existing object of interest.",
                "required_movements": ["Entry", "Read", "Write", "Exit"]
            },
            {
                "name": "Delete",
                "description": "Delete an existing data group from persistent storage.",
                "required_movements": ["Entry", "Read", "Write", "Exit"]
            },
            {
                "name": "View",
                "description": "View or consult stored data.",
                "required_movements": ["Entry", "Read", "Exit"]
            },
            {
                "name": "Search",
                "description": "Search stored data based on criteria.",
                "required_movements": ["Entry", "Read", "Exit"]
            },
            {
                "name": "Report",
                "description": "Generate output data for a functional user.",
                "required_movements": ["Entry", "Read", "Exit"]
            },
            {
                "name": "Transmit",
                "description": "Send data to an external application or functional user.",
                "required_movements": ["Entry", "Read", "Exit"]
            },
            {
                "name": "ReceiveExternalData",
                "description": "Receive data from an external application or device.",
                "required_movements": ["Entry", "Write", "Exit"]
            },
            {
                "name": "PeriodicTrigger",
                "description": "Clock or timer triggers a functional process.",
                "required_movements": ["Entry"]
            }
        ]

        for p in patterns:
            self.run(
                """
                MERGE (op:OperationPattern {name: $name})
                SET op.description = $description
                """,
                {
                    "name": p["name"],
                    "description": p["description"]
                }
            )

            for order, movement in enumerate(p["required_movements"], start=1):
                self.run(
                    """
                    MATCH (op:OperationPattern {name: $operation})
                    MATCH (m:MovementType {name: $movement})
                    MERGE (op)-[r:REQUIRES_MOVEMENT]->(m)
                    SET r.sequence_order = $order
                    """,
                    {
                        "operation": p["name"],
                        "movement": movement,
                        "order": order
                    }
                )

    def load_guidelines_from_part_2(self):
        logger.info("Loading selected guidelines from COSMIC Part 2...")

        guidelines = [
            {
                "id": "GUIDE_FP_IDENTIFICATION_CHAIN",
                "title": "Functional process identification chain",
                "content": "Identify triggering events, functional users responding to each event, triggering Entries, and the functional process started by each triggering Entry.",
                "applies_to": ["FunctionalProcess", "TriggeringEvent", "FunctionalUser", "TriggeringEntry"],
                "clarifies": "RULE_10"
            },
            {
                "id": "GUIDE_DG_DIFFERENT_FREQUENCY",
                "title": "Different frequencies indicate different objects of interest",
                "content": "Sets of data attributes with different frequencies of occurrence describe different objects of interest.",
                "applies_to": ["DataGroup", "ObjectOfInterest", "DataAttribute"],
                "clarifies": "RULE_11"
            },
            {
                "id": "GUIDE_DG_DIFFERENT_KEYS",
                "title": "Different identifying keys indicate different objects of interest",
                "content": "Sets of data attributes with the same frequency but different identifying key attributes describe different objects of interest.",
                "applies_to": ["DataGroup", "ObjectOfInterest", "DataAttribute"],
                "clarifies": "RULE_11"
            },
            {
                "id": "GUIDE_CLOCK_ENTRY",
                "title": "Clock-ticks as Entries",
                "content": "For clock-ticks that are triggering events, identify an Entry from a functional user, in this case the Clock.",
                "applies_to": ["Entry", "TriggeringEvent", "FunctionalUser"],
                "clarifies": "RULE_16"
            },
            {
                "id": "GUIDE_CONTROL_COMMANDS",
                "title": "Control commands ignored",
                "content": "In an application with a human interface, control commands are ignored because they do not involve movement of data about an object of interest.",
                "applies_to": ["ControlCommand", "DataMovement", "ObjectOfInterest"],
                "clarifies": "RULE_16"
            },
            {
                "id": "GUIDE_CONFIRMATION_ERROR_EXIT",
                "title": "Error and confirmation messages as Exits",
                "content": "One Exit is identified to account for all types of error or confirmation messages issued by one functional process from all possible causes according to its FUR.",
                "applies_to": ["Exit", "FunctionalProcess"],
                "clarifies": "RULE_17"
            },
            {
                "id": "GUIDE_NO_READ_FROM_FUNCTIONAL_USER",
                "title": "No Read from functional user",
                "content": "Do not identify a Read when the FUR specify a software or hardware functional user as the source of a data group; interaction across the boundary is handled by an Entry.",
                "applies_to": ["Read", "Entry", "FunctionalUser", "Boundary"],
                "clarifies": "RULE_18"
            },
            {
                "id": "GUIDE_NO_WRITE_TO_FUNCTIONAL_USER",
                "title": "No Write to functional user",
                "content": "Do not identify a Write when the FUR specify a software or hardware functional user as the destination of a data group; interaction across the boundary is handled by an Exit.",
                "applies_to": ["Write", "Exit", "FunctionalUser", "Boundary"],
                "clarifies": "RULE_19"
            }
        ]

        for g in guidelines:
            self.run(
                """
                MERGE (guide:Guideline {id: $id})
                SET guide.title = $title,
                    guide.content = $content,
                    guide.source = "COSMIC Measurement Manual Part 2 v5.0"

                MERGE (source:SourceDocument {id: "cosmic_part_2_v5_2024"})
                MERGE (guide)-[:FROM_SOURCE]->(source)
                """,
                {
                    "id": g["id"],
                    "title": g["title"],
                    "content": g["content"]
                }
            )

            self.run(
                """
                MATCH (guide:Guideline {id: $guide_id})
                MATCH (rule:Rule {id: $rule_id})
                MERGE (guide)-[:CLARIFIES]->(rule)
                """,
                {
                    "guide_id": g["id"],
                    "rule_id": g["clarifies"]
                }
            )

            for component in g["applies_to"]:
                self.run(
                    """
                    MATCH (guide:Guideline {id: $guide_id})
                    MERGE (c:ComponentType {name: $component})
                    MERGE (guide)-[:APPLIES_TO]->(c)
                    """,
                    {
                        "guide_id": g["id"],
                        "component": component
                    }
                )

    def verify_graph(self):
        logger.info("Verifying graph content...")

        query = """
        MATCH (n)
        RETURN labels(n) AS labels, count(n) AS count
        ORDER BY labels
        """

        with self.driver.session() as session:
            result = session.run(query)
            print("\n=== Graph content ===")
            for record in result:
                print(f"{record['labels']}: {record['count']}")

        query_rel = """
        MATCH ()-[r]->()
        RETURN type(r) AS relationship, count(r) AS count
        ORDER BY relationship
        """

        with self.driver.session() as session:
            result = session.run(query_rel)
            print("\n=== Relationships ===")
            for record in result:
                print(f"{record['relationship']}: {record['count']}")

    def load_all(self):
        self.create_constraints()
        self.load_source_documents()
        self.load_component_types()
        self.load_concepts()
        self.load_entity_types()
        self.load_movement_types()
        self.load_principles()
        self.load_rules()
        self.load_rule_relations()
        self.load_constraints()
        self.load_operation_patterns()
        self.load_guidelines_from_part_2()
        self.verify_graph()


if __name__ == "__main__":
    loader = CosmicCoreOntologyLoader()

    try:
        loader.load_all()
        print("\nCOSMIC core ontology loaded successfully.")

    finally:
        loader.close()