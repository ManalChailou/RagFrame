import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import Any, Dict, Iterable, List, Optional

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

    def run(self, query: str, params: Optional[dict] = None) -> None:
        with self.driver.session(database="neo4j") as session:
            session.run(query, params or {}).consume()

    def fetch_one(self, query: str, params: Optional[dict] = None) -> Optional[dict]:
        with self.driver.session(database="neo4j") as session:
            record = session.run(query, params or {}).single()
            return dict(record) if record else None

    # RESET DATABASE
    def reset_database(self):
        logger.warning("Deleting all existing graph data...")
        self.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared successfully.")

    # CONSTRAINTS
    def create_constraints(self):
        logger.info("Creating constraints...")

        constraints = [
            """
            CREATE CONSTRAINT component_type_name IF NOT EXISTS
            FOR (c:ComponentType)
            REQUIRE c.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT concept_name IF NOT EXISTS
            FOR (c:Concept)
            REQUIRE c.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT entity_type_name IF NOT EXISTS
            FOR (e:EntityType)
            REQUIRE e.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT movement_type_name IF NOT EXISTS
            FOR (m:MovementType)
            REQUIRE m.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT rule_id IF NOT EXISTS
            FOR (r:Rule)
            REQUIRE r.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT guideline_id IF NOT EXISTS
            FOR (g:Guideline)
            REQUIRE g.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT validation_rule_id IF NOT EXISTS
            FOR (v:ValidationRule)
            REQUIRE v.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT requirement_group_id IF NOT EXISTS
            FOR (g:RequirementGroup)
            REQUIRE g.id IS UNIQUE
            """
        ]

        for query in constraints:
            self.run(query)

    # COMPONENT TYPES
    def load_component_types(self):
        logger.info("Loading component types...")

        component_types = [
            {
                "name": "FunctionalUser",
                "label": "Functional User",
                "description": "Sender and/or intended recipient of data in the Functional User Requirements of the software being measured."
            },
            {
                "name": "FunctionalProcess",
                "label": "Functional Process",
                "description": "Elementary component of the Functional User Requirements comprising a unique, cohesive and independently executable set of data movements."
            },
            {
                "name": "TriggeringEvent",
                "label": "Triggering Event",
                "description": "Event in the world of functional users that causes a functional user to initiate one or more functional processes."
            },
            {
                "name": "TriggeringEntry",
                "label": "Triggering Entry",
                "description": "Entry that moves the data group required to start a functional process. It is a logical COSMIC movement and is not necessarily the first technical user-interface action."
            },
            {
                "name": "DataGroup",
                "label": "Data Group",
                "description": "Distinct, non-empty, non-ordered and non-redundant set of data attributes in which every attribute describes a complementary aspect of the same object of interest."
            },
            {
                "name": "ObjectOfInterest",
                "label": "Object of Interest",
                "description": "A thing identified from the perspective of the Functional User Requirements about which the software must process and/or store data."
            },
            {
                "name": "DataAttribute",
                "label": "Data Attribute",
                "description": "Smallest meaningful parcel of information within a data group. Explicit attribute extraction is optional when the data group and its object of interest are already unambiguous."
            },
            {
                "name": "SubProcess",
                "label": "Sub-process",
                "description": "Part of a functional process that either moves data or manipulates data."
            },
            {
                "name": "DataMovement",
                "label": "Data Movement",
                "description": "Sub-process that moves exactly one data group as an Entry, Exit, Read or Write. It includes the data manipulation directly associated with that movement."
            },
            {"name": "Entry", "label": "Entry", "description": "Moves one data group from a functional user across the boundary into the functional process."},
            {"name": "Exit", "label": "Exit", "description": "Moves one data group from a functional process across the boundary to a functional user."},
            {"name": "Read", "label": "Read", "description": "Moves one data group from persistent storage within reach of the functional process."},
            {"name": "Write", "label": "Write", "description": "Moves one data group from a functional process to persistent storage."},
            {
                "name": "Boundary",
                "label": "Boundary",
                "description": "Conceptual interface between the software being measured and its functional users. The boundary is established from the measurement scope."
            },
            {
                "name": "PersistentStorage",
                "label": "Persistent Storage",
                "description": "Storage on the software side of the boundary that preserves data beyond the current functional process occurrence or makes it available to another occurrence or process. It is not a functional user."
            },
            {
                "name": "ErrorConfirmationMessage",
                "label": "Error / Confirmation Message",
                "description": "Exit to a human functional user that confirms acceptance of entered data or reports an error in entered data."
            },
            {
                "name": "DataManipulation",
                "label": "Data Manipulation",
                "description": "Internal processing such as validation, calculation, comparison, transformation or formatting. It does not count separately when associated with a data movement."
            },
            {
                "name": "ControlCommand",
                "label": "Control Command",
                "description": "Command that controls use of the software but does not move data about an object of interest; it is not counted as a COSMIC data movement."
            },
            {
                "name": "FUR",
                "label": "Functional User Requirements",
                "description": "Subset of user requirements that describes what the software shall do in terms of tasks and services for its functional users."
            },
            {"name": "MeasurementPurpose", "label": "Measurement Purpose", "description": "Reason for performing the COSMIC measurement and intended use of its results."},
            {"name": "MeasurementScope", "label": "Measurement Scope", "description": "Set of Functional User Requirements included in the measurement."},
            {"name": "PieceOfSoftware", "label": "Piece of Software", "description": "Software whose functional size is being measured."},
            {"name": "Layer", "label": "Software Layer", "description": "Partition of software according to the adopted architecture and measurement strategy."},
            {"name": "LevelOfDecomposition", "label": "Level of Decomposition", "description": "Level at which software is decomposed into pieces for measurement."},
            {"name": "LevelOfGranularity", "label": "Level of Granularity", "description": "Level of detail at which Functional User Requirements are available and measured."},
            {"name": "PeerSoftware", "label": "Peer Software", "description": "Another piece of software at the same layer that may act as a functional user across the measured boundary."}
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

    # CONCEPTS
    def load_concepts(self):
        logger.info("Loading concepts...")

        concepts = [
            {
                "name": "Frequency of Occurrence",
                "type": "data_group_distinction_criterion",
                "description": "Sets of data attributes with different frequencies of occurrence describe different objects of interest."
            },
            {
                "name": "Identifying Key",
                "type": "data_group_distinction_criterion",
                "description": "Sets of data attributes with the same frequency but different identifying key attributes describe different objects of interest."
            },
            {
                "name": "Complex Output",
                "type": "data_group_case",
                "description": "A complex output such as a report may contain multiple data groups describing different objects of interest."
            },
            {
                "name": "Single Attribute Data Group",
                "type": "data_group_case",
                "description": "A data group may contain only one data attribute if this is sufficient from the perspective of the Functional User Requirements."
            },
            {
                "name": "Real-Time Single Attribute Example",
                "type": "data_group_case",
                "description": "Single-attribute data groups occur commonly in real-time software, for example a clock tick or sensor state."
            },
            {
                "name": "State Diagram",
                "type": "triggering_event_source",
                "description": "Triggering events may be identified from state diagrams when state transitions correspond to events the software must respond to."
            },
            {
                "name": "Entity Life-Cycle Diagram",
                "type": "triggering_event_source",
                "description": "Triggering events may be identified from entity life-cycle diagrams when lifecycle transitions correspond to events the software must respond to."
            },
            {
                "name": "Logical Triggering Entry",
                "type": "functional_process_identification",
                "description": "The triggering Entry is identified at the logical COSMIC level and must not be confused with preliminary navigation or interface control actions."
            },
            {
                "name": "Movement Uniqueness Key",
                "type": "data_movement_counting",
                "description": "A movement is distinguished within a functional process by movement type, moved data group or object of interest, and any explicit distinction required by the FUR."
            },
            {
                "name": "Associated Data Manipulation",
                "type": "data_movement_scope",
                "description": "Validation, formatting, calculation or transformation directly associated with a movement is included in that movement and does not count separately."
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

    # ENTITY TYPES
    def load_entity_types(self):
        logger.info("Loading entity types...")

        entity_types = [
            {
                "name": "Functional User",
                "description": "Sender and/or intended recipient of data across the boundary."
            },
            {
                "name": "Functional Process",
                "description": "Process inside the measured software that receives Entries, sends Exits and interacts with persistent storage."
            },
            {
                "name": "Persistent Storage",
                "description": "Storage on the software side of the boundary that preserves data beyond the current functional process occurrence or shares it with another occurrence or process."
            },
            {
                "name": "Boundary",
                "description": "Conceptual separation between the functional user side and the software side."
            },
            {
                "name": "Clock",
                "description": "Functional user when a clock tick is a triggering event."
            },
            {
                "name": "Hardware Functional User",
                "description": "Hardware device acting as a functional user when it sends or receives data across the boundary."
            },
            {
                "name": "Software Functional User",
                "description": "External software acting as a functional user when it sends or receives data across the boundary."
            },
            {
                "name": "Human Functional User",
                "description": "Human user acting as a functional user when sending or receiving data across the boundary."
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

    # MOVEMENT TYPES
    def load_movement_types(self):
        logger.info("Loading movement types...")

        movements = [
            {
                "name": "Entry",
                "abbreviation": "E",
                "description": "Moves a data group from a functional user across the boundary into the functional process where it is required.",
                "source": "Functional User",
                "destination": "Functional Process"
            },
            {
                "name": "Exit",
                "abbreviation": "X",
                "description": "Moves a data group from a functional process across the boundary to the functional user that requires it.",
                "source": "Functional Process",
                "destination": "Functional User"
            },
            {
                "name": "Read",
                "abbreviation": "R",
                "description": "Moves a data group from persistent storage within reach of the functional process that requires it.",
                "source": "Persistent Storage",
                "destination": "Functional Process"
            },
            {
                "name": "Write",
                "abbreviation": "W",
                "description": "Moves a data group from a functional process to persistent storage.",
                "source": "Functional Process",
                "destination": "Persistent Storage"
            }
        ]

        for item in movements:
            self.run(
                """
                MERGE (m:MovementType {name: $name})
                SET m.abbreviation = $abbreviation,
                    m.description = $description,
                    m.size_cfp = 1

                MERGE (src:EntityType {name: $source})
                MERGE (dst:EntityType {name: $destination})
                MERGE (m)-[:HAS_SOURCE]->(src)
                MERGE (m)-[:HAS_DESTINATION]->(dst)

                MERGE (dm:ComponentType {name: "DataMovement"})
                MERGE (m)-[:IS_TYPE_OF]->(dm)

                MERGE (c:ComponentType {name: $name})
                MERGE (m)-[:REPRESENTS_COMPONENT]->(c)
                """,
                item
            )

    # RULES
    def load_rules(self):
        logger.info("Loading rules...")

        rules = [
            {
                "id": "RULE_FU_IDENTIFICATION",
                "title": "Functional user identification",
                "content": "Identify all functional users that trigger, provide information to, or receive information from functional processes in the FUR.",
                "applies_to": ["FunctionalUser", "FunctionalProcess", "FUR"]
            },
            {
                "id": "RULE_STORAGE_NOT_FU",
                "title": "Persistent storage is not a functional user",
                "content": "Persistent storage is on the software side of the boundary and is not considered a functional user of the software being measured.",
                "applies_to": ["PersistentStorage", "FunctionalUser", "Boundary"]
            },
            {
                "id": "RULE_FU_PER_FP",
                "title": "Functional users are identified per functional process",
                "content": "A sender or receiver of data may be a functional user for one or more functional processes but not necessarily for other functional processes in the same software.",
                "applies_to": ["FunctionalUser", "FunctionalProcess"]
            },
            {
                "id": "RULE_FP_IDENTIFICATION",
                "title": "Functional process identification",
                "content": "Each functional process is derived from at least one identifiable FUR, is initiated by an Entry from a functional user, and comprises at least two data movements: always one Entry plus either an Exit or a Write.",
                "applies_to": ["FunctionalProcess", "FUR", "Entry", "Exit", "Write", "FunctionalUser"]
            },
            {
                "id": "RULE_TRIGGERING_CHAIN",
                "title": "Triggering event to functional process chain",
                "content": "A triggering event causes a functional user to generate a data group that is moved by the triggering Entry to start the functional process.",
                "applies_to": ["TriggeringEvent", "FunctionalUser", "TriggeringEntry", "DataGroup", "FunctionalProcess"]
            },
            {
                "id": "RULE_DG_IDENTIFICATION",
                "title": "Data group identification",
                "content": "Each data group is unique and distinguishable through its unique collection of data attributes and is directly related to one object of interest described in the FUR.",
                "applies_to": ["DataGroup", "DataAttribute", "ObjectOfInterest", "FUR"]
            },
            {
                "id": "RULE_DG_NOT_INTERNAL",
                "title": "Internal data is not a data group",
                "content": "Constants, internal variables, intermediate calculation results, and data stored only because of implementation rather than the FUR are not data groups.",
                "applies_to": ["DataGroup", "DataManipulation", "FUR"]
            },
            {
                "id": "RULE_DG_SCREEN_REPORT_FILTER",
                "title": "Screen or report data unrelated to an object of interest",
                "content": "Data appearing on input/output screens or reports that is not related to an object of interest to a functional user is not identified as a data group.",
                "applies_to": ["DataGroup", "ObjectOfInterest", "FunctionalUser"]
            },
            {
                "id": "RULE_SUBPROCESS",
                "title": "Sub-process",
                "content": "A sub-process is part of a functional process that either moves data or manipulates data.",
                "applies_to": ["SubProcess", "FunctionalProcess", "DataMovement", "DataManipulation"]
            },
            {
                "id": "RULE_DATA_MOVEMENT_IDENTIFICATION",
                "title": "Data movement identification",
                "content": "Each functional process is partitioned into data movements: Entry, Exit, Read and Write.",
                "applies_to": ["FunctionalProcess", "DataMovement", "Entry", "Exit", "Read", "Write"]
            },
            {
                "id": "RULE_ENTRY",
                "title": "Entry",
                "content": "An Entry moves one data group from a functional user across the boundary into the functional process.",
                "applies_to": ["Entry", "FunctionalUser", "Boundary", "DataGroup", "FunctionalProcess"]
            },
            {
                "id": "RULE_SINGLE_ENTRY",
                "title": "Single Entry per object of interest",
                "content": "For any one functional process, a single Entry is counted for the entry of all data describing one object of interest unless the FUR explicitly require otherwise.",
                "applies_to": ["Entry", "FunctionalProcess", "DataGroup", "ObjectOfInterest"]
            },
            {
                "id": "RULE_EXIT",
                "title": "Exit",
                "content": "An Exit moves one data group from a functional process across the boundary to the functional user that requires it.",
                "applies_to": ["Exit", "FunctionalProcess", "Boundary", "FunctionalUser", "DataGroup"]
            },
            {
                "id": "RULE_READ",
                "title": "Read",
                "content": "A Read moves one data group from persistent storage within reach of the functional process that requires it.",
                "applies_to": ["Read", "PersistentStorage", "FunctionalProcess", "DataGroup"]
            },
            {
                "id": "RULE_WRITE",
                "title": "Write",
                "content": "A Write moves one data group from a functional process to persistent storage.",
                "applies_to": ["Write", "FunctionalProcess", "PersistentStorage", "DataGroup"]
            },
            {
                "id": "RULE_SINGLE_EXIT_READ_WRITE",
                "title": "Single Exit, Read or Write per object of interest",
                "content": "A single Exit, Read or Write is counted for the movement of all data describing one object of interest unless the FUR explicitly require otherwise.",
                "applies_to": ["Exit", "Read", "Write", "FunctionalProcess", "DataGroup", "ObjectOfInterest"]
            },
            {
                "id": "RULE_OCCURRENCES",
                "title": "Movement occurrences",
                "content": "Within one functional process, repeated occurrences of the same movement type for the same data group are counted once, unless the FUR explicitly require distinct movements.",
                "applies_to": ["DataMovement", "FunctionalProcess", "Entry", "Exit", "Read", "Write"]
            },
            {
                "id": "RULE_ERROR_CONFIRMATION_EXIT",
                "title": "Error and confirmation messages are Exits",
                "content": "An error or confirmation message is an Exit issued by a functional process to a human user that confirms entered data has been accepted or reports an error in entered data.",
                "applies_to": ["ErrorConfirmationMessage", "Exit", "FunctionalProcess"]
            },
            {
                "id": "RULE_ONE_EXIT_FOR_ERROR_CONFIRMATION",
                "title": "One Exit for error and confirmation messages",
                "content": "One Exit is identified to account for all types of error or confirmation messages issued by one functional process from all possible causes according to its FUR.",
                "applies_to": ["ErrorConfirmationMessage", "Exit", "FunctionalProcess", "FUR"]
            },
            {
                "id": "RULE_ERROR_ADDITIONAL_DATA",
                "title": "Additional data in error or confirmation message",
                "content": "If a message to a human functional user provides data in addition to confirming acceptance or reporting error, the additional data is identified as a separate data group moved by an Exit.",
                "applies_to": ["ErrorConfirmationMessage", "Exit", "DataGroup", "FunctionalUser"]
            },
            {
                "id": "RULE_READ_WRITE_ERROR_CONDITIONS",
                "title": "Read and Write include associated error conditions",
                "content": "Reads and Writes account for associated reporting of error conditions; no Entry is identified for an error indication received as a result of a Read or Write of persistent data.",
                "applies_to": ["Read", "Write", "ErrorConfirmationMessage", "PersistentStorage"]
            },
            {
                "id": "RULE_ASSOCIATED_MANIPULATION",
                "title": "Associated data manipulation is included in the movement",
                "content": "Validation, formatting, calculation, comparison or transformation directly associated with an Entry, Exit, Read or Write is included in that movement and is not counted as a separate movement. Access to another data group or persistent storage may require an additional movement.",
                "applies_to": ["DataManipulation", "DataMovement", "Entry", "Exit", "Read", "Write"]
            },
            {
                "id": "RULE_CLOCK_TRIGGER_ONLY",
                "title": "Clock as functional user",
                "content": "A clock is a functional user when a clock tick is a triggering event. Merely obtaining the current date or time is not automatically a COSMIC data movement.",
                "applies_to": ["FunctionalUser", "TriggeringEvent", "TriggeringEntry", "DataMovement"]
            },
            {
                "id": "RULE_CONTROL_COMMAND_NOT_COUNTED",
                "title": "Control commands are not data movements",
                "content": "A control command that only controls use of the software and does not move data about an object of interest is not counted as an Entry or any other COSMIC data movement.",
                "applies_to": ["ControlCommand", "DataMovement", "Entry", "ObjectOfInterest"]
            },
            {
                "id": "RULE_MEASUREMENT_STRATEGY",
                "title": "Establish measurement strategy before mapping",
                "content": "Define the measurement purpose, scope, software boundary, layer, level of decomposition and level of granularity before identifying functional users and data movements.",
                "applies_to": ["MeasurementPurpose", "MeasurementScope", "Boundary", "Layer", "LevelOfDecomposition", "LevelOfGranularity", "PieceOfSoftware"]
            },
            {
                "id": "RULE_FUR_ONLY",
                "title": "Measure functional user requirements only",
                "content": "COSMIC functional size measures Functional User Requirements. Non-functional, technical and implementation requirements are not counted unless they express required functional data movements.",
                "applies_to": ["FUR", "DataMovement", "FunctionalProcess"]
            },
            {
                "id": "RULE_DATA_ATTRIBUTES_OPTIONAL",
                "title": "Explicit data attribute extraction is optional",
                "content": "Data attributes support the identification and distinction of data groups, but explicit attribute enumeration is not mandatory when the data group and its object of interest are unambiguous.",
                "applies_to": ["DataAttribute", "DataGroup", "ObjectOfInterest"]
            },
            {
                "id": "RULE_CFP_UNIT",
                "title": "Data movement size",
                "content": "Each Entry, Exit, Read or Write has size 1 CFP.",
                "applies_to": ["Entry", "Exit", "Read", "Write", "DataMovement"]
            }
        ]

        for item in rules:
            self.run(
                """
                MERGE (r:Rule {id: $id})
                SET r.title = $title,
                    r.content = $content
                """,
                {
                    "id": item["id"],
                    "title": item["title"],
                    "content": item["content"]
                }
            )

            for component in item["applies_to"]:
                self.run(
                    """
                    MATCH (r:Rule {id: $rule_id})
                    MERGE (c:ComponentType {name: $component})
                    MERGE (r)-[:APPLIES_TO]->(c)
                    """,
                    {
                        "rule_id": item["id"],
                        "component": component
                    }
                )

    # GUIDELINES
    def load_guidelines(self):
        logger.info("Loading guidelines...")

        guidelines = [
            {
                "id": "GUIDE_FP_IDENTIFICATION_STEPS",
                "title": "Steps to identify functional processes",
                "content": "Identify separate events in the world of functional users, identify which functional users respond to each event, identify the triggering Entry data groups generated by those users, then identify the functional process started by each triggering Entry.",
                "applies_to": ["TriggeringEvent", "FunctionalUser", "TriggeringEntry", "FunctionalProcess"]
            },
            {
                "id": "GUIDE_TRIGGER_EVENTS_FROM_MODELS",
                "title": "Triggering events from state and lifecycle diagrams",
                "content": "Triggering events can be identified in state diagrams and entity life-cycle diagrams when transitions correspond to events to which the software must react.",
                "applies_to": ["TriggeringEvent", "FunctionalProcess"]
            },
            {
                "id": "GUIDE_DG_FREQUENCY",
                "title": "Different frequencies indicate different objects of interest",
                "content": "Sets of data attributes with different frequencies of occurrence describe different objects of interest.",
                "applies_to": ["DataGroup", "ObjectOfInterest", "DataAttribute"]
            },
            {
                "id": "GUIDE_DG_IDENTIFYING_KEYS",
                "title": "Different identifying keys indicate different objects of interest",
                "content": "Sets of data attributes with the same frequency of occurrence but different identifying key attributes describe different objects of interest.",
                "applies_to": ["DataGroup", "ObjectOfInterest", "DataAttribute"]
            },
            {
                "id": "GUIDE_DG_SAME_GROUP_AFTER_CRITERIA",
                "title": "Attributes remaining after frequency and key criteria belong to one data group",
                "content": "All data attributes in a set resulting from applying the frequency and identifying-key criteria belong to the same data group unless the FUR specify otherwise.",
                "applies_to": ["DataGroup", "DataAttribute", "ObjectOfInterest", "FUR"]
            },
            {
                "id": "GUIDE_FU_MAY_BE_OOI",
                "title": "Functional user may be object of interest",
                "content": "A functional user of the software may be the object of interest of a data group sent or received by the functional user.",
                "applies_to": ["FunctionalUser", "ObjectOfInterest", "DataGroup"]
            },
            {
                "id": "GUIDE_SINGLE_ATTRIBUTE_REAL_TIME_DG",
                "title": "Single-attribute data groups",
                "content": "A data group may contain only one data attribute if this is all that is required from the FUR perspective; this occurs commonly in real-time software such as clock ticks or sensor states.",
                "applies_to": ["DataGroup", "DataAttribute", "TriggeringEntry"]
            },
            {
                "id": "GUIDE_OOI_CONTEXT_DEPENDENT",
                "title": "Object of interest is identified per functional process",
                "content": "A thing may be an object of interest for one or more functional processes but not for others, even in the same software being measured.",
                "applies_to": ["ObjectOfInterest", "FunctionalProcess"]
            },
            {
                "id": "GUIDE_DG_ORIGINS",
                "title": "Origins of data groups",
                "content": "A data group may originate from a storage record, volatile memory structure, clustered screen or report presentation, message between a device and a computer, or network transmission.",
                "applies_to": ["DataGroup", "PersistentStorage", "DataAttribute"]
            },
            {
                "id": "GUIDE_MULTIPLE_OOI_IN_INPUT_OUTPUT",
                "title": "Multiple objects of interest require multiple data groups",
                "content": "If attributes moved in or out of a functional process describe several objects of interest, separate data groups and corresponding data movements must be identified.",
                "applies_to": ["DataGroup", "ObjectOfInterest", "DataMovement"]
            },
            {
                "id": "GUIDE_COMPLEX_OUTPUTS",
                "title": "Complex outputs and reports",
                "content": "When analyzing complex outputs such as reports, each separate candidate data group describing a different object of interest must be distinguished and counted.",
                "applies_to": ["Complex Output", "DataGroup", "ObjectOfInterest", "Exit"]
            },
            {
                "id": "GUIDE_NO_DG_FROM_DATA_MANIPULATION",
                "title": "No data groups from data manipulation",
                "content": "No data groups are identified from data manipulation inside a functional process in addition to the data groups moved by Entries, Exits, Reads and Writes.",
                "applies_to": ["DataManipulation", "DataGroup", "Entry", "Exit", "Read", "Write"]
            },
            {
                "id": "GUIDE_ERROR_CONFIRMATION_MESSAGES",
                "title": "Error and confirmation messages",
                "content": "Error and confirmation messages are specific forms of Exit. One Exit accounts for all such messages issued by one functional process from all possible causes according to its FUR.",
                "applies_to": ["ErrorConfirmationMessage", "Exit", "FunctionalProcess"]
            },
            {
                "id": "GUIDE_OTHER_ERROR_DATA_NORMAL_ENTRY_EXIT",
                "title": "Other error-condition data follows normal Entry/Exit rules",
                "content": "All other data issued or received by the software to or from hardware or software functional users should be analyzed as Exits or Entries according to the normal COSMIC rules, regardless of whether values indicate an error condition.",
                "applies_to": ["Entry", "Exit", "FunctionalUser", "ErrorConfirmationMessage"]
            },
            {
                "id": "GUIDE_TRIGGERING_ENTRY_LOGICAL_LEVEL",
                "title": "Identify the triggering Entry at the logical level",
                "content": "Menu selection, screen navigation or requesting an empty form does not automatically create a separate Entry. Identify the logical Entry that moves the data group needed to start the functional process.",
                "applies_to": ["TriggeringEntry", "Entry", "ControlCommand", "FunctionalProcess"]
            },
            {
                "id": "GUIDE_PERSISTENT_STORAGE_TEST",
                "title": "Distinguish persistent storage from temporary memory",
                "content": "Treat storage as persistent when data survives the current functional process occurrence or is available to another occurrence or process. Temporary variables and intermediate values are not persistent storage.",
                "applies_to": ["PersistentStorage", "DataManipulation", "Read", "Write"]
            },
            {
                "id": "GUIDE_MOVEMENT_DEDUP_KEY",
                "title": "Deduplicate movements using type and data group",
                "content": "Deduplicate only repeated occurrences that share the same functional process, movement type and data group. Do not merge movements of different data groups, and preserve distinctions explicitly required by the FUR.",
                "applies_to": ["FunctionalProcess", "DataMovement", "DataGroup", "FUR"]
            },
            {
                "id": "GUIDE_OS_ERROR_NOT_COUNTED",
                "title": "External error messages not required by FUR are not counted",
                "content": "No Entry or Exit is identified for an error message issued while using the software if the message is not required to be processed by the FUR, such as an operating-system error message.",
                "applies_to": ["Entry", "Exit", "ErrorConfirmationMessage", "FUR"]
            }
        ]

        for item in guidelines:
            self.run(
                """
                MERGE (g:Guideline {id: $id})
                SET g.title = $title,
                    g.content = $content

                """,
                {
                    "id": item["id"],
                    "title": item["title"],
                    "content": item["content"]
                }
            )

            for component in item["applies_to"]:
                self.run(
                    """
                    MATCH (g:Guideline {id: $guideline_id})
                    MERGE (c:ComponentType {name: $component})
                    MERGE (g)-[:APPLIES_TO]->(c)
                    """,
                    {
                        "guideline_id": item["id"],
                        "component": component
                    }
                )

    # VALIDATION RULES
    def load_validation_rules(self):
        logger.info("Loading validation rules...")

        validation_rules = [
            {
                "id": "VALIDATE_NO_STORAGE_AS_FU",
                "title": "Reject persistent storage as functional user",
                "content": "Persistent storage must not be extracted as a functional user.",
                "applies_to": ["FunctionalUser", "PersistentStorage"]
            },
            {
                "id": "VALIDATE_FP_MINIMUM_MOVEMENTS",
                "title": "Functional process minimum structure",
                "content": "A functional process must have at least one Entry and at least one Exit or Write.",
                "applies_to": ["FunctionalProcess", "Entry", "Exit", "Write"]
            },
            {
                "id": "VALIDATE_ONE_DG_PER_MOVEMENT",
                "title": "One data group per movement",
                "content": "Each data movement must move one data group.",
                "applies_to": ["DataMovement", "DataGroup"]
            },
            {
                "id": "VALIDATE_ONE_OOI_PER_DG",
                "title": "One object of interest per data group",
                "content": "Each data group must describe one object of interest.",
                "applies_to": ["DataGroup", "ObjectOfInterest"]
            },
            {
                "id": "VALIDATE_IGNORE_INTERNAL_MANIPULATION",
                "title": "Ignore internal data manipulation",
                "content": "Internal data manipulation should not be counted as a separate COSMIC data movement.",
                "applies_to": ["DataManipulation", "DataMovement"]
            },
            {
                "id": "VALIDATE_ERROR_CONFIRMATION_DEDUP",
                "title": "Deduplicate error and confirmation Exits",
                "content": "All error and confirmation messages for one functional process should count as one Exit unless additional business data is provided.",
                "applies_to": ["ErrorConfirmationMessage", "Exit", "FunctionalProcess"]
            },
            {
                "id": "VALIDATE_MOVEMENT_DEDUP_KEY",
                "title": "Deduplicate movements safely",
                "content": "Merge repeated movements only when functional process, movement type and data group are the same and the FUR do not explicitly require distinct movements.",
                "applies_to": ["FunctionalProcess", "DataMovement", "DataGroup", "FUR"]
            },
            {
                "id": "VALIDATE_CONTROL_COMMAND_EXCLUSION",
                "title": "Reject pure control commands as Entries",
                "content": "A navigation or control action that moves no data about an object of interest must not be classified as an Entry.",
                "applies_to": ["ControlCommand", "Entry", "ObjectOfInterest"]
            },
            {
                "id": "VALIDATE_CLOCK_USAGE",
                "title": "Validate clock-triggered processes",
                "content": "Classify Clock as a functional user only when a clock tick triggers the functional process; ordinary date or time retrieval is not an Entry.",
                "applies_to": ["FunctionalUser", "TriggeringEvent", "Entry"]
            },
            {
                "id": "VALIDATE_MEASUREMENT_CONTEXT",
                "title": "Require measurement context",
                "content": "Functional user and movement classification must be interpreted using the declared measurement scope and software boundary.",
                "applies_to": ["MeasurementScope", "Boundary", "FunctionalUser", "DataMovement"]
            }
        ]

        for item in validation_rules:
            self.run(
                """
                MERGE (v:ValidationRule {id: $id})
                SET v.title = $title,
                    v.content = $content
                """,
                {
                    "id": item["id"],
                    "title": item["title"],
                    "content": item["content"]
                }
            )

            for component in item["applies_to"]:
                self.run(
                    """
                    MATCH (v:ValidationRule {id: $validation_id})
                    MERGE (c:ComponentType {name: $component})
                    MERGE (v)-[:APPLIES_TO]->(c)
                    """,
                    {
                        "validation_id": item["id"],
                        "component": component
                    }
                )

    # SEMANTIC RELATIONS
    def load_semantic_relations(self):
        logger.info("Loading semantic relations...")

        queries = [
            """
            MATCH (fu:ComponentType {name: "FunctionalUser"})
            MATCH (fp:ComponentType {name: "FunctionalProcess"})
            MERGE (fu)-[:IDENTIFIED_PER]->(fp)
            """,
            """
            MATCH (te:ComponentType {name: "TriggeringEvent"})
            MATCH (fu:ComponentType {name: "FunctionalUser"})
            MATCH (ten:ComponentType {name: "TriggeringEntry"})
            MATCH (dg:ComponentType {name: "DataGroup"})
            MATCH (fp:ComponentType {name: "FunctionalProcess"})
            MERGE (te)-[:CAUSES_RESPONSE_FROM]->(fu)
            MERGE (fu)-[:GENERATES]->(dg)
            MERGE (dg)-[:MOVED_BY]->(ten)
            MERGE (ten)-[:STARTS]->(fp)
            """,
            """
            MATCH (fp:ComponentType {name: "FunctionalProcess"})
            MATCH (ten:ComponentType {name: "TriggeringEntry"})
            MATCH (dm:ComponentType {name: "DataMovement"})
            MATCH (entry:MovementType {name: "Entry"})
            MERGE (fp)-[:MUST_BE_INITIATED_BY]->(ten)
            MERGE (fp)-[:MUST_BE_PARTITIONED_INTO]->(dm)
            MERGE (fp)-[:MUST_HAVE]->(entry)
            """,
            """
            MERGE (group:RequirementGroup {
                id: "FP_COMPLETION_MOVEMENT",
                operator: "OR",
                min_required: 1,
                description: "A functional process must contain at least one Exit or one Write in addition to its Entry."
            })
            WITH group
            MATCH (fp:ComponentType {name: "FunctionalProcess"})
            MATCH (exit:MovementType {name: "Exit"})
            MATCH (write:MovementType {name: "Write"})
            MERGE (fp)-[:MUST_SATISFY]->(group)
            MERGE (group)-[:HAS_OPTION]->(exit)
            MERGE (group)-[:HAS_OPTION]->(write)
            """,
            """
            MATCH (ten:ComponentType {name: "TriggeringEntry"})
            MATCH (entry:MovementType {name: "Entry"})
            MERGE (ten)-[:IS_A]->(entry)
            """,
            """
            MATCH (dg:ComponentType {name: "DataGroup"})
            MATCH (ooi:ComponentType {name: "ObjectOfInterest"})
            MATCH (attr:ComponentType {name: "DataAttribute"})
            MERGE (dg)-[:DESCRIBES_EXACTLY_ONE]->(ooi)
            MERGE (dg)-[:COMPOSED_OF]->(attr)
            """,
            """
            MATCH (dg:ComponentType {name: "DataGroup"})
            MATCH (freq:Concept {name: "Frequency of Occurrence"})
            MATCH (key:Concept {name: "Identifying Key"})
            MERGE (dg)-[:DISTINGUISHED_BY]->(freq)
            MERGE (dg)-[:DISTINGUISHED_BY]->(key)
            """,
            """
            MATCH (ooi:ComponentType {name: "ObjectOfInterest"})
            MATCH (fp:ComponentType {name: "FunctionalProcess"})
            MERGE (ooi)-[:IDENTIFIED_PER]->(fp)
            """,
            """
            MATCH (fu:ComponentType {name: "FunctionalUser"})
            MATCH (ooi:ComponentType {name: "ObjectOfInterest"})
            MERGE (fu)-[:MAY_BE]->(ooi)
            """,
            """
            MATCH (dm:ComponentType {name: "DataMovement"})
            MATCH (dg:ComponentType {name: "DataGroup"})
            MATCH (manip:ComponentType {name: "DataManipulation"})
            MERGE (dm)-[:MOVES_EXACTLY_ONE]->(dg)
            MERGE (dm)-[:INCLUDES_ASSOCIATED]->(manip)
            """,
            """
            MATCH (sp:ComponentType {name: "SubProcess"})
            MATCH (dm:ComponentType {name: "DataMovement"})
            MATCH (manip:ComponentType {name: "DataManipulation"})
            MERGE (sp)-[:CAN_BE]->(dm)
            MERGE (sp)-[:CAN_BE]->(manip)
            """,
            """
            MATCH (cmd:ComponentType {name: "ControlCommand"})
            MATCH (dm:ComponentType {name: "DataMovement"})
            MERGE (cmd)-[:NOT_COUNTED_AS]->(dm)
            """,
            """
            MATCH (purpose:ComponentType {name: "MeasurementPurpose"})
            MATCH (scope:ComponentType {name: "MeasurementScope"})
            MATCH (pos:ComponentType {name: "PieceOfSoftware"})
            MATCH (boundary:ComponentType {name: "Boundary"})
            MATCH (layer:ComponentType {name: "Layer"})
            MATCH (decomp:ComponentType {name: "LevelOfDecomposition"})
            MATCH (gran:ComponentType {name: "LevelOfGranularity"})
            MATCH (fur:ComponentType {name: "FUR"})
            MERGE (purpose)-[:DETERMINES]->(scope)
            MERGE (scope)-[:SELECTS]->(fur)
            MERGE (scope)-[:DEFINES_MEASURED]->(pos)
            MERGE (scope)-[:ESTABLISHES]->(boundary)
            MERGE (scope)-[:USES]->(layer)
            MERGE (scope)-[:USES]->(decomp)
            MERGE (scope)-[:USES]->(gran)
            """,
            """
            MATCH (msg:ComponentType {name: "ErrorConfirmationMessage"})
            MATCH (exit:MovementType {name: "Exit"})
            MERGE (msg)-[:CLASSIFIED_AS]->(exit)
            """,
            """
            MATCH (read:MovementType {name: "Read"})
            MATCH (write:MovementType {name: "Write"})
            MATCH (msg:ComponentType {name: "ErrorConfirmationMessage"})
            MERGE (read)-[:ACCOUNTS_FOR_ASSOCIATED_ERROR_CONDITIONS]->(msg)
            MERGE (write)-[:ACCOUNTS_FOR_ASSOCIATED_ERROR_CONDITIONS]->(msg)
            """,
            """
            MATCH (rule:Rule {id: "RULE_CFP_UNIT"})
            MATCH (entry:MovementType {name: "Entry"})
            MATCH (exit:MovementType {name: "Exit"})
            MATCH (read:MovementType {name: "Read"})
            MATCH (write:MovementType {name: "Write"})
            MERGE (rule)-[:ASSIGNS_ONE_CFP_TO]->(entry)
            MERGE (rule)-[:ASSIGNS_ONE_CFP_TO]->(exit)
            MERGE (rule)-[:ASSIGNS_ONE_CFP_TO]->(read)
            MERGE (rule)-[:ASSIGNS_ONE_CFP_TO]->(write)
            """
        ]

        for query in queries:
            self.run(query)

    # VERIFICATION
    def verify_graph(self):
        logger.info("Verifying graph content...")

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

        with self.driver.session() as session:
            print("\n=== Graph content ===")
            for record in session.run(node_query):
                print(f"{record['labels']}: {record['count']}")

            print("\n=== Relationships ===")
            for record in session.run(rel_query):
                print(f"{record['relationship']}: {record['count']}")

    # LOAD ALL
    def load_all(self):
        self.reset_database()
        self.create_constraints()
        self.load_component_types()
        self.load_concepts()
        self.load_entity_types()
        self.load_movement_types()
        self.load_rules()
        self.load_guidelines()
        self.load_validation_rules()
        self.load_semantic_relations()
        self.verify_graph()


if __name__ == "__main__":
    loader = CosmicCoreOntologyLoader()

    try:
        loader.load_all()
        print("\nCOSMIC core ontology replaced successfully with selected user-provided information only.")

    finally:
        loader.close()
