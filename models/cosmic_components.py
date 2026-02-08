from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class DataMovementType(Enum):
    ENTRY = "Entry"
    READ = "Read" 
    WRITE = "Write"
    EXIT = "Exit"

@dataclass
class FunctionalProcess:
    name: str
    triggering_event: str
    functional_user: str
    triggering_entry: str

@dataclass
class DataGroup:
    name: str
    object_of_interest: str
    attributes: List[str]
    frequency: Optional[str] = None
    key_attribute: Optional[str] = None

@dataclass
class DataMovement:
    action_verb: str
    data_group: str
    object_of_interest: str
    source: str
    destination: str
    movement_type: DataMovementType
    functional_process: str = ""

@dataclass
class SubProcess:
    process_name: str     
    step_name: str         
    action_verb: str       
    data_group: str 
    object_of_interest: str      
    source: str          
    destination: str   

@dataclass
class CosmicComponents:
    functional_processes: List[FunctionalProcess]
    functional_users: List[str]
    data_groups: List[DataGroup]
    sub_processes: List[SubProcess]
    data_movements: List[DataMovement]

    
    