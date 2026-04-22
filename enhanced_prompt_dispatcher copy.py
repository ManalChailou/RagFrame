import json
import re
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional
import logging
from rag_system import CosmicRAGSystem
from llm_router import LLMConfig, BuildLLM

logger = logging.getLogger(__name__)
load_dotenv() 

class EnhancedPromptDispatcher:
    
    def __init__(self, llm_provider: str = "openai", llm_model: str = "gpt-4", llm_base_url: Optional[str] = None):
        self.temperature = 0.2
        self._app_domain = ""

        # NEW: LLM instance
        self.llm_cfg = LLMConfig(
            provider=llm_provider,
            model=llm_model,
            temperature=self.temperature,
            base_url=llm_base_url
        )
        self.llm = BuildLLM(self.llm_cfg)
        
    
    def set_llm(self, provider: str, model: str, base_url: Optional[str] = None, temperature: Optional[float] = None):
        """Allow changing LLM between runs"""
        if temperature is not None:
            self.temperature = temperature
        self.llm_cfg = LLMConfig(
            provider=provider,
            model=model,
            temperature=self.temperature,
            base_url=base_url
        )
        self.llm = BuildLLM(self.llm_cfg)

        
    def create_functional_users_prompt(self, requirements: List[str]) -> str:
        base_prompt = f"""
Role: You are a software measurement expert specializing in the COSMIC method for functional size measurement.

Task Description: Your task involves identifying the functional users in the provided Functional User Requirements (FUR).
"""
        # Ajouter le contexte RAG
        rag_context = ""
        if self.rag_system:
            rag_context = self.rag_system.get_context_for_functional_users(requirements, app_domain=self._app_domain)
            base_prompt += f"\n{rag_context}\n"
        
        self.last_rag_contexts["functional_users"] = rag_context
        base_prompt += f"""
Step-by-Step Instructions:
1. Identify all entities that send data TO the system.
2. Identify all entities that receive data FROM the system.
3. Include human users, external systems, and other software components that interact with the system via data exchanges.
4. Do not include internal modules or components unless they represent external interfaces.

Expected Output Format: Present the result in this Python dictionary format:
  functional_users_format = {{
    "FunctionalUsers": ["User", "ExternalSystem"]
  }}

Example:
> "When viewing grades, the student provides the student ID to the system"
Identified functional users:
{{
  "FunctionalUsers": ["Student"]
}}

Output MUST be a single JSON object only. No prose, no markdown, no code fences. If you add anything else, the answer will be rejected

Requirements:
{json.dumps(requirements, indent=2)}
"""
        return base_prompt

    def create_functional_process_prompt(self, requirements: List[str], functional_users: List[str]) -> str:
        base_prompt = f"""
Role: You are a software measurement expert specializing in the COSMIC method for functional size measurement.

Task Description: Your task involves identifying functional processes from the provided Functional User Requirements (FUR).
"""
        # Ajouter le contexte RAG
        rag_context = ""
        if self.rag_system:
            rag_context = self.rag_system.get_context_for_functional_processes(requirements, app_domain=self._app_domain)
            base_prompt += f"\n{rag_context}\n"

        self.last_rag_contexts["functional_processes"] = rag_context
        allowed_fu = functional_users or []

        base_prompt += f"""
Step-by-Step Instructions:
1. Identify the Triggering Events: Distinct events in the world of the functional users that the software must respond to.
2. Identify the Functional User types: Users who respond to each triggering event.
3. Identify the Triggering Entry: Data that each functional user generates in response.
4. Identify the Functional Process: Initiated by each triggering entry, includes all necessary operations to fulfill the FUR.

FunctionalUser MUST be chosen EXACTLY from this allowed set : {json.dumps(allowed_fu)}

IMPORTANT COSMIC GRANULARITY RULES:
- Each triggering EVENT should result in ONE functional process
- Multiple related activities from same event = sub-processes within same functional process
- Same user performing related tasks triggered by same event = single functional process

Expected Output Format:
  functional_process_format = {{
    "Process-Name": {{
      "TriggeringEvents": "..",
      "FunctionalUser": "..",
      "TriggeringEntry": ".."
    }}
  }}

Example:
> "When viewing grades, the student provides the student ID to the system..."

Identified functional process:
{{
  "View-Grade": {{
    "TriggeringEvents": "Student request",
    "FunctionalUser": "Student",
    "TriggeringEntry": "Student ID"
  }}
}}

Output MUST be a single JSON object only. No prose, no markdown, no code fences. If you add anything else, the answer will be rejected

Requirements:
{json.dumps(requirements, indent=2)}
"""
        return base_prompt

    def create_data_groups_prompt(self, requirements: List[str], functional_processes: List[Dict]) -> str:
        base_prompt = f"""
Role: You are a software measurement expert specializing in the COSMIC method for functional size measurement.

Task Description: Your task involves identifying Data Groups from the Functional User Requirements (FUR) and functional processes.
"""
        
        # Ajouter le contexte RAG
        rag_context = ""
        if self.rag_system:
            rag_context = self.rag_system.get_context_for_data_groups(requirements, functional_processes, app_domain=self._app_domain)
            base_prompt += f"\n{rag_context}\n"

        self.last_rag_contexts["data_groups"] = rag_context
        
        base_prompt += f"""
Step-by-Step Instructions:
1. Identify the Objects of Interest: Things about which the software stores or processes data.
2. Identify the related Data Attributes: Fields that describe aspects of the same Object of Interest.
3. Group attributes by Object of Interest into Data Groups.
4. Consider frequency of occurrence and key attributes to distinguish between similar groups.
5. Exclude control commands or display-only labels from data groups.
6. Each data group must describe a cohesive set of attributes about one Object of Interest.
7. Do NOT include storage media (RAM, ROM, database, cache) as Data Groups: these are Storage.


COSMIC DATA GROUP PRINCIPLES:
- Data groups represent ENTITIES in the business domain (Student, Course, Assignment, etc.)
- Generated outputs (Reports, Invoices) are separate data groups from their source data
- Each object of interest should have meaningful attributes that describe it completely

Expected Output Format: Present the result in this Python dictionary format:
  data_groups_format = {{
    "Course": {{
    "ObjectOfInterest": "Course",
    "Attributes": ["Course ID", "Name", "Credits"]
    }}
  }}

Example:
> "When managing student information, the administrator provides the student ID, name, and date of birth to the system."

Identified data groups:
{{
  "Student": {{
    "ObjectOfInterest": "Student",
    "Attributes": ["Student ID", "Name", "Date of birth"]
  }}
}}

Output MUST be a single JSON object only. No prose, no markdown, no code fences. If you add anything else, the answer will be rejected

Requirements: {json.dumps(requirements, indent=2)}
Functional Processes: {json.dumps(functional_processes, indent=2)}
"""
        return base_prompt
    
    def create_sub_processes_prompt(self, requirements: List[str], functional_processes: List[Dict], data_groups: List[Dict]) -> str:
        base_prompt = f"""
Role: You are a software measurement expert specializing in the COSMIC method for functional size measurement.

Task Description: Your task involves breaking down each functional process into sub-processes and identifying data movement components.
"""
        
        # Ajouter le contexte RAG
        rag_context = ""
        if self.rag_system:
            rag_context = self.rag_system.get_context_for_sub_processes(requirements,functional_processes,data_groups, app_domain=self._app_domain)
            base_prompt += f"\n{rag_context}\n"
        
        self.last_rag_contexts["sub_processes"] = rag_context

        base_prompt += f"""
IMPORTANT COSMIC RULES TO FOLLOW:
1. NEVER create direct movements: User ↔ Storage (always go through System)
2. NEVER create internal movements: System ↔ System (these are excluded from CFP)
3. Always decompose complex operations into elementary data movements
4. Follow the COSMIC data movement rules provided in the knowledge context above
5. For EACH sub-process, "MovedDataGroup" MUST be exactly one of the identified Data Groups (or a valid synonym referring to exactly one ObjectOfInterest).

Step-by-Step Instructions:
1. For each functional process, identify the individual steps or sub-processes needed to fulfill the Functional User Requirement.
2. For each sub-process, identify:
   - the Action Verb (describes the operation: read, write, etc.)
   - the Moved Data Group involved (which object the data belongs to)
   - the Source (origin of the data)
   - the Destination (target of the data)
3. Always decompose complex operations into elementary data movements
4. If the system performs a decision (approve/reject), generates a result (report, confirmation), or updates data (modify, approve), 
   then you MUST include a movement from System to User (or External) to deliver the result — this is an Exit movement.
5. If the system saves data to Storage, but no feedback is shown to the User, include a confirmation step from System → User (Exit).
6. Do NOT skip output movements just because the FUR doesn't say "confirmation" — always assume users expect feedback.
7. Apply Entry, Exit, Read, Write rules from the COSMIC knowledge context above.

Guidelines for Sources and Destinations:
- Functional User = "User"
- Internal system = "System"
- Storage/Persistent store = "Storage"
- External application = "External"

ACTION VERB GUIDELINES:
- Use WRITE only for: System → Storage (save, store, persist, archive)
- Use SEND/SUBMIT for: System → External (transmit, export, submit, forward)
- Use EXIT/SHOW for: System → User (display, show, present, notify)
- Use ENTER for: User → System (input, provide, submit)
- Use READ for: Storage → System (retrieve, fetch, load, query)

FORBIDDEN SYSTEM→SYSTEM MOVEMENTS:
- NEVER use "System" as both Source AND Destination
- Internal calculations, processing, analysis are NOT data movements in COSMIC
- "Calculate", "Process", "Analyze", "Identify" operations are data manipulation WITHIN movements
- These internal operations should be combined with actual data movements (Read/Write)

CORRECT PATTERNS:
WRONG: "Calculate interest" System → System (forbidden)
RIGHT: "Read account" Storage → System + "Write calculation" System → Storage

FORBIDDEN Movements (will be automatically decomposed):
- Storage → User (decompose to: Storage → System → User)
- User → Storage (decompose to: User → System → Storage)  
- System → System (internal processing, excluded from CFP)

Expected Output Format:
  sub_processes_format = {{
    "ProcessName": [
      {{
        "StepName": "Step description",
        "ActionVerb": "Verb",
        "MovedDataGroup": "GroupName",
        "ObjectOfInterest": "EntityName",
        "Source": "Entity",
        "Destination": "Entity"
      }}
    ]
  }}

Example:
> Functional Process: "View Customer Data"

CORRECT Decomposition:
{{
  "View Customer Data": [
    {{
      "StepName": "Request customer ID",
      "ActionVerb": "Enter",
      "MovedDataGroup": "Customer data",
      "ObjectOfInterest": "Customer",
      "Source": "User",
      "Destination": "System"
    }},
    {{
      "StepName": "Retrieve customer record", 
      "ActionVerb": "Read",
      "MovedDataGroup": "Customer data",
      "ObjectOfInterest": "Customer",
      "Source": "Storage", 
      "Destination": "System"
    }},
    {{
      "StepName": "Display customer data",
      "ActionVerb": "Show",
      "MovedDataGroup": "Custome data",
      "ObjectOfInterest": "Customer", 
      "Source": "System",
      "Destination": "User"
    }}
  ]
}}

Output MUST be a single JSON object only. No prose, no markdown, no code fences. If you add anything else, the answer will be rejected

Requirements: {json.dumps(requirements, indent=2)}
Functional Processes: {json.dumps(functional_processes, indent=2)}
Data Groups: {json.dumps(data_groups, indent=2)}
"""
        return base_prompt

    def extract_components(self, requirements: List[str]) -> Dict:
        """Extract all COSMIC components using sequential prompts with RAG enhancement"""
        results = {}

        try:
            # 1. Functional Users
            logger.info("Extracting Functional Users with RAG context...")
            fu_prompt = self.create_functional_users_prompt(requirements)
            raw_fu = self.llm.generate(fu_prompt)
            fu_data = self.extract_json_from_text(raw_fu)
            results.update(fu_data)

            functional_users_list = self._sanitize_functional_users(fu_data.get("FunctionalUsers", []))
            results["FunctionalUsers"] = functional_users_list

            # 2. Functional Processes
            logger.info("Extracting Functional Processes with RAG context...")
            fp_prompt = self.create_functional_process_prompt(requirements, functional_users_list)
            raw_fp = self.llm.generate(fp_prompt)
            fp_data = self.extract_json_from_text(raw_fp)
            results["functional_processes"] = [{"name": k, **v} for k, v in fp_data.items()]
            
            logger.info(f"Extracted {len(fp_data)} functional processes")

            # 3. Data Groups
            logger.info("Extracting Data Groups with RAG context...")
            dg_prompt = self.create_data_groups_prompt(requirements, results["functional_processes"])
            raw_dg = self.llm.generate(dg_prompt)
            dg_data = self.extract_json_from_text(raw_dg)
            results["data_groups"] = [{"name": k, **v} for k, v in dg_data.items()]

            # 4. Sub-Processes
            logger.info("Extracting Sub-processes with RAG context...")
            sp_prompt = self.create_sub_processes_prompt(requirements, results["functional_processes"], results["data_groups"])
            raw_sp = self.llm.generate(sp_prompt)
            sp_data = self.extract_json_from_text(raw_sp)
            results["sub_processes"] = [
                {**sp, "process_name": k} for k, steps in sp_data.items() for sp in steps
            ]

            # 5) Metadata
            results["llm_metadata"] = {
                "provider": self.llm_cfg.provider,
                "model": self.llm_cfg.model,
                "base_url": self.llm_cfg.base_url,
                "temperature": self.llm_cfg.temperature
            }

            logger.info("Component extraction with RAG completed successfully")
            return results

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            logger.error(f"Error in component extraction: {e}")
            raise
    
    def extract_json_from_text(self, text: str) -> dict:
        """
        Robust JSON extractor:
        - prefers ```json fenced blocks
        - falls back to brace-balanced scanning
        - if multiple JSON dicts are found, merges them (right-most wins on key conflicts)
        - raises if nothing JSON-like is found
        """
        import json, re

        def _try_load(s):
            s = s.strip()
            return json.loads(s)

        # 1) Prefer ```json fenced blocks
        fence_pat = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
        fenced = fence_pat.findall(text or "")
        json_parts = []

        if fenced:
            for block in fenced:
                try:
                    json_parts.append(_try_load(block))
                except Exception:
                    # ignore bad fenced block, continue
                    pass

        # 2) If nothing (or still incomplete), do brace-balanced extraction over the whole text
        if not json_parts:
            s = (text or "").strip()
            blocks = []
            buf = []
            depth = 0
            in_string = False
            escape = False

            for ch in s:
                if in_string:
                    buf.append(ch)
                    if escape:
                        escape = False
                    elif ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                        buf.append(ch)
                    elif ch == '{':
                        depth += 1
                        buf.append(ch)
                    elif ch == '}':
                        depth -= 1
                        buf.append(ch)
                        if depth == 0:
                            blocks.append(''.join(buf))
                            buf = []
                    else:
                        # only accumulate when we're inside a JSON object
                        if depth > 0:
                            buf.append(ch)

            # Try to load each balanced block
            for b in blocks:
                try:
                    json_parts.append(_try_load(b))
                except Exception:
                    # ignore and continue
                    pass

        if not json_parts:
            raise ValueError(f"Invalid JSON response from LLM: not a JSON object\n{text}")

        # 3) Merge parts:
        #    - If there’s a single dict: return it
        #    - If multiple dicts: shallow-merge (suitable for Sub-processes blocks)
        #    - If lists show up, raise (caller expects dicts in this pipeline)
        merged = {}
        for part in json_parts:
            if isinstance(part, dict):
                merged.update(part)
            else:
                raise ValueError("Invalid JSON response from LLM: expected object, got non-object")

        return merged


    def validate_response_format(self, response_data: Dict, expected_keys: List[str]) -> bool:
        """Validate that LLM response contains expected keys"""
        for key in expected_keys:
            if key not in response_data:
                logger.warning(f"Missing expected key: {key}")
                return False
        return True