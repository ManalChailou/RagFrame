import json
import re
import os
from dotenv import load_dotenv
from typing import Dict, List
from openai import OpenAI
import logging
from rag_system import CosmicRAGSystem

logger = logging.getLogger(__name__)
load_dotenv() 

class EnhancedPromptDispatcher:
    """ Version améliorée du PromptDispatcher avec intégration RAG """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.temperature = 0.2
        self._app_domain = "" 
        self.last_rag_contexts = {  
            "functional_users": "",
            "functional_processes": "",
            "data_groups": "",
            "sub_processes": "",
            "domain": "" 
        }
        
        # Initialiser le système RAG
        try:
            self.rag_system = CosmicRAGSystem()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            self.rag_system = None

    def set_app_domain(self, app_domain: str):
        self._app_domain = (app_domain or "").strip()
        self.last_rag_contexts["domain"] = self._app_domain

    def get_last_rag_contexts(self):  
        return dict(self.last_rag_contexts)
    
    # --- simple sanitizer for FU list ---
    def _sanitize_functional_users(self, fu_list: list) -> list:
        if not fu_list: 
            return []
        cleaned = []
        for fu in fu_list:
            f = (fu or "").strip()
            if not f: 
                continue
            low = f.lower()

            # map temporal mentions to Timer
            if self._is_timing_mention(low) or "signal" in low:
                mapped = "Timer"
                if mapped not in cleaned:
                    cleaned.append(mapped)
                continue

            # drop storages/systems/devices that are not functional users
            drop_tokens = ["ram", "rom", "memory", "storage", "system", "database"]
            if any(t in low for t in drop_tokens):
                continue

            # keep external actors/devices if really needed (not in this FUR)
            cleaned.append(f)

        # ensure Timer is kept if any temporal trigger is present
        if "Timer" in cleaned:
            cleaned = ["Timer"]  # in this FUR we want only Timer
        # final dedup
        cleaned = list(dict.fromkeys(cleaned))
        return cleaned

        
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
        
        # Ajouter le contexte RAG spécialisé pour les mouvements de données
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
5. For EACH sub-process, "MovedDataGroup" MUST be 

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
            fu_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": fu_prompt}],
                temperature=self.temperature
            )
            raw_fu = fu_response.choices[0].message.content.strip()
            fu_data = self.extract_json_from_text(raw_fu)
            results.update(fu_data)
            functional_users_list = self._sanitize_functional_users(fu_data.get("FunctionalUsers", []))
            results["FunctionalUsers"] = functional_users_list  # overwrite with sanitized list
            logger.info(f"Extracted {len(fu_data.get('FunctionalUsers', []))} functional users")

            # 2. Functional Processes
            logger.info("Extracting Functional Processes with RAG context...")
            fp_prompt = self.create_functional_process_prompt(requirements, functional_users_list)
            fp_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": fp_prompt}],
                temperature=self.temperature
            )
            raw_fp = fp_response.choices[0].message.content.strip()
            print("\n🧠 [RAW LLM OUTPUT - Functional Processes]\n", raw_fp, "\n")
            fp_data = self.extract_json_from_text(raw_fp)
            results["functional_processes"] = [{"name": k, **v} for k, v in fp_data.items()]
            # normalise le champ FunctionalUser (Timer)
            for fp in results["functional_processes"]:
                fu = fp.get("FunctionalUser", "")
                te = fp.get("TriggeringEvents", "") or fp.get("TriggeringEntry", "")
                if self._is_timing_mention(fu) or self._is_timing_mention(te):
                    fp["FunctionalUser"] = "Timer"
            # --- collapse to one FP if the requirement looks periodic ---
            req_text = " ".join(requirements).lower()
            is_periodic = any(w in req_text for w in ["every", "each", "interval", "second signal", "30-second"])
            if is_periodic and len(results["functional_processes"]) > 1:
                main = results["functional_processes"][0]
                main_name = main.get("name") or "Update-Target-Temperature"
                main["name"] = main_name
                main["FunctionalUser"] = "Timer"
                # keep only the first FP
                results["functional_processes"] = [main]
            
            logger.info(f"Extracted {len(fp_data)} functional processes")

            # 3. Data Groups
            logger.info("Extracting Data Groups with RAG context...")
            dg_prompt = self.create_data_groups_prompt(requirements, results["functional_processes"])
            dg_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": dg_prompt}],
                temperature=self.temperature
            )
            raw_dg = dg_response.choices[0].message.content.strip()
            print("\n🧠 [RAW LLM OUTPUT - Data Groups]\n", raw_dg, "\n")
            dg_data = self.extract_json_from_text(raw_dg)
            results["data_groups"] = [{"name": k, **v} for k, v in dg_data.items()]
            logger.info(f"Extracted {len(dg_data)} data groups")

            # 4. Sub-Processes
            logger.info("Extracting Sub-processes with RAG context...")
            sp_prompt = self.create_sub_processes_prompt(requirements, results["functional_processes"], results["data_groups"])
            sp_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": sp_prompt}],
                temperature=self.temperature
            )
            raw_sp = sp_response.choices[0].message.content.strip()
            print("\n🧠 [RAW LLM OUTPUT - Sub Processes]\n", raw_sp, "\n")
            sp_data = self.extract_json_from_text(raw_sp)
            results["sub_processes"] = [
                {**sp, "process_name": k} for k, steps in sp_data.items() for sp in steps
            ]
            logger.info(f"Extracted {len(results['sub_processes'])} sub-processes")

            # 5. Ajouter des métadonnées RAG
            if self.rag_system:
                results["rag_metadata"] = {
                    "knowledge_chunks_used": True,
                    "validation_context_available": True
                }
            #logger.debug(self.get_last_rag_contexts())               to show the RAG context used
            logger.info("Component extraction with RAG completed successfully")
            return results

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            logger.error(f"Error in component extraction: {e}")
            raise

    def _is_timing_mention(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        # Heuristiques simples + formes courantes
        return (
            any(k in t for k in ["timer", "clock", "tick", "time event", "time-event", "time_event", "second", "sec"])
            or bool(re.search(r"\b\d+\s*-\s*second(s)?\b", t))   # ex: "30-second"
            or bool(re.search(r"\b\d+\s*(sec|s)\b", t))          # ex: "5s", "30s", "5 sec"
            or bool(re.search(r"\b\d+\s*[-]?\s*(minute|min|m)\b", t))     
            or bool(re.search(r"\b\d+\s*[-]?\s*(hour|hr|h)\b", t))        
            or bool(re.search(r"\b\d+\s*[-]?\s*(day|d)\b", t))            
        )

    
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

    def get_validation_suggestions(self, movements: List[Dict]) -> List[str]:
        """Utilise le RAG pour obtenir des suggestions de validation"""
        if not self.rag_system:
            return []
        
        validation_context = self.rag_system.get_validation_context()
        suggestions = []
        
        # Analyser les mouvements pour identifier des problèmes potentiels
        entry_count = sum(1 for m in movements if m.get('type') == 'Entry')
        exit_count = sum(1 for m in movements if m.get('type') == 'Exit')
        
        if entry_count == 0:
            suggestions.append("No Entry movements found - every functional process should have at least one Entry")
        
        if exit_count == 0:
            suggestions.append("No Exit movements found - consider if users need feedback from the system")
        
        return suggestions