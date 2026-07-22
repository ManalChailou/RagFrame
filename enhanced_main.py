import json, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

from enhanced_prompt_dispatcher import EnhancedPromptDispatcher
from enhanced_rule_engine import EnhancedCosmicRuleEngine
from llm_router import resolve_llm_from_name
from rag_system import CosmicRAGSystem

app = FastAPI(title="Enhanced COSMIC Framework API with RAG", version="2.0.0")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequirementInput(BaseModel):
    requirements: List[str]
    format: str = "user_stories"
    enable_rag: bool = True
    app_domain: Optional[str] = "business"  #business | real_time
    # NEW: LLM selection
    llm_name: str = "gpt" # gpt | claude | gemini | deepseek | grok | minimax

class PerStoryResult(BaseModel):
    user_story: str
    cosmic_components: Dict
    data_movements: List[Dict]
    cfp_summary: Dict
    rag_insights: Optional[List[str]] = None
    validation_report: Optional[Dict] = None

class MeasureResponse(BaseModel):
    results: List[PerStoryResult]
    global_cfp_total: int
    global_cfp_by_type: Dict[str, int]
    rag_contexts: Optional[Dict] = None

def _remove_storage_data_groups(components: Dict, normalizer) -> None:
    """
    Supprime des Data Groups ceux qui sont en réalité du Storage (RAM, ROM, DB, files...)
    normalizer: fonction str -> "User" | "Storage" | "System" | "External Application" | "Clock"
    Modifie components IN-PLACE.
    """
    dgs = components.get("data_groups") or []
    cleaned = []
    for dg in dgs:
        name = (dg.get("name") or "").strip()
        if not name:
            continue
        if normalizer(name) == "Storage":
            # On ignore RAM, ROM, Database, File, etc.
            continue
        cleaned.append(dg)
    components["data_groups"] = cleaned


def _add_prediction_dg(complete_dg: Dict, dg_to_ooi: Dict, object_of_interest: str, data_group_name: str, attributes=None) -> None:
    """
    Add one ObjectOfInterest -> DataGroup -> Attributes-list entry for prediction output.

    Final prediction DG format:
    {
      "<ObjectOfInterest>": {
        "<DataGroupName>": ["<attribute_1>", "<attribute_2>"]
      }
    }
    """
    object_of_interest = str(object_of_interest or "").strip()
    data_group_name = str(data_group_name or "").strip()
    if not object_of_interest or not data_group_name:
        return

    if attributes is None:
        attributes = []
    if not isinstance(attributes, list):
        attributes = [attributes]

    clean_attrs = []
    for attr in attributes:
        attr = str(attr or "").strip()
        if attr and attr not in clean_attrs:
            clean_attrs.append(attr)

    complete_dg.setdefault(object_of_interest, {})
    complete_dg[object_of_interest].setdefault(data_group_name, [])

    # Safety if an older run/code path created {"Attributes": [...]}
    if isinstance(complete_dg[object_of_interest][data_group_name], dict):
        complete_dg[object_of_interest][data_group_name] = (
            complete_dg[object_of_interest][data_group_name].get("Attributes")
            or complete_dg[object_of_interest][data_group_name].get("attributes")
            or []
        )

    existing_attrs = complete_dg[object_of_interest][data_group_name]
    for attr in clean_attrs:
        if attr not in existing_attrs:
            existing_attrs.append(attr)

    dg_to_ooi[data_group_name.lower()] = object_of_interest


def _normalize_data_groups_for_prediction(data_groups) -> tuple[Dict, Dict]:
    """
    Convert extracted data groups to the final prediction format:
    {
      "<ObjectOfInterest>": {
        "<DataGroupName>": ["<attribute_1>", "<attribute_2>"]
      }
    }

    Supports all shapes currently produced by the dispatcher:
    1) {"Student": {"Student details": ["Student ID", "Name"]}}
    2) {"Student": {"Student details": {"Attributes": ["Student ID", "Name"]}}}
    3) [{"name": "Student", "Student details": ["Student ID", "Name"]}]
    4) [{"name": "Student", "Student details": {"Attributes": ["Student ID", "Name"]}}]
    5) [{"name": "Professor", "ObjectOfInterest": "Professor", "Attributes": [...]}]
    """
    complete_dg: Dict = {}
    dg_to_ooi: Dict = {}

    def extract_attrs(payload):
        """Return an attributes list from list, dict, scalar, or missing payload."""
        if payload is None:
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("Attributes") or payload.get("attributes") or []
        return [payload]

    def handle_mapping(mapping: Dict) -> None:
        for object_of_interest, groups in mapping.items():
            object_of_interest = str(object_of_interest or "").strip()
            if not object_of_interest:
                continue

            # Shape: {"Student": ["Student ID", "Name"]} or {"Student": {"Attributes": [...]}}
            # Treat the key itself as the data group name only for old/flat outputs.
            if isinstance(groups, list):
                _add_prediction_dg(complete_dg, dg_to_ooi, object_of_interest, object_of_interest, groups)
                continue

            if isinstance(groups, dict) and ("Attributes" in groups or "attributes" in groups):
                attrs = extract_attrs(groups)
                _add_prediction_dg(complete_dg, dg_to_ooi, object_of_interest, object_of_interest, attrs)
                continue

            # Shape: {"Student": {"Student details": [...]}}
            # or     {"Student": {"Student details": {"Attributes": [...]}}}
            if isinstance(groups, dict):
                for data_group_name, payload in groups.items():
                    if data_group_name in {"ObjectOfInterest", "Attributes", "attributes", "DataGroupName", "name"}:
                        continue
                    attrs = extract_attrs(payload)
                    _add_prediction_dg(complete_dg, dg_to_ooi, object_of_interest, data_group_name, attrs)

    # Raw LLM output can be a dict.
    if isinstance(data_groups, dict):
        handle_mapping(data_groups)
        return complete_dg, dg_to_ooi

    # components["data_groups"] is normally a list.
    for item in data_groups or []:
        if not isinstance(item, dict):
            continue

        # Explicit normalized record shape.
        explicit_ooi = item.get("ObjectOfInterest") or item.get("object_of_interest")
        explicit_dg = (
            item.get("DataGroupName")
            or item.get("data_group_name")
            or item.get("DataGroup")
            or item.get("data_group")
        )
        if explicit_ooi and explicit_dg:
            attrs = extract_attrs(item.get("Attributes") or item.get("attributes") or [])
            _add_prediction_dg(complete_dg, dg_to_ooi, explicit_ooi, explicit_dg, attrs)
            continue

        # Current extract_components() shape:
        # {"name": "Timer", "30-second signal": ["30-second signal"]}
        # {"name": "Timer", "30-second signal": {"Attributes": ["30-second signal"]}}
        if "name" in item:
            object_of_interest = str(item.get("name") or "").strip()
            nested_found = False

            for key, payload in item.items():
                if key in {"name", "ObjectOfInterest", "object_of_interest"}:
                    continue
                if key in {"Attributes", "attributes"}:
                    continue

                # IMPORTANT FIX:
                # The dispatcher may output payload as a LIST, not only {"Attributes": [...]}
                attrs = extract_attrs(payload)
                _add_prediction_dg(complete_dg, dg_to_ooi, object_of_interest, key, attrs)
                nested_found = True

            # Older flat format:
            # {"name": "Professor", "ObjectOfInterest": "Professor", "Attributes": [...]}
            if not nested_found and (item.get("Attributes") or item.get("attributes")):
                attrs = extract_attrs(item.get("Attributes") or item.get("attributes"))
                old_ooi = item.get("ObjectOfInterest") or item.get("object_of_interest") or object_of_interest
                _add_prediction_dg(complete_dg, dg_to_ooi, old_ooi, object_of_interest, attrs)

            continue

        # Fallback for a one-item raw mapping inside a list.
        handle_mapping(item)

    return complete_dg, dg_to_ooi


def _collect_moved_data_groups_for_fp(related_steps: List[Dict], fp_movements: List[Dict], dg_to_ooi: Dict) -> List[Dict]:
    """Return unique moved data groups for one FP, with their Object of Interest when available."""
    moved = []
    seen = set()

    def add(data_group_name: str, object_of_interest: str = "") -> None:
        data_group_name = str(data_group_name or "").strip()
        object_of_interest = str(object_of_interest or "").strip()
        if not data_group_name:
            return
        if not object_of_interest:
            object_of_interest = dg_to_ooi.get(data_group_name.lower(), "")
        key = (object_of_interest.lower(), data_group_name.lower())
        if key in seen:
            return
        seen.add(key)
        moved.append({
            "ObjectOfInterest": object_of_interest,
            "DataGroup": data_group_name
        })

    for sp in related_steps:
        mdg = (
            sp.get("MovedDataGroup")
            or sp.get("moved_data_group")
            or sp.get("DataGroup")
            or sp.get("data_group")
        )
        ooi = sp.get("ObjectOfInterest") or sp.get("object_of_interest") or ""
        add(mdg, ooi)

    # Fallback after rule-engine processing.
    if not moved:
        for m in fp_movements:
            add(m.get("data_group"), m.get("object_of_interest"))

    return moved


def save_prediction_jsonl(components: Dict, movements: List[Dict], path: str = "config/prediction.jsonl", overwrite: bool = False):
    """
    Append predicted components in JSONL format, one line per Functional Process (FP).

    Output:
    - DG contains the complete extracted data groups for the whole user story.
    - Only DG is written; moved data groups are not exported as a separate field.
    """
    try:
        dirpath = os.path.dirname(path) or "."
        os.makedirs(dirpath, exist_ok=True)
        mode = "w" if overwrite else "a"

        fp_data = components.get("functional_processes", []) or []
        data_groups = components.get("data_groups", []) or []
        sub_processes = components.get("sub_processes", []) or []

        complete_dg, _ = _normalize_data_groups_for_prediction(data_groups)

        with open(path, mode, encoding="utf-8") as f:
            for fp in fp_data:
                fp_name = fp.get("name")

                related_steps = [sp for sp in sub_processes if sp.get("process_name") == fp_name]
                fp_movements = [m for m in movements if m.get("process") == fp_name]
                movement_counts = {
                    "Entry": sum(1 for m in fp_movements if m.get("type") == "Entry"),
                    "Read":  sum(1 for m in fp_movements if m.get("type") == "Read"),
                    "Write": sum(1 for m in fp_movements if m.get("type") == "Write"),
                    "Exit":  sum(1 for m in fp_movements if m.get("type") == "Exit"),
                }

                record = {
                    "FP": fp_name,
                    "TEn": fp.get("TriggeringEvents"),
                    "FU": components.get("FunctionalUsers", []),

                    # Complete extracted Data Groups from the DG extraction step.
                    "DG": complete_dg,

                    "SP": [sp.get("StepName") for sp in related_steps],
                    "Action": [sp.get("ActionVerb") for sp in related_steps],
                    "Source": [sp.get("Source") for sp in related_steps],
                    "Destination": [sp.get("Destination") for sp in related_steps],
                    "Movements": movement_counts,
                    "RAG_Enhanced": components.get("rag_metadata", {}).get("knowledge_chunks_used", False),
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Prediction appended to {path}")

    except Exception as e:
        logger.error(f"Failed to save prediction.jsonl: {e}")

# Initialize components
prompt_dispatcher = EnhancedPromptDispatcher(llm_provider="openai", llm_model="gpt-4o-mini", llm_base_url=None)
rule_engine = EnhancedCosmicRuleEngine()

# Initialize standalone RAG system for query endpoints
try:
    standalone_rag = CosmicRAGSystem()
    logger.info("Standalone RAG system initialized for query endpoints")
except Exception as e:
    logger.error(f"Failed to initialize standalone RAG system: {e}")
    standalone_rag = None

@app.post("/measure", response_model=MeasureResponse)
async def measure_cosmic(input_data: RequirementInput):
    """Main endpoint for COSMIC measurement per FUR"""
    overwrite_flag = True
    original_rag_system = prompt_dispatcher.rag_system

    try:
        logger.info(f"[measure] Processing {len(input_data.requirements)} user stories...")

        results: List[PerStoryResult] = []
        global_counts: Dict[str, int] = {"Entry": 0, "Exit": 0, "Read": 0, "Write": 0}
        
        prompt_dispatcher.set_app_domain(input_data.app_domain)

        llm_cfg = resolve_llm_from_name(input_data.llm_name)

        prompt_dispatcher.set_llm(
            provider=llm_cfg.provider,
            model=llm_cfg.model,
            base_url=llm_cfg.base_url,
            temperature=llm_cfg.temperature
        )
        
        if not input_data.enable_rag:
            prompt_dispatcher.rag_system = None

        for us in input_data.requirements:
            # 1) Extraction de composants pour UNE user story
            components = prompt_dispatcher.extract_components([us])
            _remove_storage_data_groups(components, rule_engine.normalize_entity)

            # 2) Rewrite StepName labels using a constrained LLM call, then convert SP -> raw movements
            # The rewrite call changes ONLY StepName. It preserves ActionVerb, DataGroup,
            # Source, Destination and process_name, so CFP counting remains rule-based.
            sub_processes = components.get("sub_processes", [])
            sub_processes = prompt_dispatcher.rewrite_sub_process_step_names(
                requirements=[us],
                sub_processes=sub_processes,
                functional_processes=components.get("functional_processes", []),
                data_groups=components.get("data_groups", []),
            )
            components["sub_processes"] = sub_processes
            raw_movements = rule_engine.convert_sub_processes_to_movements(sub_processes)

            # 3) Application des règles COSMIC (RU1..RU4, normalisations, etc.)
            functional_processes = components.get("functional_processes", [])
            processed_movements = rule_engine.process_movements(raw_movements, functional_processes)

            # 4) CFP par user story
            cfp_summary = rule_engine.get_cfp_summary(processed_movements)
            movements_detail = cfp_summary.get("movements_detail", [])

            # Ajout : sauvegarder pour chaque US (append)
            save_prediction_jsonl(components, movements_detail, overwrite=overwrite_flag)
            overwrite_flag = False

            # 5) Validation (optionnelle selon le niveau demandé)
            validation_report = None
            if getattr(input_data, "validation_level", None) in ["standard", "comprehensive"]:
                validation_report = rule_engine.get_rag_enhanced_validation(processed_movements)

            # 6) Insights / rapport (optionnels)
            measurement_report = rule_engine.generate_measurement_report(processed_movements, functional_processes)
            rag_insights = measurement_report.get("rag_insights", [])

            # 7) Accumulation du résultat par user story
            per_story = PerStoryResult(
                user_story=us,
                cosmic_components=components,
                data_movements=cfp_summary.get("movements_detail", []),
                cfp_summary={
                    "cfp_by_type": cfp_summary.get("cfp_by_type", {}),
                    "cfp_by_process": cfp_summary.get("cfp_by_process", {}),
                    "total_cfp": cfp_summary.get("total_cfp", 0),
                    "quality_metrics": cfp_summary.get("quality_metrics", {}),
                },
                rag_insights=rag_insights if getattr(input_data, "enable_rag", False) else None,
                validation_report=validation_report,
            )
            results.append(per_story)

            # 8) Agrégation globale
            for t, n in cfp_summary.get("cfp_by_type", {}).items():
                global_counts[t] = global_counts.get(t, 0) + n

        global_total = sum(global_counts.values())

        single_rag_ctx = None
        try:
            # Si le dispatcher expose un getter, on l’utilise
            single_rag_ctx = getattr(prompt_dispatcher, "get_last_rag_contexts", lambda: None)()
        except Exception:
            single_rag_ctx = None

        # Si non disponible via getter, tenter de le lire dans la 1re US (cosmic_components)
        if not single_rag_ctx and results and isinstance(results[0].cosmic_components, dict):
            single_rag_ctx = results[0].cosmic_components.get("rag_contexts")

        # Réponse finale (RAG context une seule fois)
        return MeasureResponse(
            results=results,
            global_cfp_total=global_total,
            global_cfp_by_type=global_counts,
            rag_contexts=single_rag_ctx
        )

    except Exception as e:
        logger.error(f"[measure] Error: {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # ✅ Always restore for next requests
        prompt_dispatcher.rag_system = original_rag_system

@app.get("/health")
async def health_check():
    """Enhanced health check with RAG system status"""
    rag_status = "available" if standalone_rag else "unavailable"
    prompt_rag_status = "available" if prompt_dispatcher.rag_system else "unavailable"
    rule_rag_status = "available" if rule_engine.rag_system else "unavailable"
    
    return {
        "status": "healthy", 
        "version": "2.0.0",
        "rag_systems": {
            "standalone_rag": rag_status,
            "prompt_dispatcher_rag": prompt_rag_status,
            "rule_engine_rag": rule_rag_status
        },
        "knowledge_base_loaded": len(standalone_rag.knowledge_chunks) if standalone_rag else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)