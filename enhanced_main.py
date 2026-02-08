import json, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

from enhanced_prompt_dispatcher import EnhancedPromptDispatcher
from enhanced_rule_engine import EnhancedCosmicRuleEngine
from rag_system import CosmicRAGSystem

app = FastAPI(title="Enhanced COSMIC Framework API with RAG", version="2.0.0")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequirementInput(BaseModel):
    requirements: List[str]
    format: str = "user_stories"
    enable_rag: bool = True
    validation_level: str = "standard"
    rag_contexts: Optional[Dict] = None
    app_domain: Optional[str] = ""
    # NEW: LLM selection
    llm_provider: str = "openai"          # openai | anthropic | gemini | openai_compat
    llm_model: str = "gpt-4"
    llm_base_url: Optional[str] = None  # for openai_compat endpoints
    llm_temperature: float = 0.2

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
    dgs = components.get("data_groups", [])
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

def save_prediction_jsonl(components: Dict, movements: List[Dict], path: str = "config/prediction.jsonl", overwrite: bool = False):
    """Append predicted components in JSONL format, one line per Functional Process (FP)."""
    try:
        # Assurer l'existence du dossier
        dirpath = os.path.dirname(path) or "."
        os.makedirs(dirpath, exist_ok=True)
        mode = "w" if overwrite else "a"

        fp_data = components.get("functional_processes", []) or []
        data_groups = components.get("data_groups", []) or []
        sub_processes = components.get("sub_processes", []) or []

        with open(path, mode, encoding="utf-8") as f:
            for fp in fp_data:
                fp_name = fp.get("name")

                # Sous-processus de CE FP
                related_steps = [sp for sp in sub_processes if sp.get("process_name") == fp_name]

                # 1) DG = liste des MovedDataGroup (ordre d’apparition, sans doublons)
                moved_dg_list = []
                for sp in related_steps:
                    mdg = (
                        sp.get("MovedDataGroup")
                        or sp.get("moved_data_group")
                        or sp.get("DataGroup")
                        or sp.get("data_group")
                    )
                    if mdg:
                        mdg = str(mdg).strip()
                        if mdg and mdg not in moved_dg_list:
                            moved_dg_list.append(mdg)

                # 2) fallback si pas de SP : unique data_group depuis les mouvements de CE FP
                if not moved_dg_list:
                    fp_movs = [m for m in movements if m.get("process") == fp_name]
                    mdg_from_movs = []
                    for m in fp_movs:
                        dg = (m.get("data_group") or "").strip()
                        if dg and dg not in mdg_from_movs:
                            mdg_from_movs.append(dg)
                    moved_dg_list = mdg_from_movs

                # 3) fallback final : anciens data_groups extraits (noms)
                if not moved_dg_list:
                    moved_dg_list = [dg.get("name") for dg in data_groups if dg.get("name")]

                # Comptes de mouvements de CE FP
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
                    "FU": fp.get("FunctionalUser"),
                    "DG": moved_dg_list,
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
prompt_dispatcher = EnhancedPromptDispatcher(llm_provider="openai", llm_model="gpt-4o-mini", llm_base_url=None, temperature=0.2)
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

        prompt_dispatcher.set_llm(
            provider=input_data.llm_provider,
            model=input_data.llm_model,
            base_url=input_data.llm_base_url,
            temperature=input_data.llm_temperature
        )
        
        if not input_data.enable_rag:
            prompt_dispatcher.rag_system = None

        for us in input_data.requirements:
            # 1) Extraction de composants pour UNE user story
            components = prompt_dispatcher.extract_components([us])
            _remove_storage_data_groups(components, rule_engine.normalize_entity)

            # 2) SP -> mouvements bruts
            sub_processes = components.get("sub_processes", [])
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