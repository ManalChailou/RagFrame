import json, os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, List, Dict, Optional, Literal
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhanced_prompt_dispatcher_MFS import EnhancedPromptDispatcher
from enhanced_rule_engine import EnhancedCosmicRuleEngine
from llm_router import resolve_llm_from_name
from managed_grounding_backend import ManagedGroundingBackend
from rag_system import CosmicRAGSystem

app = FastAPI(title="Enhanced COSMIC Framework API with RAG", version="2.0.0")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequirementInput(BaseModel):
    requirements: List[str]
    format: str = "user_stories"
    retrieval_backend: Literal["managed_file_search", "local_rag", "none"] = "managed_file_search"
    app_domain: Optional[Literal["business", "real_time"]] = "business"
    llm_name: str = "openai"
    validation_level: Literal["none", "standard", "comprehensive"] = "none"

    class Config:
        schema_extra = {
            "example": {
                "requirements": [
                    "When a Registrar selects Add Professor, the Registrar enters Professor data and C-Reg creates the record if valid."
                ],
                "format": "user_stories",
                "retrieval_backend": "managed_file_search",
                "app_domain": "business",
                "llm_name": "openai",
                "validation_level": "none",
            }
        }


class PerStoryResult(BaseModel):
    user_story: str
    cosmic_components: Dict[str, Any]
    data_movements: List[Dict[str, Any]]
    cfp_summary: Dict[str, Any]
    rag_insights: Optional[List[str]] = None
    validation_report: Optional[Dict[str, Any]] = None


class MeasureResponse(BaseModel):
    results: List[PerStoryResult]
    global_cfp_total: int
    global_cfp_by_type: Dict[str, int]
    rag_contexts: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "user_story": "When a Registrar selects Add Professor...",
                        "cosmic_components": {
                            "FunctionalUsers": ["Registrar"],
                            "functional_processes": [],
                            "data_groups": [],
                            "sub_processes": [],
                        },
                        "data_movements": [],
                        "cfp_summary": {
                            "cfp_by_type": {"Entry": 0, "Exit": 0, "Read": 0, "Write": 0},
                            "cfp_by_process": {},
                            "total_cfp": 0,
                            "quality_metrics": {},
                        },
                        "rag_insights": [],
                        "validation_report": None,
                    }
                ],
                "global_cfp_total": 0,
                "global_cfp_by_type": {"Entry": 0, "Exit": 0, "Read": 0, "Write": 0},
                "rag_contexts": {
                    "story_1": {
                        "functional_users": [],
                        "functional_processes": [],
                        "data_groups": [],
                        "sub_processes": [],
                        "domain": "business",
                    }
                },
            }
        }

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
prompt_dispatcher = EnhancedPromptDispatcher(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    llm_base_url=None,
    retrieval_backend="none",
)
rule_engine = EnhancedCosmicRuleEngine()

# Local RAG is initialized lazily only when explicitly requested.
standalone_rag = None

@app.post("/measure", response_model=MeasureResponse)
async def measure_cosmic(input_data: RequirementInput):
    """Main endpoint for COSMIC measurement per FUR"""
    overwrite_flag = True
    original_rag_system = prompt_dispatcher.rag_system
    original_managed_grounding = prompt_dispatcher.managed_grounding
    original_backend = prompt_dispatcher.retrieval_backend

    try:
        logger.info(f"[measure] Processing {len(input_data.requirements)} user stories...")

        results: List[PerStoryResult] = []
        global_counts: Dict[str, int] = {"Entry": 0, "Exit": 0, "Read": 0, "Write": 0}
        all_rag_contexts: Dict[str, Any] = {}
        
        prompt_dispatcher.set_app_domain(input_data.app_domain)

        llm_cfg = resolve_llm_from_name(input_data.llm_name)

        prompt_dispatcher.set_llm(
            provider=llm_cfg.provider,
            model=llm_cfg.model,
            base_url=llm_cfg.base_url,
            temperature=llm_cfg.temperature
        )
        
        backend = input_data.retrieval_backend
        prompt_dispatcher.retrieval_backend = backend

        if backend == "none":
            prompt_dispatcher.rag_system = None
            prompt_dispatcher.managed_grounding = None

        elif backend == "local_rag":
            prompt_dispatcher.managed_grounding = None
            if prompt_dispatcher.rag_system is None:
                prompt_dispatcher.rag_system = CosmicRAGSystem()

        elif backend == "managed_file_search":
            prompt_dispatcher.rag_system = None
            prompt_dispatcher.managed_grounding = ManagedGroundingBackend(
                temperature=llm_cfg.temperature,
                max_num_results=5,
            )

        for story_index, us in enumerate(input_data.requirements, start=1):
            # 1) Extraction de composants pour UNE user story
            components = prompt_dispatcher.extract_components([us])
            all_rag_contexts[f"story_{story_index}"] = prompt_dispatcher.get_last_rag_contexts()
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
            if input_data.validation_level in ["standard", "comprehensive"]:
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
                rag_insights=rag_insights if backend != "none" else None,
                validation_report=validation_report,
            )
            results.append(per_story)

            # 8) Agrégation globale
            for t, n in cfp_summary.get("cfp_by_type", {}).items():
                global_counts[t] = global_counts.get(t, 0) + n

        global_total = sum(global_counts.values())

        return MeasureResponse(
            results=results,
            global_cfp_total=global_total,
            global_cfp_by_type=global_counts,
            rag_contexts=all_rag_contexts if backend != "none" else None,
        )

    except Exception as e:
        logger.error(f"[measure] Error: {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # ✅ Always restore for next requests
        prompt_dispatcher.rag_system = original_rag_system
        prompt_dispatcher.managed_grounding = original_managed_grounding
        prompt_dispatcher.retrieval_backend = original_backend

@app.get("/health")
async def health_check():
    """Health check without eagerly loading the local embedding model."""
    managed_configured = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_VECTOR_STORE_ID"))
    return {
        "status": "healthy",
        "version": "2.1.0",
        "default_retrieval_backend": "managed_file_search",
        "managed_file_search": "configured" if managed_configured else "not_configured",
        "local_rag": "lazy_initialization",
        "active_dispatcher_backend": prompt_dispatcher.retrieval_backend,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
