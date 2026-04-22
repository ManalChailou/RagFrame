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
    app_domain: Optional[str] = "business"  #business | realtime
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

# Initialize components
prompt_dispatcher = EnhancedPromptDispatcher(llm_provider="openai", llm_model="gpt-4o-mini", llm_base_url=None)
rule_engine = EnhancedCosmicRuleEngine()


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

            # 2) SP -> mouvements bruts
            sub_processes = components.get("sub_processes", [])
            raw_movements = rule_engine.convert_sub_processes_to_movements(sub_processes)

            # 3) Application des règles COSMIC (RU1..RU4, normalisations, etc.)
            functional_processes = components.get("functional_processes", [])
            processed_movements = rule_engine.process_movements(raw_movements, functional_processes)

            # 4) CFP par user story
            cfp_summary = rule_engine.get_cfp_summary(processed_movements)
            movements_detail = cfp_summary.get("movements_detail", [])

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
            )
            results.append(per_story)

            # 8) Agrégation globale
            for t, n in cfp_summary.get("cfp_by_type", {}).items():
                global_counts[t] = global_counts.get(t, 0) + n

        global_total = sum(global_counts.values())

        # Réponse finale 
        return MeasureResponse(
            results=results,
            global_cfp_total=global_total,
            global_cfp_by_type=global_counts,
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
    prompt_rag_status = "available" if prompt_dispatcher.rag_system else "unavailable"
    rule_rag_status = "available" if rule_engine.rag_system else "unavailable"
    
    return {
        "status": "healthy", 
        "version": "2.0.0",
        "rag_systems": {
            "prompt_dispatcher_rag": prompt_rag_status,
            "rule_engine_rag": rule_rag_status
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)