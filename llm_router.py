import os
import json
import logging
from dataclasses import dataclass
import time
from typing import Optional, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)

APIFREE_DEFAULT_BASE_URL = os.getenv("APIFREE_BASE_URL") or "https://api.apifree.ai/v1"

APIFREE_MODEL_PRESETS = {
    "gpt":      {"provider": "apifree", "model": "openai/gpt-4o-mini",            "base_url": APIFREE_DEFAULT_BASE_URL},
    #"claude":   {"provider": "apifree", "model": "anthropic/claude-sonnet-4.5","base_url": APIFREE_DEFAULT_BASE_URL},
    #"gemini":   {"provider": "apifree", "model": "google/gemini-2.5-pro/on-demand",      "base_url": APIFREE_DEFAULT_BASE_URL},
    "deepseek": {"provider": "apifree", "model": "deepseek-ai/deepseek-v3.2",    "base_url": APIFREE_DEFAULT_BASE_URL},
    "grok":     {"provider": "apifree", "model": "xai/grok-4-fast",                    "base_url": APIFREE_DEFAULT_BASE_URL},
    "minimax":  {"provider": "apifree", "model": "minimax/minimax-m2.5",               "base_url": APIFREE_DEFAULT_BASE_URL},
    "qwen":  {"provider": "apifree", "model": "qwen/qwen3-coder-480b-a35b",               "base_url": APIFREE_DEFAULT_BASE_URL},
    "openai":   {"provider": "openai", "model": "gpt-4", "base_url": None},
}

# --------- Shared config ---------
@dataclass
class LLMConfig:
    provider: str = "openai"          # openai | anthropic | gemini ...
    model: str = "gpt-4"
    temperature: float = 0.2
    base_url: Optional[str] = None    # for openai_compat
    extra: Optional[Dict[str, Any]] = None

def resolve_llm_from_name(name: str) -> LLMConfig:
    key = (name or "").strip().lower()
    if key not in APIFREE_MODEL_PRESETS:
        raise ValueError(f"Unknown llm_name='{name}'. Allowed: {list(APIFREE_MODEL_PRESETS.keys())}")

    preset = APIFREE_MODEL_PRESETS[key]
    return LLMConfig(
        provider=preset["provider"],
        model=preset["model"],
        base_url=preset["base_url"],
        temperature=0.2,
    )

class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class BuildLLM(BaseLLM):

    def __init__(self, cfg: LLMConfig):
        
        # Choose API key based on provider
        if cfg.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY for OpenAI provider")
        else:
            api_key = os.getenv("APIFREE_KEY")
            if not api_key:
                raise RuntimeError("Missing APIFREE_KEY for APIFree-backed provider")

        # allow local servers without real keys
        if not api_key:
            if cfg.base_url and ("localhost" in cfg.base_url or "127.0.0.1" in cfg.base_url):
                api_key = "local-no-key"
            else:
                raise RuntimeError("Missing API key")
            
        self.client = OpenAI(api_key=api_key, base_url=cfg.base_url)
        self.model = cfg.model
        self.temperature = cfg.temperature

    def generate(self, prompt: str) -> str:
        last_error = None

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
            except Exception as e:
                last_error = e
                logger.exception("LLM API call failed for model=%s attempt=%s", self.model, attempt + 1)
                time.sleep(1.5 * (attempt + 1))
                continue

            if resp is None:
                last_error = RuntimeError(f"LLM returned None for model={self.model}")
                time.sleep(1.5 * (attempt + 1))
                continue

            choices = getattr(resp, "choices", None)
            if not choices:
                try:
                    raw = resp.model_dump()
                except Exception:
                    raw = str(resp)

                if isinstance(raw, dict) and raw.get("error"):
                    err = raw["error"]
                    msg = (err.get("message") or "").lower()

                    if "request failed" in msg and attempt < 2:
                        logger.warning(
                            "Transient provider error for model=%s attempt=%s: %s",
                            self.model, attempt + 1, raw
                        )
                        time.sleep(2 * (attempt + 1))
                        continue

                    raise RuntimeError(
                        f"Provider error for model={self.model}: "
                        f"{err.get('type')} - {err.get('message')}"
                    )

                raise RuntimeError(
                    f"LLM returned no choices for model={self.model}. Raw response: {raw}"
                )

            first = choices[0]
            message = getattr(first, "message", None)
            if message is None:
                try:
                    raw = resp.model_dump()
                except Exception:
                    raw = str(resp)
                raise RuntimeError(
                    f"LLM returned first choice without message for model={self.model}. Raw response: {raw}"
                )

            finish_reason = getattr(first, "finish_reason", None)
            content = getattr(message, "content", None)

            if isinstance(content, str):
                text = content.strip()
                if text:
                    return text

            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and item.get("text"):
                            parts.append(item["text"])
                    else:
                        item_type = getattr(item, "type", None)
                        item_text = getattr(item, "text", None)
                        if item_type == "text" and item_text:
                            parts.append(item_text)

                text = "\n".join(parts).strip()
                if text:
                    return text

            refusal = getattr(message, "refusal", None)
            if refusal:
                raise RuntimeError(f"LLM refusal for model={self.model}: {refusal}")

            try:
                raw = resp.model_dump()
            except Exception:
                raw = str(resp)

            if finish_reason == "content_filter" or "SAFETY_CHECK" in str(raw):
                raise RuntimeError(
                    f"Content filtered by provider for model={self.model}. Raw response: {raw}"
                )

            raise RuntimeError(
                f"Could not extract text content from model={self.model}. Raw response: {raw}"
            )

        raise RuntimeError(f"LLM API failed after retries for model={self.model}: {last_error}")