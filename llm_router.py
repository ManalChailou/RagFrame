import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)

APIFREE_DEFAULT_BASE_URL = os.getenv("APIFREE_BASE_URL") or "https://api.apifree.ai/v1"

APIFREE_MODEL_PRESETS = {
    "gpt":      {"provider": "apifree", "model": "openai/gpt-4o-mini",            "base_url": APIFREE_DEFAULT_BASE_URL},
    "claude":   {"provider": "apifree", "model": "anthropic/claude-sonnet-4.5","base_url": APIFREE_DEFAULT_BASE_URL},
    "gemini":   {"provider": "apifree", "model": "google/gemini-1.5-pro",      "base_url": APIFREE_DEFAULT_BASE_URL},
    "deepseek": {"provider": "apifree", "model": "deepseek/deepseek-chat",    "base_url": APIFREE_DEFAULT_BASE_URL},
    "grok":     {"provider": "apifree", "model": "xai/grok-2",                    "base_url": APIFREE_DEFAULT_BASE_URL},
    "minimax":  {"provider": "apifree", "model": "minimax/abab",               "base_url": APIFREE_DEFAULT_BASE_URL},
    "openai":   {"provider": "openai", "model": "gpt-4", "base_url": None},
}

# --------- Shared config ---------
@dataclass
class LLMConfig:
    provider: str = "openai"          # openai | anthropic | gemini
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
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    


# # --------- Anthropic Claude (REST) ---------
# class AnthropicLLM(BaseLLM):
#     """
#     Uses Anthropic Messages API via REST.
#     Needs: ANTHROPIC_API_KEY
#     """
#     def __init__(self, cfg: LLMConfig):
#         import requests
#         self.requests = requests
#         self.api_key = os.getenv("ANTHROPIC_API_KEY")
#         if not self.api_key:
#             raise RuntimeError("Missing ANTHROPIC_API_KEY")
#         self.model = cfg.model
#         self.temperature = cfg.temperature
# 
#     def generate(self, prompt: str) -> str:
#         url = "https://api.anthropic.com/v1/messages"
#         headers = {
#             "x-api-key": self.api_key,
#             "anthropic-version": "2023-06-01",
#             "content-type": "application/json",
#         }
#         payload = {
#             "model": self.model,
#             "max_tokens": 4096,
#             "temperature": self.temperature,
#             "messages": [{"role": "user", "content": prompt}],
#         }
#         r = self.requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
#         r.raise_for_status()
#         data = r.json()
#         # content is a list of blocks
#         blocks = data.get("content", [])
#         text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
#         return (text or "").strip()


# --------- Google Gemini (REST) ---------
# class GeminiLLM(BaseLLM):
#     """
#     Uses Gemini generateContent via REST.
#     Needs: GEMINI_API_KEY
#     """
#     def __init__(self, cfg: LLMConfig):
#         import requests
#         self.requests = requests
#         self.api_key = os.getenv("GEMINI_API_KEY")
#         if not self.api_key:
#             raise RuntimeError("Missing GEMINI_API_KEY")
#         self.model = cfg.model
#         self.temperature = cfg.temperature

#     def generate(self, prompt: str) -> str:
#         # v1beta endpoint
#         url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
#         payload = {
#             "generationConfig": {"temperature": self.temperature, "maxOutputTokens": 4096},
#             "contents": [{"role": "user", "parts": [{"text": prompt}]}],
#         }
#         r = self.requests.post(url, json=payload, timeout=120)
#         r.raise_for_status()
#         data = r.json()
#         candidates = data.get("candidates", [])
#         if not candidates:
#             return ""
#         parts = candidates[0].get("content", {}).get("parts", [])
#         text = "".join(p.get("text", "") for p in parts)
#         return (text or "").strip()

