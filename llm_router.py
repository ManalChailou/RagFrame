# llm_router.py
import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# --------- Shared config ---------
@dataclass
class LLMConfig:
    provider: str = "openai"          # openai | openai_compat | anthropic | gemini
    model: str = "gpt-4"
    temperature: float = 0.2
    base_url: Optional[str] = None    # for openai_compat
    extra: Optional[Dict[str, Any]] = None


class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


# --------- OpenAI / OpenAI-compatible ---------
class OpenAICompatibleLLM(BaseLLM):
    """
    Works for:
      - OpenAI official (base_url=None)
      - Any OpenAI-compatible provider if you set base_url + api_key in env
        Examples: Mistral, DeepSeek, Qwen (depending on the endpoint you use)
    """
    def __init__(self, cfg: LLMConfig):
        from openai import OpenAI  # already used in your code
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("MISTRAL_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("QWEN_API_KEY")
        )
        
        # allow local servers without real keys
        if not api_key:
            if cfg.base_url and ("localhost" in cfg.base_url or "127.0.0.1" in cfg.base_url):
                api_key = "local-no-key"
            else:
                raise RuntimeError("Missing API key for OpenAI/OpenAI-compatible provider") 
            
        # base_url optional: if provided, routes to openai-compatible endpoint
        if cfg.base_url:
            self.client = OpenAI(api_key=api_key, base_url=cfg.base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        self.model = cfg.model
        self.temperature = cfg.temperature

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return (resp.choices[0].message.content or "").strip()


# --------- Anthropic Claude (REST) ---------
class AnthropicLLM(BaseLLM):
    """
    Uses Anthropic Messages API via REST.
    Needs: ANTHROPIC_API_KEY
    """
    def __init__(self, cfg: LLMConfig):
        import requests
        self.requests = requests
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")
        self.model = cfg.model
        self.temperature = cfg.temperature

    def generate(self, prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = self.requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        data = r.json()
        # content is a list of blocks
        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return (text or "").strip()


# --------- Google Gemini (REST) ---------
class GeminiLLM(BaseLLM):
    """
    Uses Gemini generateContent via REST.
    Needs: GEMINI_API_KEY
    """
    def __init__(self, cfg: LLMConfig):
        import requests
        self.requests = requests
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        self.model = cfg.model
        self.temperature = cfg.temperature

    def generate(self, prompt: str) -> str:
        # v1beta endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": 4096},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }
        r = self.requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        return (text or "").strip()


def build_llm(cfg: LLMConfig) -> BaseLLM:
    provider = (cfg.provider or "").lower().strip()
    if provider in ["openai", "openai_compat", "openai-compatible", "compatible"]:
        return OpenAICompatibleLLM(cfg)
    if provider in ["anthropic", "claude"]:
        return AnthropicLLM(cfg)
    if provider in ["gemini", "google"]:
        return GeminiLLM(cfg)

    raise ValueError(f"Unknown LLM provider: {cfg.provider}")
