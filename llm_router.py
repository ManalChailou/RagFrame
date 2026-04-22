import os
import json
import logging
from dataclasses import dataclass
import time
from typing import Optional, Dict, Any
from openai import OpenAI
import anthropic
from google import genai

logger = logging.getLogger(__name__)

APIFREE_DEFAULT_BASE_URL = os.getenv("APIFREE_BASE_URL") or "https://api.apifree.ai/v1"

APIFREE_MODEL_PRESETS = {
    "gpt":      {"provider": "apifree", "model": "openai/gpt-4o-mini",            "base_url": APIFREE_DEFAULT_BASE_URL},
    "deepseek": {"provider": "apifree", "model": "deepseek-ai/deepseek-v3.2",    "base_url": APIFREE_DEFAULT_BASE_URL},
    "grok":     {"provider": "apifree", "model": "xai/grok-4-fast",                    "base_url": APIFREE_DEFAULT_BASE_URL},
    "minimax":  {"provider": "apifree", "model": "minimax/minimax-m2.5",               "base_url": APIFREE_DEFAULT_BASE_URL},
    "qwen":     {"provider": "apifree", "model": "qwen/qwen3-coder-480b-a35b",               "base_url": APIFREE_DEFAULT_BASE_URL},
    "openai":   {"provider": "openai", "model": "gpt-4", "base_url": None},
    "claude":   {"provider": "anthropic", "model": "claude-sonnet-4-6","base_url": None},
    "gemini":   {"provider": "gemini", "model": "gemini-2.5-pro",      "base_url": None},
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
        self.cfg = cfg
        self.provider = cfg.provider
        self.model = cfg.model
        self.temperature = cfg.temperature

        if cfg.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY for OpenAI provider")
            self.client = OpenAI(api_key=api_key, base_url=cfg.base_url)

        elif cfg.provider == "apifree":
            api_key = os.getenv("APIFREE_KEY")
            if not api_key:
                raise RuntimeError("Missing APIFREE_KEY for APIFree-backed provider")
            self.client = OpenAI(api_key=api_key, base_url=cfg.base_url)

        elif cfg.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("Missing ANTHROPIC_API_KEY for Anthropic provider")
            self.client = anthropic.Anthropic(api_key=api_key)

        elif cfg.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing GEMINI_API_KEY for Gemini provider")
            self.client = genai.Client(api_key=api_key)

        else:
            raise RuntimeError(f"Unsupported provider: {cfg.provider}")

    def _generate_openai_compatible(self, prompt: str) -> str:
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
                logger.exception(
                    "LLM API call failed for model=%s attempt=%s",
                    self.model, attempt + 1
                )
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

    def _generate_anthropic(self, prompt: str) -> str:
        last_error = None

        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as e:
                last_error = e
                logger.exception(
                    "Anthropic API call failed for model=%s attempt=%s",
                    self.model, attempt + 1
                )
                time.sleep(1.5 * (attempt + 1))
                continue

            if resp is None:
                last_error = RuntimeError(f"Anthropic returned None for model={self.model}")
                time.sleep(1.5 * (attempt + 1))
                continue

            content = getattr(resp, "content", None)
            if not content:
                raise RuntimeError(f"Anthropic returned empty content for model={self.model}: {resp}")

            parts = []
            for block in content:
                if getattr(block, "type", None) == "text":
                    block_text = getattr(block, "text", None)
                    if block_text:
                        parts.append(block_text)

            text = "\n".join(parts).strip()
            if text:
                return text

            raise RuntimeError(f"Anthropic could not extract text for model={self.model}: {resp}")

        raise RuntimeError(f"Anthropic API failed after retries for model={self.model}: {last_error}")

    def _generate_gemini(self, prompt: str) -> str:
        last_error = None

        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
            except Exception as e:
                last_error = e
                logger.exception(
                    "Gemini API call failed for model=%s attempt=%s",
                    self.model, attempt + 1
                )
                time.sleep(1.5 * (attempt + 1))
                continue

            if resp is None:
                last_error = RuntimeError(f"Gemini returned None for model={self.model}")
                time.sleep(1.5 * (attempt + 1))
                continue

            text = getattr(resp, "text", None)
            if text and text.strip():
                return text.strip()

            candidates = getattr(resp, "candidates", None)
            if candidates:
                parts = []
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    for part in getattr(content, "parts", None) or []:
                        part_text = getattr(part, "text", None)
                        if part_text:
                            parts.append(part_text)

                text = "\n".join(parts).strip()
                if text:
                    return text

            raise RuntimeError(f"Gemini could not extract text for model={self.model}: {resp}")

        raise RuntimeError(f"Gemini API failed after retries for model={self.model}: {last_error}")

    def generate(self, prompt: str) -> str:
        if self.provider in {"openai", "apifree"}:
            return self._generate_openai_compatible(prompt)

        if self.provider == "anthropic":
            return self._generate_anthropic(prompt)

        if self.provider == "gemini":
            return self._generate_gemini(prompt)

        raise RuntimeError(f"Unsupported provider at generate(): {self.provider}")