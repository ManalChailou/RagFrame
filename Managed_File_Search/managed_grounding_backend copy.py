import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI


class ManagedGroundingBackend:
    """
    Managed retrieval backend based on an OpenAI Vector Store.

    The retrieval is intentionally compact:
    - one rule-oriented search;
    - one example-oriented search;
    - at most one Principles passage, one Guidelines passage and one Example;
    - duplicate/noisy passages are removed;
    - every passage and the total context are truncated.
    """

    MAX_QUERY_CHARS = 3900

    def __init__(
        self,
        vector_store_id: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_num_results: int = 8,
        max_contexts: int = 3,
        max_chars_per_context: int = 1200,
        max_total_context_chars: int = 3200,
        min_score: float = 0.70,
    ):
        self.vector_store_id = vector_store_id or os.getenv("OPENAI_VECTOR_STORE_ID")
        self.model = model
        self.temperature = temperature
        self.max_num_results = max(3, int(max_num_results))
        self.max_contexts = max(1, int(max_contexts))
        self.max_chars_per_context = max(300, int(max_chars_per_context))
        self.max_total_context_chars = max(
            self.max_chars_per_context,
            int(max_total_context_chars),
        )
        self.min_score = float(min_score)

        if not self.vector_store_id:
            raise RuntimeError(
                "Missing OPENAI_VECTOR_STORE_ID. Create a vector store and set it in .env"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _as_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return {}

    @classmethod
    def _extract_text(cls, result: Any) -> str:
        data = cls._as_dict(result)
        content = data.get("content") or getattr(result, "content", None) or []

        parts: List[str] = []
        for item in content:
            item_data = cls._as_dict(item)
            text = (
                item_data.get("text")
                or item_data.get("content")
                or getattr(item, "text", None)
            )
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

        direct_text = data.get("text") or getattr(result, "text", None)
        if isinstance(direct_text, str) and direct_text.strip():
            parts.append(direct_text.strip())

        return "\n".join(dict.fromkeys(parts)).strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _filename_group(filename: str) -> str:
        name = (filename or "").lower()
        if "example" in name:
            return "examples"
        if "guideline" in name:
            return "guidelines"
        if "principle" in name or "definition" in name or "rule" in name:
            return "principles"
        return "other"

    @staticmethod
    def _is_noise(text: str) -> bool:
        low = (text or "").lower()
        return any(
            marker in low
            for marker in [
                "table of contents",
                "all rights reserved",
                "other members of cosmic",
                "editors:",
                "permission to copy",
                "measurement reporting",
                "local extensions labeling",
            ]
        )

    def _trim_text(self, text: str) -> str:
        compact = self._normalize_text(text)
        if len(compact) <= self.max_chars_per_context:
            return compact

        cut = compact[: self.max_chars_per_context]
        last_stop = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(": "))
        if last_stop >= int(self.max_chars_per_context * 0.65):
            cut = cut[: last_stop + 1]
        return cut.rstrip() + " ..."

    def _search(
        self,
        query: str,
        max_num_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query = self._normalize_text(query)
        if not query:
            return []

        if len(query) > self.MAX_QUERY_CHARS:
            head_size = 2300
            tail_size = self.MAX_QUERY_CHARS - head_size - 5
            query = f"{query[:head_size].rstrip()} ... {query[-tail_size:].lstrip()}"

        limit = max(1, int(max_num_results or self.max_num_results))

        try:
            response = self.client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=query,
                max_num_results=limit,
            )
        except TypeError:
            response = self.client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=query,
            )

        raw_results = getattr(response, "data", None)
        if raw_results is None:
            raw_results = self._as_dict(response).get("data", [])

        normalized: List[Dict[str, Any]] = []
        for result in list(raw_results or [])[:limit]:
            data = self._as_dict(result)
            text = self._extract_text(result)
            if not text:
                continue

            score = data.get("score") or getattr(result, "score", None)
            try:
                score_value = float(score) if score is not None else 0.0
            except (TypeError, ValueError):
                score_value = 0.0

            normalized.append(
                {
                    "file_id": data.get("file_id") or getattr(result, "file_id", None),
                    "filename": data.get("filename") or getattr(result, "filename", None),
                    "score": score_value,
                    "text": text,
                    "attributes": data.get("attributes")
                    or getattr(result, "attributes", None),
                }
            )

        return normalized

    def retrieve_contexts(
        self,
        query: str,
        max_num_results: Optional[int] = None,
        context_key: Optional[str] = None,
        app_domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a short and diverse context.

        The second search explicitly targets COSMIC-Examples.pdf so matching
        examples are not hidden by the rule manuals.
        """
        limit = max_num_results or self.max_num_results

        rule_results = self._search(query, limit)
        domain = (app_domain or "").strip().lower()

        if domain == "business":
            example_filename = "COSMIC-Examples-Business.txt"
        elif domain == "real_time":
            example_filename = "COSMIC-Examples-Real-Time.txt"
        else:
            example_filename = "COSMIC-Examples"

        example_query = (
            f"{query}\n"
            f"Find the most similar worked COSMIC example from "
            f"{example_filename}. "
            "Match the requirement vocabulary, actors, objects of interest, "
            "triggering event, data groups and movement pattern."
        )
        example_results = self._search(example_query, limit)

        combined = rule_results + example_results

        unique: List[Dict[str, Any]] = []
        seen = set()

        for item in sorted(
            combined,
            key=lambda value: value.get("score", 0),
            reverse=True,
        ):
            text = self._normalize_text(item.get("text", ""))

            if not text or self._is_noise(text):
                continue
            if item.get("score", 0) < self.min_score:
                continue

            signature = re.sub(r"\W+", "", text.lower())[:500]
            if signature in seen:
                continue
            seen.add(signature)

            clean = dict(item)
            clean["text"] = self._trim_text(text)
            clean["source_type"] = self._filename_group(
                clean.get("filename") or ""
            )
            if clean["source_type"] == "examples":
                filename = (clean.get("filename") or "").lower()

                if domain == "business":
                    if "cosmic-examples-business" not in filename:
                        continue

                elif domain == "real_time":
                    if "cosmic-examples-real-time" not in filename:
                        continue
            unique.append(clean)

        selected: List[Dict[str, Any]] = []

        # Guarantee source diversity when available.
        for source_type in ("examples", "principles", "guidelines"):
            candidate = next(
                (
                    item
                    for item in unique
                    if item.get("source_type") == source_type
                ),
                None,
            )
            if candidate:
                selected.append(candidate)

        for item in unique:
            if len(selected) >= self.max_contexts:
                break
            if item not in selected:
                selected.append(item)

        selected = selected[: self.max_contexts]

        final: List[Dict[str, Any]] = []
        used_chars = 0

        for item in selected:
            remaining = self.max_total_context_chars - used_chars
            if remaining <= 0:
                break

            text = item["text"]
            if len(text) > remaining:
                text = text[:remaining].rstrip() + " ..."

            compact_item = dict(item)
            compact_item["text"] = text
            final.append(compact_item)
            used_chars += len(text)

        return final

    @staticmethod
    def format_context(contexts: List[Dict[str, Any]]) -> str:
        labels = {
            "examples": "SIMILAR EXAMPLE",
            "principles": "MANDATORY RULES",
            "guidelines": "GUIDANCE",
            "other": "REFERENCE",
        }

        blocks: List[str] = []
        for context in contexts or []:
            text = (context.get("text") or "").strip()
            if not text:
                continue

            source_type = context.get("source_type", "other")
            blocks.append(
                f"[{labels.get(source_type, 'REFERENCE')}]\n{text}"
            )

        return "\n\n".join(blocks)

    def generate_grounded_json_with_context(self, prompt: str) -> Dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store_id],
                    "max_num_results": self.max_num_results,
                }
            ],
            include=["file_search_call.results"],
        )

        retrieved_contexts: List[Dict[str, Any]] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "file_search_call":
                continue

            for result in getattr(item, "results", None) or []:
                data = self._as_dict(result)
                retrieved_contexts.append(
                    {
                        "file_id": data.get("file_id")
                        or getattr(result, "file_id", None),
                        "filename": data.get("filename")
                        or getattr(result, "filename", None),
                        "score": data.get("score")
                        or getattr(result, "score", None),
                        "text": self._extract_text(result),
                    }
                )

        return {
            "text": (getattr(response, "output_text", "") or "").strip(),
            "contexts": retrieved_contexts,
        }

    def generate_grounded_json(self, prompt: str) -> str:
        return self.generate_grounded_json_with_context(prompt)["text"]
