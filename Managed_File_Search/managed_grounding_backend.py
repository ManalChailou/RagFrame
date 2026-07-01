import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI


class ManagedGroundingBackend:
    """
    Managed retrieval backend for clean COSMIC text passages.

    Metadata is encoded in filenames and used internally for filtering.
    Returned contexts contain only the useful passage text.
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
        self.vector_store_id = (
            vector_store_id or os.getenv("OPENAI_VECTOR_STORE_ID")
        )
        self.model = model
        self.temperature = temperature
        self.max_num_results = max(3, int(max_num_results))
        self.max_contexts = max(1, int(max_contexts))
        self.max_chars_per_context = max(
            300, int(max_chars_per_context)
        )
        self.max_total_context_chars = max(
            self.max_chars_per_context,
            int(max_total_context_chars),
        )
        self.min_score = float(min_score)

        if not self.vector_store_id:
            raise RuntimeError(
                "Missing OPENAI_VECTOR_STORE_ID. "
                "Create a vector store and set it in .env"
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
    def _filename_metadata(filename: str) -> Dict[str, str]:
        name = (filename or "").lower()

        metadata = {
            "app_domain": "general",
            "cosmic_component": "",
            "record_type": "knowledge",
        }

        domain_match = re.search(
            r"__domain-(.+?)__component-",
            name,
        )
        component_match = re.search(
            r"__component-(.+?)__type-",
            name,
        )
        type_match = re.search(
            r"__type-(.+?)__record-",
            name,
        )

        if domain_match:
            metadata["app_domain"] = domain_match.group(1)

        if component_match:
            metadata["cosmic_component"] = component_match.group(1)

        if type_match:
            metadata["record_type"] = type_match.group(1)

        return metadata

    @staticmethod
    def _source_type(record_type: str) -> str:
        normalized = (record_type or "").lower()

        if normalized == "example":
            return "examples"

        if normalized in {
            "rule",
            "rules",
            "definition",
            "validation",
            "quality_assurance",
        }:
            return "principles"

        return "guidelines"

    @staticmethod
    def _allowed_components(context_key: Optional[str]) -> set[str]:
        component = (context_key or "").strip().lower()

        if component == "sub_processes":
            return {
                "sub_processes",
                "data_movements",
                "quality_assurance",
            }

        return {component} if component else set()

    def _trim_text(self, text: str) -> str:
        compact = self._normalize_text(text)

        if len(compact) <= self.max_chars_per_context:
            return compact

        cut = compact[: self.max_chars_per_context]
        last_stop = max(
            cut.rfind(". "),
            cut.rfind("; "),
            cut.rfind(": "),
        )

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
            query = query[: self.MAX_QUERY_CHARS].rstrip()

        limit = max(
            1,
            int(max_num_results or self.max_num_results),
        )

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

            filename = (
                data.get("filename")
                or getattr(result, "filename", None)
                or ""
            )
            metadata = self._filename_metadata(filename)

            normalized.append(
                {
                    "score": score_value,
                    "text": text,
                    "source_type": self._source_type(
                        metadata["record_type"]
                    ),
                    **metadata,
                }
            )

        return normalized

    def retrieve_contexts(
        self,
        query: str,
        max_num_results: Optional[int] = None,
        context_key: Optional[str] = None,
        app_domain: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Retrieve focused passages.

        The method uses metadata internally, but the returned result exposes
        only {"text": "..."} so debugging output and LLM context stay clean.
        """
        limit = max_num_results or self.max_num_results
        requested_domain = (
            app_domain or "general"
        ).strip().lower()
        allowed_components = self._allowed_components(context_key)

        results = self._search(query, limit)

        filtered: List[Dict[str, Any]] = []
        seen = set()

        for item in results:
            if item.get("score", 0) < self.min_score:
                continue

            item_domain = item.get("app_domain", "general")
            if item_domain not in {"general", requested_domain}:
                continue

            item_component = item.get("cosmic_component", "")
            if allowed_components and item_component not in allowed_components:
                continue

            text = self._normalize_text(item.get("text", ""))
            if not text:
                continue

            signature = re.sub(r"\W+", "", text.lower())[:500]
            if signature in seen:
                continue
            seen.add(signature)

            clean = dict(item)
            clean["text"] = self._trim_text(text)
            filtered.append(clean)

        filtered.sort(
            key=lambda item: item.get("score", 0),
            reverse=True,
        )

        selected: List[Dict[str, Any]] = []

        # Prefer one example, one rule/definition, and one guidance passage.
        for source_type in ("examples", "principles", "guidelines"):
            candidate = next(
                (
                    item
                    for item in filtered
                    if item.get("source_type") == source_type
                    and item not in selected
                ),
                None,
            )
            if candidate:
                selected.append(candidate)

        for item in filtered:
            if len(selected) >= self.max_contexts:
                break
            if item not in selected:
                selected.append(item)

        final: List[Dict[str, str]] = []
        used_chars = 0

        for item in selected[: self.max_contexts]:
            remaining = self.max_total_context_chars - used_chars
            if remaining <= 0:
                break

            text = item["text"]

            if len(text) > remaining:
                text = text[:remaining].rstrip() + " ..."

            # Expose only necessary information.
            final.append({"text": text})
            used_chars += len(text)

        return final

    @staticmethod
    def format_context(contexts: List[Dict[str, str]]) -> str:
        """
        Send only useful passage text to the LLM.
        No labels such as TYPE, COMPONENT, GUIDANCE, or SIMILAR EXAMPLE.
        """
        passages = []

        for context in contexts or []:
            text = (context.get("text") or "").strip()
            if text:
                passages.append(text)

        return "\n\n---\n\n".join(passages)
