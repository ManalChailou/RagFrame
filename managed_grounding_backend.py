import os
from typing import Optional
from openai import OpenAI


class ManagedGroundingBackend:
    """
    Managed grounding backend using OpenAI File Search / Vector Stores.

    This replaces local embedding + cosine retrieval.
    The provider handles:
    - document parsing
    - chunking
    - embedding
    - vector storage
    - retrieval
    """

    def __init__(
        self,
        vector_store_id: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ):
        self.vector_store_id = vector_store_id or os.getenv("OPENAI_VECTOR_STORE_ID")
        self.model = model
        self.temperature = temperature

        if not self.vector_store_id:
            raise RuntimeError(
                "Missing OPENAI_VECTOR_STORE_ID. Create a vector store and set it in .env"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key)

    def generate_grounded_json_with_context(self, prompt: str) -> dict:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store_id],
                    "max_num_results": 5,
                }
            ],
            include=["file_search_call.results"],
        )

        retrieved_contexts = []

        for item in response.output:
            if getattr(item, "type", None) == "file_search_call":
                results = getattr(item, "results", None) or []
                for r in results:
                    retrieved_contexts.append({
                        "file_id": getattr(r, "file_id", None),
                        "filename": getattr(r, "filename", None),
                        "score": getattr(r, "score", None),
                        "text": getattr(r, "text", None),
                    })

        return {
            "text": response.output_text.strip(),
            "contexts": retrieved_contexts,
        }
    
    def generate_grounded_json(self, prompt: str) -> str:
        result = self.generate_grounded_json_with_context(prompt)
        return result["text"]