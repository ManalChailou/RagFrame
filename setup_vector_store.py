# setup_vector_store.py

from contextlib import ExitStack
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

vector_store = client.vector_stores.create(
    name="COSMIC Knowledge Base"
)

file_paths = [
    "rag_docs/Part-1-MM-Principles-Definitions-Rules-v5.0-Aug-2021.pdf",
    "rag_docs/Part-2-MM-Guidelines-v5.0-Sep-2024.pdf",
]

with ExitStack() as stack:
    file_streams = [stack.enter_context(open(path, "rb")) for path in file_paths]

    batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=file_streams,
    )

print("Vector Store ID:", vector_store.id)
print("Batch status:", batch.status)