from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("Service.vector_store/index.faiss")

with open("Service.vector_store/docs.json", "r") as f:
    documents = json.load(f)

def retrieve_context(query: str, top_k: int = 5) -> str:
    embedding = model.encode([query])
    _, indices = index.search(np.array(embedding), top_k)
    context = "\n".join([documents[i] for i in indices[0]])
    return context