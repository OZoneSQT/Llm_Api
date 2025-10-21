import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, Query
from .rag import retrieve_context
from .ollama_client import query_ollama
app = FastAPI()

@app.get("/ask")
def ask(query: str = Query(..., description="User question")):
    context = retrieve_context(query)
    full_prompt = f"Answer the question using the context below:\n\n{context}\n\nQuestion: {query}"
    answer = query_ollama(full_prompt)
    return {"answer": answer}