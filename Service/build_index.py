from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import glob
import PyPDF2

# Sample documents
#documents = [
#    "CE marking ensures compliance with EU safety regulations.",
#    "ISO/IEC 27001 is a standard for information security management.",
#    "NIS2 directive enhances cybersecurity across EU member states.",
#    "Docker simplifies containerized deployments.",
#    "Tailscale provides secure remote access via WireGuard."
#]

documents = []

# Load TXT files (each file as one document)
for txt_path in glob.glob("docs_folder/*.txt"):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if text:
            documents.append(text)

# Load PDF files (each file as one document)
for pdf_path in glob.glob("docs_folder/*.pdf"):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        if text.strip():
            documents.append(text.strip())

if not documents:
    print("No documents found. Skipping RAG.")
    exit()

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and metadata
os.makedirs("Service.vector_store", exist_ok=True)
faiss.write_index(index, "Service.vector_store/index.faiss")

with open("Service.vector_store/docs.json", "w") as f:
    json.dump(documents, f)

print("✅ FAISS index built and saved.")