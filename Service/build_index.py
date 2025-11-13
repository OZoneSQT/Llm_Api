from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import glob
import PyPDF2
import ebooklib
from ebooklib import epub
import mobi

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
for txt_path in glob.glob("docs_folder/*.txt") + glob.glob("docs_folder/*.TXTB"):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if text:
            documents.append(text)

# Load PDF files (each file as one document)
for pdf_path in glob.glob("docs_folder/*.pdf") + glob.glob("docs_folder/*.PDF"):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        if text.strip():
            documents.append(text.strip())

# Load EPUB files (each file as one document)
for epub_path in glob.glob("docs_folder/*.epub") + glob.glob("docs_folder/*.EPUB"):
    book = epub.read_epub(epub_path)
    text = ""
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            text += item.get_body_content_str()
    if text.strip():
        documents.append(text.strip())

# Load MOBI files (each file as one document)
for mobi_path in glob.glob("docs_folder/*.mobi") + glob.glob("docs_folder/*.MOBI"):
    book = mobi.Mobi(mobi_path)
    book.parse()
    text = book.get_text()
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