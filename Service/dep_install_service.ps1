# Install dependencies for offline installation for services:
python -m pip install --no-index --find-links=wheelhouse `
    fastapi `
    uvicorn `
    requests `
    faiss-cpu `
    sentence-transformers `
    PyPDF2