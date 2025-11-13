# Install dependencies for offline installation for services:
python -m pip install --no-index --find-links=wheelhouse `
    fastapi `
    "fastapi>=0.103.0,<2.0.0" `
    uvicorn[standard] `
    requests `
    faiss-cpu `
    sentence-transformers `
    PyPDF2 `
    torch `
    transformers `
    sentencepiece `
    datasets `
    ebooklib `
    mobi