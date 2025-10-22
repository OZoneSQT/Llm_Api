# Install dependencies for offline installation for services:
python -m pip install --no-index --find-links=wheelhouse `
    fastapi `
    uvicorn `
    requests `
    faiss-cpu `
    sentence-transformers `
    PyPDF2


if (!(Test-Path "$PSScriptRoot/data")) {
	New-Item -ItemType Directory -Path "$PSScriptRoot/data" | Out-Null
}

if (!(Test-Path "$PSScriptRoot/vector_store")) {
	New-Item -ItemType Directory -Path "$PSScriptRoot/vector_store" | Out-Null
}

if (!(Test-Path "$PSScriptRoot/doc_folder")) {
	New-Item -ItemType Directory -Path "$PSScriptRoot/doc_folder" | Out-Null
}
