# Download dependencies for offline installation

$LibraryPath = "wheelhouse"
$timestampFile = "last_updated.txt"

if (Test-Path $LibraryPath) {
    Remove-Item $LibraryPath -Recurse -Force
}
New-Item -ItemType Directory -Path $LibraryPath | Out-Null

python -m pip download --dest $LibraryPath `
    "fastapi>=0.103.0,<2.0.0" `
    "uvicorn[standard]" `
    "requests" `
    "faiss-cpu" `
    "sentence-transformers" `
    "PyPDF2" `
    "torch" `
    "transformers" `
    "sentencepiece" `
    "datasets" `
    "ebooklib" `
    "mobi"

# Remove old timestamp file if it exists
if (Test-Path $timestampFile) {
    Remove-Item $timestampFile -Force
}

if ($LASTEXITCODE -eq 0) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Set-Content -Path $timestampFile -Value "Dependencies downloaded: $timestamp"
} else {
    Write-Host "Dependency download failed. Timestamp not updated."
}