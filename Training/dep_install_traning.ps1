if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
	Write-Host "Script is not running as administrator. Relaunching as admin..."
	Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File \"$PSCommandPath\"" -Verb RunAs
	exit
}

winget install Kitware.CMake

$env:Path += ";C:\Program Files\CMake\bin"

if (!(Test-Path .\lib)) {
	New-Item -ItemType Directory -Path .\lib | Out-Null
}

Set-Location .\lib

git clone https://github.com/ggerganov/llama.cpp
Set-Location .\llama.cpp
pip install -r requirements.txt
Set-Location ..

# https://github.com/huggingface
git clone https://github.com/huggingface/transformers.git
Set-Location .\transformers
pip install .
Set-Location ..

git clone https://github.com/huggingface/datasets.git
Set-Location .\datasets
pip install .
Set-Location ..

git clone https://github.com/huggingface/accelerate.git
Set-Location .\accelerate
pip install .
Set-Location ..

git clone https://github.com/huggingface/diffusers.git
Set-Location .\diffusers
pip install .
Set-Location ..

git clone https://github.com/huggingface/peft.git
Set-Location .\peft
pip install .
Set-Location ..

git clone https://github.com/huggingface/optimum.git
Set-Location .\optimum
pip install .
Set-Location ..\..

pip install PyPDF2
pip install ebooklib
pip install mobi
pip install glob
pip install os
pip install json
pip install torch
pip install time
pip install datetime
pip install python-docx
python -m pip install --upgrade "pip<24.1"
pip install textract
python -m pip install --upgrade pip
pip install --upgrade six

if (!(Test-Path "$PSScriptRoot/data")) {
	New-Item -ItemType Directory -Path "$PSScriptRoot/data" | Out-Null
}
if (!(Test-Path "$PSScriptRoot/doc_folder")) {
	New-Item -ItemType Directory -Path "$PSScriptRoot/doc_folder" | Out-Null
}
