if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
	Write-Host "Script is not running as administrator. Relaunching as admin..."
	Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File \"$PSCommandPath\"" -Verb RunAs
	exit
}

$env:Path += ";C:\Program Files\CMake\bin"

Set-Location .\lib\llama.cpp
git pull
python -m pip install -r requirements.txt

Set-Location .\lib\transformers
git pull
pip install --upgrade .

Set-Location .\lib\datasets
git pull
pip install --upgrade .

Set-Location .\lib\accelerate
git pull
pip install --upgrade .

Set-Location .\lib\diffusers
git pull
pip install --upgrade .

Set-Location .\lib\peft
git pull
pip install --upgrade .

Set-Location .\lib\optimum
git pull
pip install --upgrade .

Set-Location .\
