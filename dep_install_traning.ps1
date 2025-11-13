<#
dep_install_traning.ps1

Idempotent installer for Training dependencies. Reads package list from
`Training\requirements.txt`. Recommended to run inside a Python virtualenv.

Usage:
  .\dep_install_traning.ps1          # install from PyPI
  .\dep_install_traning.ps1 -Wheelhouse # install from local wheelhouse
#>

param(
    [switch]$Wheelhouse
)

if (-not (Test-Path '.\Training\requirements.txt')) {
    Write-Error "Training\requirements.txt not found. Please add it before running this installer."
    exit 1
}

if (-not (Test-Path '.venv') -and -not $env:VIRTUAL_ENV) {
    Write-Warning "No virtual environment detected. It's recommended to run in a venv to avoid system-wide installs."
}

try {
    Write-Host "Upgrading pip..."
    python -m pip install --upgrade pip
} catch {
    Write-Error "Python/pip not available on PATH or failed to run."
    exit 2
}

if ($Wheelhouse) {
    Write-Host "Installing from local wheelhouse..."
    python -m pip install --no-index --find-links=wheelhouse -r Training\requirements.txt
} else {
    Write-Host "Installing from PyPI..."
    python -m pip install -r Training\requirements.txt
}

# Create commonly used folders if missing
foreach ($d in @("data", "doc_folder")) {
    $path = Join-Path $PSScriptRoot $d
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
        Write-Host "Created: $path"
    }
}

Write-Host "dep_install_traning.ps1 finished successfully."
