Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-CondaExecutable {
        if (Test-Path Variable:\script:CondaExecutable) {
            return $script:CondaExecutable
        }

    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($command) {
        $script:CondaExecutable = $command.Source
        return $script:CondaExecutable
    }

    $candidatePaths = @()
    if ($env:CONDA_PREFIX) {
        $candidatePaths += Join-Path $env:CONDA_PREFIX 'Scripts\conda.exe'
        $candidatePaths += Join-Path $env:CONDA_PREFIX 'condabin\conda.bat'
    }
    $candidatePaths += Join-Path $env:USERPROFILE 'Miniconda3\Scripts\conda.exe'
    $candidatePaths += Join-Path $env:USERPROFILE 'Miniconda3\condabin\conda.bat'
    $candidatePaths += Join-Path $env:USERPROFILE 'Anaconda3\Scripts\conda.exe'
    $candidatePaths += Join-Path $env:USERPROFILE 'Anaconda3\condabin\conda.bat'

    $candidates = $candidatePaths | Where-Object { $_ -and (Test-Path $_) }

    if ($candidates.Count -eq 0) {
        throw 'Unable to locate conda executable. Install Miniconda or Anaconda and ensure conda is on PATH.'
    }

    $script:CondaExecutable = $candidates[0]
    return $script:CondaExecutable
}

function Invoke-CondaCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $condaPath = Get-CondaExecutable
    & $condaPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "conda command failed: $($Arguments -join ' ')"
    }
}
