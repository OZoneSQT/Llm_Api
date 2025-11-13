[CmdletBinding()]
param(
    [string]$EnvironmentName = 'llm_api',
    [string]$EnvironmentFile,
    [switch]$RunTests,
    [string[]]$PytestArgs = @('Training/tests', '-m', 'not smoke')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'conda_helpers.ps1')

if (-not $EnvironmentFile) {
    $trainingRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
    $EnvironmentFile = Join-Path $trainingRoot 'environment.yml'
}

if (-not (Test-Path $EnvironmentFile)) {
    throw "Environment specification not found at $EnvironmentFile"
}

Invoke-CondaCommand -Arguments @('env', 'update', '--name', $EnvironmentName, '--file', $EnvironmentFile, '--prune')
Invoke-CondaCommand -Arguments @('run', '-n', $EnvironmentName, 'python', '-m', 'pip', 'install', '--upgrade', 'pip')

Write-Host "Environment '$EnvironmentName' is ready. Run 'conda activate $EnvironmentName' to start using it." -ForegroundColor Green

if ($RunTests) {
    Write-Host 'Executing pytest suite inside the freshly provisioned environment...' -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot 'run-tests.ps1') -EnvironmentName $EnvironmentName -PytestArgs $PytestArgs
}
