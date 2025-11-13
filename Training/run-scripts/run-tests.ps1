[CmdletBinding()]
param(
    [string]$EnvironmentName = 'llm_api',
    [string]$Marker = 'not smoke',
    [switch]$IncludeSmoke,
    [string[]]$ExtraArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'conda_helpers.ps1')

$pytestArgs = @('Training/tests')
if ($Marker -and -not $IncludeSmoke) {
    $pytestArgs += @('-m', $Marker)
}
if ($ExtraArgs) {
    $pytestArgs += $ExtraArgs
}

if ($IncludeSmoke) {
    $pytestCommand = ($pytestArgs | ForEach-Object { '"{0}"' -f ($_ -replace '"', '``"') })
    $commandString = 'Set-Item Env:RUN_SMOKE_TESTS 1; ' + ('python -m pytest ' + ($pytestCommand -join ' '))
    Invoke-CondaCommand -Arguments @('run', '-n', $EnvironmentName, 'powershell', '-NoProfile', '-Command', $commandString)
} else {
    Invoke-CondaCommand -Arguments (@('run', '-n', $EnvironmentName, 'python', '-m', 'pytest') + $pytestArgs)
}
