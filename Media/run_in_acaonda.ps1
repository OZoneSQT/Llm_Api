[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Script,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Try to locate the Training scripts helper (conda_helpers.ps1). We expect this repo layout:
# <repo>/Training/scripts/conda_helpers.ps1
$possible = @(
    (Join-Path $PSScriptRoot '..\Training\scripts\conda_helpers.ps1'),
    (Join-Path $PSScriptRoot '..\..\Training\scripts\conda_helpers.ps1')
)

$condaHelpers = $possible | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $condaHelpers) {
    throw "Could not find Training/scripts/conda_helpers.ps1. Expected one of: $($possible -join '; ')"
}

# The helper which provides Invoke-CondaCommand
. $condaHelpers

# Resolve the target script path and build conda run arguments
$resolvedScript = Resolve-Path -Path $Script -ErrorAction Stop
$argsArray = @('run', '-n', 'acaonda', 'python', $resolvedScript.Path) + $ScriptArgs
$activity = "acaonda :: $($resolvedScript.Path)"
$argumentPreview = if ($ScriptArgs) { $ScriptArgs -join ' ' } else { '(no additional arguments)' }
Write-Host "[acaonda] Starting: python $($resolvedScript.Path) $argumentPreview" -ForegroundColor Cyan
$start = Get-Date

$progressId = Get-Random -Maximum 10000
$progressActivity = "[acaonda] $($resolvedScript.Path)"
$spinner = @('|','/','-','\')
$timer = New-Object System.Timers.Timer 250
$timer.AutoReset = $true
$eventRegistration = Register-ObjectEvent -InputObject $timer -EventName Elapsed -MessageData ([hashtable]::Synchronized(@{ Index = 0 })) -Action {
    $data = $event.MessageData
    $data.Index = ($data.Index + 1) % $using:spinner.Count
    Write-Progress -Id $using:progressId -Activity $using:progressActivity -Status ("Running " + $using:spinner[$data.Index]) -PercentComplete -1
}
$timer.Start()
Write-Progress -Id $progressId -Activity $progressActivity -Status 'Starting...' -PercentComplete -1

try {
    Invoke-CondaCommand -Arguments $argsArray
    $duration = (Get-Date) - $start
    Write-Progress -Id $progressId -Activity $progressActivity -Completed -Status 'Completed'
    Write-Host "[acaonda] Completed: $activity in $([math]::Round($duration.TotalSeconds, 2))s" -ForegroundColor Green
}
catch {
    $duration = (Get-Date) - $start
    Write-Progress -Id $progressId -Activity $progressActivity -Completed -Status 'Failed'
    Write-Host "[acaonda] Failed: $activity after $([math]::Round($duration.TotalSeconds, 2))s" -ForegroundColor Red
    throw
}
finally {
    $timer.Stop()
    $timer.Dispose()
    if ($eventRegistration) {
        Unregister-Event -SourceIdentifier $eventRegistration.Name -ErrorAction SilentlyContinue
        Remove-Job -Id $eventRegistration.Id -Force -ErrorAction SilentlyContinue
    }
}
