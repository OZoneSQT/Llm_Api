[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script = Join-Path $PSScriptRoot 'audio_generator.py'
& (Join-Path $PSScriptRoot 'run_in_acaonda.ps1') -Script $script -ScriptArgs $Args
