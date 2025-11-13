[CmdletBinding()]
param(
	[string]$EnvironmentName = 'llm_api'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '..\scripts\conda_helpers.ps1')

Invoke-CondaCommand -Arguments @('run', '-n', $EnvironmentName, 'python', '-m', 'Training.frameworks_drivers.cli.train_model_cli')