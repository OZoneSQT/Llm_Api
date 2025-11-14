[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script = Join-Path $PSScriptRoot 'image_modifier.py'
# Use the cross-platform Python wrapper to run inside the `acaonda` environment
& python (Join-Path $PSScriptRoot 'run_in_acaonda.py') $script @Args
