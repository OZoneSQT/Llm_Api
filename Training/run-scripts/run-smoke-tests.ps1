param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

# Set env var for smoke tests and forward to the existing run-tests.ps1 implementation
$env:RUN_SMOKE_TESTS = '1'
$script = Join-Path $PSScriptRoot 'run-tests.ps1'
& $script -IncludeSmoke -ExtraArgs $Args
exit $LASTEXITCODE
