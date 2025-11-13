param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$pythonExe = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$commandArgs = @('-m', 'Training.frameworks_drivers.cli.download_runner_cli') + $Args

& $pythonExe @commandArgs
exit $LASTEXITCODE
