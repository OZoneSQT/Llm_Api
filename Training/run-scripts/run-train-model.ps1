param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$pythonExe = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$commandArgs = @('-m', 'Training.frameworks_drivers.cli.train_model_cli') + $Args

& $pythonExe @commandArgs
exit $LASTEXITCODE
