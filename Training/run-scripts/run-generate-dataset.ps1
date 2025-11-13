param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$pythonExe = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$commandArgs = @('-m', 'Training.frameworks_drivers.cli.generate_dataset_cli') + $Args

& $pythonExe @commandArgs
exit $LASTEXITCODE
