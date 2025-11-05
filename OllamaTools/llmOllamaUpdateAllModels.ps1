# Updating AI models by enumerating installed models from `ollama list`
# For each model found, run `ollama pull <model>` and log results to CSV.
# Follows fail-fast and simple validation.

$scriptName = Split-Path -Leaf $PSCommandPath
$loggerScript = Join-Path $PSScriptRoot 'AI\Logger.ps1'

function Write-LogMessage {
    param(
        [ValidateSet('INFO','WARN','ERROR')][string]$Level = 'INFO',
        [Parameter(Mandatory)][string]$Id,
        [Parameter(Mandatory)][string]$Message
    )

    switch ($Level) {
        'ERROR' { Write-Error $Message }
        'WARN'  { Write-Warning $Message }
        default { Write-Host $Message }
    }

    if (Test-Path -LiteralPath $loggerScript) {
        & $loggerScript -LogLevel $Level -Source $scriptName -Id $Id -Message $Message
    }
}

$logDir = Join-Path $PSScriptRoot 'logs'
$logPath = Join-Path $logDir 'ollama-update-log.csv'

function Fail([string]$msg) {
    Write-LogMessage -Level 'ERROR' -Id 'Fail' -Message $msg
    exit 1
}

# Ensure ollama is available
if (-not (Get-Command 'ollama' -ErrorAction SilentlyContinue)) {
    Fail "The 'ollama' CLI was not found in PATH. Install ollama or adjust PATH."
}

# Ensure log directory exists
if (-not (Test-Path $logDir)) {
    New-Item -Path $logDir -ItemType Directory | Out-Null
    Write-LogMessage -Id 'CreateLogDir' -Message "Created log directory at $logDir"
}

function Get-InstalledModels {
    # Run ollama list and extract model identifiers like "name:tag"
    $raw = & ollama list 2>&1
    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to run 'ollama list': $($raw -join ' | ')"
    }

    $models = @()
    foreach ($line in $raw) {
        $line = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        # match token containing ':' (e.g. deepseek-r1:1.5b or gemma2:latest)
        # Note: avoid invalid escape sequences like \_ in .NET regex character classes
        if ($line -match '([A-Za-z0-9._-]+:[^\s,;]+)') {
            $models += $matches[1]
        }
    }

    return $models | Select-Object -Unique
}

function PullModel {
    param([string]$model)

    Write-Host "Updating model: $model"
    $output = & ollama pull $model 2>&1
    $exit = $LASTEXITCODE

    $message = ($output -join ' | ')
    $status = if ($exit -eq 0) { 'SUCCESS' } else { 'ERROR' }

    $record = [PSCustomObject]@{
        Timestamp = (Get-Date).ToString('o')
        Model     = $model
        Status    = $status
        Message   = $message
    }

    # Write header if file doesn't exist, otherwise append
    if (-not (Test-Path $logPath)) {
        $record | Export-Csv -Path $logPath -NoTypeInformation
    } else {
        $record | Export-Csv -Path $logPath -NoTypeInformation -Append
    }

    if ($exit -ne 0) {
        Write-LogMessage -Level 'WARN' -Id 'PullModel' -Message "Failed updating $model : $message"
    } else {
        Write-LogMessage -Id 'PullModel' -Message "Updated $model successfully"
    }

    return $exit
}

# Main
$models = Get-InstalledModels
if (-not $models -or $models.Count -eq 0) {
    Write-LogMessage -Level 'WARN' -Id 'GetInstalledModels' -Message "No models found by 'ollama list'."
    exit 0
}

foreach ($m in $models) {
    Write-LogMessage -Id 'PullModel' -Message "Updating model: $m"
    PullModel -model $m
}
