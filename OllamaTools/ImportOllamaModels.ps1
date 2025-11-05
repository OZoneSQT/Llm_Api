param(
    [string]$BackupFolder = (Join-Path (Get-Location) "backup"),
    [switch]$Force
)

$scriptName = Split-Path -Leaf $PSCommandPath
$loggerScript = Join-Path $PSScriptRoot 'Logger.ps1'

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

function Invoke-OllamaCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & ollama @Arguments 2>&1

    if ($LASTEXITCODE -ne 0) {
        $message = if ($output) { ($output -join "`n") } else { "Unknown error" }
        $err = "ollama $($Arguments -join ' ') failed: $message"
        Write-LogMessage -Level 'ERROR' -Id 'InvokeOllamaCommand' -Message $err
        throw $err
    }

    if ($null -eq $output) {
        return ""
    }

    return ($output -join "`n").Trim()
}

if (-not (Test-Path -LiteralPath $BackupFolder)) {
    $err = "Backup folder '$BackupFolder' does not exist."
    Write-LogMessage -Level 'ERROR' -Id 'ValidateBackupFolder' -Message $err
    throw $err
}

$resolvedBackupRoot = (Resolve-Path -LiteralPath $BackupFolder).Path

$directoriesToProcess = @()
$directoriesToProcess += Get-Item -LiteralPath $resolvedBackupRoot
$directoriesToProcess += Get-ChildItem -Path $resolvedBackupRoot -Directory -Recurse

foreach ($directory in $directoriesToProcess) {
    $ggufFiles = Get-ChildItem -Path $directory.FullName -Filter *.gguf -File -ErrorAction SilentlyContinue
    if (-not $ggufFiles) {
        continue
    }

    $modelName = $directory.Name
    Write-LogMessage -Id 'ImportModel' -Message "Importing model: $modelName"

    $modelFileCandidates = @('ModelFile', 'modelfile') |
        ForEach-Object { Join-Path -Path $directory.FullName -ChildPath $_ }

    $modelFilePath = $modelFileCandidates |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

    if (-not $modelFilePath) {
        Write-LogMessage -Level 'WARN' -Id 'ModelFileMissing' -Message "Skipping '$modelName' because no ModelFile was found."
        continue
    }

    $arguments = @('create', $modelName, '-f', $modelFilePath)
    if ($Force.IsPresent) {
        $arguments += '--force'
    }

    try {
        $result = Invoke-OllamaCommand -Arguments $arguments
        if ($result) {
            Write-LogMessage -Id 'ImportModelResult' -Message $result
        }
    } catch {
        Write-LogMessage -Level 'WARN' -Id 'ImportModel' -Message "Failed to import '$modelName'. $_"
    }
}
