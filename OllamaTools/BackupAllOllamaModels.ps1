param(
    [string]$BackupFolder = (Join-Path (Get-Location) "backup"),
    [string]$OllamaModelsDirectory = (Join-Path -Path (Join-Path -Path $env:USERPROFILE -ChildPath ".ollama") -ChildPath "models")
)

$scriptName = Split-Path -Leaf $PSCommandPath
$loggerScript = Join-Path $PSScriptRoot 'Logger.ps1'

function Write-LogMessage {
    param(
        [ValidateSet('INFO','WARN','ERROR')][string]$Level = 'INFO',
        [Parameter(Mandatory = $true)][string]$Id,
        [Parameter(Mandatory = $true)][string]$Message
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

function ConvertTo-SafeModelName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $cleanName = $Name -replace ":latest$", ""
    return ($cleanName -replace '[<>:"/\|?*.]', "-")
}

try {
    $listOutput = Invoke-OllamaCommand -Arguments @('list')
} catch {
    Write-LogMessage -Level 'ERROR' -Id 'ListModels' -Message $_
    exit 1
}

if (-not $listOutput) {
    Write-LogMessage -Level 'WARN' -Id 'ListModels' -Message "No models were returned by 'ollama list'."
    exit 0
}

$models = $listOutput -split "`r?`n" | Select-Object -Skip 1 |
    ForEach-Object { ($_ -split '\s+')[0] } |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    Sort-Object -Unique

if (-not $models) {
    Write-LogMessage -Level 'WARN' -Id 'ResolveModels' -Message "No models found to back up."
    exit 0
}

if (-not (Test-Path -LiteralPath $BackupFolder)) {
    try {
        [void](New-Item -ItemType Directory -Path $BackupFolder -Force)
        Write-LogMessage -Id 'CreateBackupRoot' -Message "Created backup root folder: $BackupFolder"
    } catch {
        Write-LogMessage -Level 'ERROR' -Id 'CreateBackupRoot' -Message "Unable to create backup folder at $BackupFolder. $_"
        exit 1
    }
}

$resolvedModelRoot = try { (Resolve-Path -LiteralPath $OllamaModelsDirectory).Path } catch { $null }
if (-not $resolvedModelRoot) {
    Write-LogMessage -Level 'WARN' -Id 'ResolveModelRoot' -Message "Ollama models directory '$OllamaModelsDirectory' does not exist. gguf files may not be copied."
    $resolvedModelRoot = $OllamaModelsDirectory
}

$regexModelRoot = [Regex]::Escape($resolvedModelRoot)

foreach ($model in $models) {
    Write-LogMessage -Id 'ProcessModel' -Message "Processing model: $model"

    try {
        $template = Invoke-OllamaCommand -Arguments @('show', '--template', $model)
    } catch {
        Write-LogMessage -Level 'WARN' -Id 'FetchTemplate' -Message "Skipping '$model' because the template could not be retrieved. $_"
        continue
    }

    if (-not $template) {
        Write-LogMessage -Level 'WARN' -Id 'FetchTemplate' -Message "Template for '$model' is empty; skipping this model."
        continue
    }

    $parameters = try { Invoke-OllamaCommand -Arguments @('show', '--parameters', $model) } catch { "" }
    $systemMessage = try { Invoke-OllamaCommand -Arguments @('show', '--system', $model) } catch { "" }
    $modelfileOutput = try { Invoke-OllamaCommand -Arguments @('show', '--modelfile', $model) } catch { "" }

    $sanitizedModelName = ConvertTo-SafeModelName -Name $model
    $modelFolder = Join-Path -Path $BackupFolder -ChildPath $sanitizedModelName

    if (Test-Path -LiteralPath $modelFolder) {
        Write-LogMessage -Level 'WARN' -Id 'ExistingBackup' -Message "Model '$sanitizedModelName' already exists in backup folder; skipping."
        continue
    }

    try {
        [void](New-Item -ItemType Directory -Path $modelFolder -Force)
        Write-LogMessage -Id 'CreateModelFolder' -Message "Created folder: $modelFolder"
    } catch {
        Write-LogMessage -Level 'WARN' -Id 'CreateModelFolder' -Message "Unable to create folder for '$sanitizedModelName'. $_"
        continue
    }

    $modelFileContent = "FROM $sanitizedModelName.gguf`n"
    $modelFileContent += 'TEMPLATE """' + "`n" + $template + "`n" + '"""' + "`n"

    if ($parameters) {
        foreach ($line in $parameters -split "`r?`n") {
            if (-not [string]::IsNullOrWhiteSpace($line)) {
                $modelFileContent += "PARAMETER $line`n"
            }
        }
    }

    if ($systemMessage) {
        $escapedSystemMessage = $systemMessage -replace '"', '\"'
    $modelFileContent += ('system "{0}"' + "`n" -f $escapedSystemMessage)
    }

    $modelFilePath = Join-Path -Path $modelFolder -ChildPath 'ModelFile'
    try {
        Set-Content -LiteralPath $modelFilePath -Value $modelFileContent -Encoding UTF8
        Write-LogMessage -Id 'WriteModelFile' -Message "Model file created: $modelFilePath"
    } catch {
        Write-LogMessage -Level 'WARN' -Id 'WriteModelFile' -Message "Failed to write model file for '$sanitizedModelName'. $_"
    }

    $pattern = 'FROM\s+({0}[^\s`"]+)' -f $regexModelRoot
    $ggufMatch = [Regex]::Match($modelfileOutput, $pattern)
    if ($ggufMatch.Success) {
        $sourceGgufPath = $ggufMatch.Groups[1].Value
        $targetGgufPath = Join-Path -Path $modelFolder -ChildPath ("{0}.gguf" -f $sanitizedModelName)

        if (Test-Path -LiteralPath $sourceGgufPath) {
            try {
                Copy-Item -LiteralPath $sourceGgufPath -Destination $targetGgufPath -Force
                Write-LogMessage -Id 'CopyGguf' -Message "Copied model file to: $targetGgufPath"
            } catch {
                Write-LogMessage -Level 'WARN' -Id 'CopyGguf' -Message "Failed to copy gguf file for '$sanitizedModelName'. $_"
            }
        } else {
            Write-LogMessage -Level 'WARN' -Id 'LocateGguf' -Message "Model file not found at: $sourceGgufPath"
        }
    } else {
        Write-LogMessage -Level 'WARN' -Id 'LocateGguf' -Message "Could not determine gguf file path for '$sanitizedModelName'."
    }
}
