param(
    [Parameter(Mandatory = $true)]
    [string]$ModelName,

    [string]$BackupFolder = (Join-Path (Get-Location) "backup"),

    [string]$OllamaModelsDirectory = (Join-Path -Path (Join-Path -Path $env:USERPROFILE -ChildPath ".ollama") -ChildPath "models")
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

function ConvertTo-SafeModelName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $cleanName = $Name -replace ":latest$", ""
    return ($cleanName -replace '[<>:"/\|?*.]', "-")
}

if (-not $ModelName) {
    throw "ModelName is required."
}

$sanitizedModelName = ConvertTo-SafeModelName -Name $ModelName.Trim()

if (-not (Test-Path -LiteralPath $BackupFolder)) {
    try {
        [void](New-Item -ItemType Directory -Path $BackupFolder -Force)
        Write-LogMessage -Id 'CreateBackupRoot' -Message "Created backup root folder: $BackupFolder"
    } catch {
        $err = "Unable to create backup folder at $BackupFolder. $_"
        Write-LogMessage -Level 'ERROR' -Id 'CreateBackupRoot' -Message $err
        throw $err
    }
}

$resolvedModelRoot = try { (Resolve-Path -LiteralPath $OllamaModelsDirectory).Path } catch { $null }
if (-not $resolvedModelRoot) {
    Write-LogMessage -Level 'WARN' -Id 'ResolveModelRoot' -Message "Ollama models directory '$OllamaModelsDirectory' does not exist. gguf file may not be copied."
    $resolvedModelRoot = $OllamaModelsDirectory
}

$regexModelRoot = [Regex]::Escape($resolvedModelRoot)

try {
    $template = Invoke-OllamaCommand -Arguments @('show', '--template', $ModelName)
} catch {
    $err = "Unable to fetch template for '$ModelName'. $_"
    Write-LogMessage -Level 'ERROR' -Id 'FetchTemplate' -Message $err
    throw $err
}

if (-not $template) {
    $err = "Template for '$ModelName' is empty."
    Write-LogMessage -Level 'ERROR' -Id 'FetchTemplate' -Message $err
    throw $err
}

$parameters = try { Invoke-OllamaCommand -Arguments @('show', '--parameters', $ModelName) } catch { "" }
$systemMessage = try { Invoke-OllamaCommand -Arguments @('show', '--system', $ModelName) } catch { "" }
$modelfileOutput = try { Invoke-OllamaCommand -Arguments @('show', '--modelfile', $ModelName) } catch { "" }

$modelFolder = Join-Path -Path $BackupFolder -ChildPath $sanitizedModelName
if (-not (Test-Path -LiteralPath $modelFolder)) {
    [void](New-Item -ItemType Directory -Path $modelFolder -Force)
    Write-LogMessage -Id 'CreateModelFolder' -Message "Created folder: $modelFolder"
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
Set-Content -LiteralPath $modelFilePath -Value $modelFileContent -Encoding UTF8
Write-LogMessage -Id 'WriteModelFile' -Message "Model file created: $modelFilePath"

$pattern = 'FROM\s+({0}[^\s`"]+)' -f $regexModelRoot
$ggufMatch = [Regex]::Match($modelfileOutput, $pattern)
if ($ggufMatch.Success) {
    $sourceGgufPath = $ggufMatch.Groups[1].Value
    $targetGgufPath = Join-Path -Path $modelFolder -ChildPath ("{0}.gguf" -f $sanitizedModelName)

    if (Test-Path -LiteralPath $sourceGgufPath) {
        Copy-Item -LiteralPath $sourceGgufPath -Destination $targetGgufPath -Force
        Write-LogMessage -Id 'CopyGguf' -Message "Copied model file to: $targetGgufPath"
    } else {
        Write-LogMessage -Level 'WARN' -Id 'CopyGguf' -Message "Model file not found at: $sourceGgufPath"
    }
} else {
    Write-LogMessage -Level 'WARN' -Id 'CopyGguf' -Message "Could not determine gguf file path for '$ModelName'."
}
