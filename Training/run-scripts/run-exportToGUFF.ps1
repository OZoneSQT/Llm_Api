# Configuration

# Resolve repository root (one level up from Training folder)
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$possibleConversionPaths = @(
    (Join-Path $repoRoot "llama.cpp\convert_hf_to_gguf.py"),
    (Join-Path $repoRoot "lib\llama.cpp\convert_hf_to_gguf.py")
)

$conversionScriptPath = $possibleConversionPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $conversionScriptPath) {
    Write-Host "Error: Unable to locate convert_hf_to_gguf.py. Checked: $($possibleConversionPaths -join '; ')" -ForegroundColor Red
    exit 1
}

# Resolve configured model root (defaults to E:\AI\Models)
$modelRoot = if ([string]::IsNullOrWhiteSpace($env:HF_MODEL_ROOT)) { 'E:\AI\Models' } else { $env:HF_MODEL_ROOT }
try {
    $modelRoot = (Resolve-Path $modelRoot).Path
} catch {
    Write-Host "Warning: model root $modelRoot does not exist yet; it will be created." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null
}

$customModelRoot = Join-Path $modelRoot 'custom'
if (-not (Test-Path $customModelRoot)) {
    $customModelRoot = $modelRoot
}

$exportRoot = Join-Path $modelRoot 'exports'
if (-not (Test-Path $exportRoot)) {
    New-Item -ItemType Directory -Path $exportRoot -Force | Out-Null
}

###########################################################################
###########################################################################

# Prompt for model name
$modelName = Read-Host "Enter the model folder name (e.g. my-model)"

# Prompt for output type
Write-Host ""
Write-Host "Select output type for quantization (choose one):" -BackgroundColor Blue -ForegroundColor White
Write-Host ""
Write-Host "Supported Quantized types" -BackgroundColor Blue -ForegroundColor White
Write-Host "Documentation: https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py "
Write-Host "f32: Full 32-bit floating point precision. Highest accuracy, largest file size, slowest inference. Used for research or debugging."
Write-Host "f16: 16-bit floating point (half precision). Smaller file size, faster inference, minimal loss in accuracy for most models. Common for GPU inference."
Write-Host "bf16: Brain floating point 16-bit. Similar to f16, but with a different format that can improve stability and performance on some hardware (especially newer GPUs/TPUs)."
Write-Host "q8_0: 8-bit quantization. Reduces model size and speeds up inference, with some loss in accuracy. Good balance for CPU inference."
Write-Host "tq2_0: Ternary quantization (2-bit). Extreme compression, very small file size, but significant loss in accuracy. Used for experiments or very resource-constrained environments."
Write-Host "tq1_0: Ternary quantization (1-bit). Extreme compression, very small file size, but significant loss in accuracy. Used for experiments or very resource-constrained environments."
Write-Host "auto: Automatically selects the best quantization type based on your hardware and use case. Lets the conversion script decide."
Write-Host ""
$outType = Read-Host "Enter output type"

# Guard against invalid input
$validTypes = @("f32", "f16", "bf16", "q8_0", "tq2_0", "tq1_0", "auto")
if ($validTypes -notcontains $outType) {
    Write-Host "Error: Please choose a valid output type, options are: $($validTypes -join ', ')"
    $outType = Read-Host "Enter output type"
}

# Resolve source and target paths based on configured roots
$modelSrcPath = Join-Path $customModelRoot $modelName
if (-not (Test-Path $modelSrcPath)) {
    Write-Host "Error: Model folder not found at $modelSrcPath" -BackgroundColor Red -ForegroundColor White
    exit 1
}

$modelIndexPath = Join-Path $modelSrcPath 'model.safetensors.index.json'
if (-not (Test-Path $modelIndexPath)) {
    Write-Host "Error: Expected model.safetensors.index.json under $modelSrcPath" -BackgroundColor Red -ForegroundColor White
    exit 1
}

$targetModelRoot = Join-Path $exportRoot $modelName
if (-not (Test-Path $targetModelRoot)) {
    New-Item -ItemType Directory -Path $targetModelRoot -Force | Out-Null
}

# Extract parameters
$json = Get-Content $modelIndexPath | ConvertFrom-Json
$val = $json.metadata.total_parameters
if ($val -ge 1e9) {
    $val = "{0:N2}B" -f ($val / 1e9)
} elseif ($val -ge 1e6) {
    $val = "{0:N2}M" -f ($val / 1e6)
} elseif ($val -ge 1e3) {
    $val = "{0:N2}K" -f ($val / 1e3)
} else {
    $val = "$val"
}

# Set output filename (include output type)
$outFolder = "{0}_{1}_{2}" -f $modelName, $outType, $val
$outFile = "{0}.gguf" -f $outFolder
$modelOutputPath = $targetModelRoot
$outputPathFull = Join-Path $modelOutputPath $outFolder

if (Test-Path -Path $outputPathFull) {
    Remove-Item -Path $outputPathFull -Recurse -Force
}
New-Item -ItemType Directory -Path $outputPathFull | Out-Null
$outputPathFull = (Resolve-Path $outputPathFull).Path

# Full outfile path
$outfileFullPath = Join-Path $outputPathFull $outFile

# Generate GGUF model
# TODO: handle if model path is different than latest
# Call conversion script (use the resolved python on PATH). Ensure paths are quoted.
& python "$conversionScriptPath" "$(Resolve-Path $modelSrcPath)" --outfile "$outfileFullPath" --outtype $outType
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Model conversion failed." -BackgroundColor Red -ForegroundColor White
    if (Test-Path $outputPathFull) {
        Remove-Item -Path $outputPathFull -Recurse -Force
    }

    Write-Host "Conversion Script Path: $conversionScriptPath (Exists: $(Test-Path $conversionScriptPath))" -ForegroundColor Yellow
    Write-Host "Source Path: $modelSrcPath (Exists: $(Test-Path $modelSrcPath))" -ForegroundColor Yellow
    Write-Host "Output Path: $outputPathFull (Exists: $(Test-Path ($outputPathFull)))" -ForegroundColor Yellow
    Write-Host "Output type: $outType" -ForegroundColor Yellow
    exit 1
}


###########################################################################
###########################################################################
# Generate Modelfile
# Dokumentation: https://docs.ollama.com/modelfile#valid-parameters-and-values

$defaultSystemMessage = "You are a helpful assistant."
$modelfilePath = Join-Path $outputPathFull "modelfile"

$systemMessage = Read-Host "Enter system message / model instruction (or leave blank for none)"
if ([string]::IsNullOrWhiteSpace($systemMessage)) {
    $systemMessage = $defaultSystemMessage
}

# Escape double-quotes inside system message to avoid breaking the modelfile
# Replace " with \" so the Modelfile string remains valid
$systemMessageSafe = $systemMessage -replace '"', '\"'

# Create modelfile content
$modelfileContent = @"
FROM ./$outFile

# Additional parameters (optional)
PARAMETER mirostat 2.0
PARAMETER mirostat_eta 0.1
PARAMETER mirostat_tau 5.0
PARAMETER num_ctx 65535
PARAMETER repeat_last_n 64
PARAMETER repeat_penalty 1.1
PARAMETER temperature 0.6
PARAMETER seed 0
PARAMETER stop "STOP"
PARAMETER num_predict 42
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER min_p 0.05

SYSTEM "$systemMessageSafe"


# model_name: $modelName
# quantization: $outType
"@

Set-Content -Path $modelfilePath -Value $modelfileContent -Encoding UTF8


<#

### TEMPLATE Format:
TEMPLATE of the full prompt template to be passed into the model. It may include (optionally) a system message, a user’s message and the response from the model. 

Variable:	        Description
{{ .System }}	    The system message used to specify custom behavior.
{{ .Prompt }}	    The user prompt message.
{{ .Response }}	    The response from the model. When generating a response, text after this variable is omitted.

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

### MESSAGE Format:
The MESSAGE instruction allows you to specify a message history for the model to use when responding. Use multiple iterations of the MESSAGE command to build up a conversation which will guide the model to answer in a similar way.

Valid roles	        Description
system	            Alternate way of providing the SYSTEM message for the model.
user	            An example message of what the user could have asked.
assistant	        An example message of how the model should respond.

Example conversation:
MESSAGE user Is Toronto in Canada?
MESSAGE assistant yes
MESSAGE user Is Sacramento in Canada?
MESSAGE assistant no
MESSAGE user Is Ontario in Canada?
MESSAGE assistant yes

### Notes:
- the Modelfile is not case sensitive. In the examples, uppercase instructions are used to make it easier to distinguish it from arguments.
- Instructions can be in any order. In the examples, the FROM instruction is first to keep it easily readable.

#>

###########################################################################
###########################################################################

# Setup Script
 $setupfilePath = Join-Path $outputPathFull "setup.ps1"

# Create setup file content
$setupfileContent = @"
# Check if ollama is installed
if (-not (Get-Command "ollama" -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Ollama is not installed or not in PATH."
    exit 1
}

# Check if ollama is running
try {
    & ollama list > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Ollama is installed but not responding. Please start the Ollama service."
        exit 1
    }
} catch {
    Write-Host "Error: Unable to run 'ollama list'. Ensure Ollama is running and accessible."
    exit 1
}

# Create model in Ollama using the generated Modelfile in this folder
Set-Location -Path "$outputPathFull"
ollama create "$outFolder" -f .\modelfile

# Run the model to verify
ollama run "$outFolder"
"@

Set-Content -Path $setupfilePath -Value $setupfileContent -Encoding UTF8

###

# Uninstall Script
 $uninstallFilePath = Join-Path $outputPathFull "uninstall.ps1"

# Create uninstall file content
$uninstallFileContent = @"
& ollama rm "$outFolder"
"@

Set-Content -Path $uninstallFilePath -Value $uninstallFileContent -Encoding UTF8


###########################################################################
###########################################################################

Write-Host "GGUF model exported successfully to $(Join-Path $outputPathFull $outFile)" -ForegroundColor Green
Write-Host "Modelfile generated successfully to $modelfilePath" -ForegroundColor Green
Write-Host "Setup-file generated successfully to $setupfilePath" -ForegroundColor Green
Write-Host "Uninstall-file generated successfully to $uninstallFilePath" -ForegroundColor Green
