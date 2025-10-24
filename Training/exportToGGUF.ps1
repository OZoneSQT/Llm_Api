# Configuration
$conversionScriptPath = "..\lib\llama.cpp\convert_hf_to_gguf.py"
$modelOutputPath = "..\models"

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
    Write-Host "Error: Please choose a valid output type: $($validTypes -join ', ')"
    $outType = Read-Host "Enter output type"
}

# Set output filename (include output type)
$outFile = "$modelName" + "_" + "$outType" + ".gguf"
$outFolder = "$modelName" + "_" + "$outType" + "\"
$outputPath = "$modelOutputPath" + "\$modelName" + "\"

# Create output directory if it doesn't exist
if (-not (Test-Path -Path $outputPath)) {
    New-Item -ItemType Directory -Path $outputPath | Out-Null
}

# Generate GGUF model
# Example: python "C:\GitHub\Llm_Api\lib\llama.cpp\convert_hf_to_gguf.py" "C:\GitHub\Llm_Api\models\latest" --outfile "c:\GitHub\Llm_Api\models\MyModel_f16\MyModel_f16.gguf" --outtype f16
python $conversionScriptPath "$modelOutputPath\latest" --outfile "$outputPath\$outFolder\$outFile" --outtype $outType
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Model conversion failed." -BackgroundColor Red -ForegroundColor White
    Remove-Item -Path $modelOutputPath -Recurse -Force
    exit 1
}


###########################################################################
###########################################################################
# Generate Modelfile
# Dokumentation: https://docs.ollama.com/modelfile#valid-parameters-and-values

$defaultSystemMessage = "You are a helpful assistant."
$modelfilePath = "$outputPath\modelfile"

$systemMessage = Read-Host "Enter system message / model instruction (or leave blank for none)"
if ([string]::IsNullOrWhiteSpace($systemMessage)) {
    $systemMessage = $defaultSystemMessage
}

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
PARAMETER stop “STOP“
PARAMETER num_predict 42
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER min_p 0.05

SYSTEM """$systemMessage"""


# model_name: $modelName
# quantization: $outType
"@

Set-Content -Path $modelfilePath -Value $modelfileContent


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
$setupfilePath = "$outputPath\setup.ps1"

# Create setup file content
$setupfileContent = @"
# Check if ollama is installed
if (-not (Get-Command "ollama" -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Ollama is not installed or not in PATH."
    exit 1
}

# Check if ollama is running
try {
    $ollamaStatus = ollama list 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Ollama is installed but not running. Please start the Ollama service."
        exit 1
    }
} catch {
    Write-Host "Error: Unable to communicate with Ollama. Please ensure it is running."
    exit 1
}

# Create model in ollama
ollama create $outFolder -f ./Modelfile

# Run the model to verify
ollama run $outFolder
"@

Set-Content -Path $setupfilePath -Value $setupfileContent

###

# Uninstall Script
$uninstallFilePath = "$outputPath\uninstall.ps1"

# Create uninstall file content
$uninstallFileContent = @"
ollama rm $outFolder
"@

Set-Content -Path $uninstallFilePath -Value $uninstallFileContent


###########################################################################
###########################################################################

Write-Host "GGUF model exported successfully to $outputPath\$outFolder\$outFile" -ForegroundColor Green
Write-Host "Modelfile generated successfully to $modelfilePath" -ForegroundColor Green
Write-Host "Setup-file generated successfully to $setupfilePath" -ForegroundColor Green
Write-Host "Uninstall-file generated successfully to $uninstallFilePath" -ForegroundColor Green
