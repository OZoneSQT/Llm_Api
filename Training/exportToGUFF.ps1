# Prompt for model name
$modelName = Read-Host "Enter the model folder name (e.g. my-model)"

# Output type guide (sorted by bit quantization)
Write-Host ""
Write-Host "Select output type for quantization (choose one):" -BackgroundColor Blue -ForegroundColor White
Write-Host ""
Write-Host "Quantized types" -BackgroundColor Blue -ForegroundColor White
Write-Host "  q4_0   - 4-bit quantization (smallest, lowest accuracy)" -ForegroundColor Blue
Write-Host "  q4_1   - Improved 4-bit quantization" -ForegroundColor Blue
Write-Host "  q4_K   - Blockwise 4-bit quantization (best accuracy for 4-bit, preferred if available)" -ForegroundColor Blue
Write-Host "  q5_0   - 5-bit quantization (medium size, better accuracy)" -ForegroundColor Blue
Write-Host "  q5_1   - Improved 5-bit quantization" -ForegroundColor Blue
Write-Host "  q5_K   - Blockwise 5-bit quantization (best accuracy for 5-bit, preferred if available)" -ForegroundColor Blue
Write-Host "  q8_0   - 8-bit quantization (high accuracy, large size)" -ForegroundColor Blue
Write-Host "  q8_1   - Improved 8-bit quantization" -ForegroundColor Blue
Write-Host "  q8_K   - Blockwise 8-bit quantization (best accuracy for 8-bit, preferred if available)" -ForegroundColor Blue
Write-Host ""
Write-Host "Float types" -BackgroundColor Blue -ForegroundColor White
Write-Host "  f16    - 16-bit float (high accuracy, smaller than f32)" -ForegroundColor Blue
Write-Host "  f32    - 32-bit float (highest accuracy, largest size)" -ForegroundColor Blue
Write-Host ""
Write-Host "Note: Not all output types are supported by every converter/model." -ForegroundColor Yellow
Write-Host ""

# Prompt for output type
$outType = Read-Host "Enter output type"

# Set output filename (include output type)
$outFile = "$modelName-$outType.gguf"

# Run the conversion
python convert.py "./$modelName" --outfile "$outFile" --outtype "$outType"