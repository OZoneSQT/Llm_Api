param(
    [string]$hostApi = "http://localhost:11434",
    [string]$model = "phi3:mini",
    [string]$prompt = "What is the capital of France?",
    [bool]$printRaw = $false, 
    [bool]$enableLogging = $true
)

#$hostApi = "http://localhost:11434"
#$hostApi = "http://truenas:11434"


#Compile:
#gcc -o OllamaAPI OllamaAPI.c -lcurl -lcjson


########################################
# Logging

$logFile = Join-Path $PSScriptRoot "llm.log"

function Write-Log {
    #Log
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    # SystemUsage
    $cpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples[0].CookedValue
    $mem = Get-WmiObject -Class Win32_OperatingSystem
    $totalMem = $mem.TotalVisibleMemorySize
    $freeMem = $mem.FreePhysicalMemory
    $usedMemPercent = [math]::Round((($totalMem - $freeMem) / $totalMem) * 100, 2)
    $cpuPercent = [math]::Round($cpu, 2)

    Add-Content -Path $logFile -Value "$timestamp $message, CPU-Load: $cpuPercent%, Memory-Load: $usedMemPercent%"
}


########################################
# Handle Api call and response

$body = @{
    model = $model
    prompt = $prompt
} | ConvertTo-Json

if($enableLogging) { Write-Log "Run: hostApi='$hostApi', model='$model', prompt='$prompt', printRaw=$printRaw" }

try {
    $request = Invoke-WebRequest -Uri "$hostApi/api/generate" -Method Post -Body $body -ContentType "application/json"
    if($enableLogging) { Write-Log "STATUS CODE: $($request.StatusCode)" }
    Write-Host "STATUS CODE: $($request.StatusCode)"
    $responseString = [System.Text.Encoding]::UTF8.GetString($request.Content)
    $fullResponse = ""
    $lines = $responseString -split "`n"
    foreach ($line in $lines) {
        if ($line.Trim()) {
            $json = $line | ConvertFrom-Json
            $fullResponse += $json.response
        }
    }
    if ($printRaw) {
        Write-Output $responseString
        if($enableLogging) { Write-Log "RAW RESPONSE: $($responseString.Substring(0, [Math]::Min(200, $responseString.Length)))..." }
    } else {
        Write-Output $fullResponse
        if($enableLogging) { Write-Log "FULL RESPONSE: $($fullResponse.Substring(0, [Math]::Min(200, $fullResponse.Length)))..." }  
    }
} catch {
    Write-Host "ERROR:"
    Write-Host $_
    if($enableLogging) { Write-Log "ERROR: $_" }
}

########################################
