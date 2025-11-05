param (
    [string]$LogLevel,    # Apply appropriate log levels (INFO, WARN, ERROR)
    [string]$Source,      # Source script
    [string]$Id,    	  # Function as identifier
    [string]$Message      # Payload, can be comma separated for more parameters
)

# How to:
# & .\Logger.ps1 -LogLevel "INFO" -Source "MyScript" -Id "MyFunction" -Message "I did something"

# Format log entry
$timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

$logFilePath = "$PSScriptRoot\log.csv"
	
# Create log file with headers if it doesn't exist
if (-Not (Test-Path -Path $logFilePath)) {
	"Timestamp,LogLevel,Source,Id,Message" | Out-File -FilePath $logFilePath -Encoding UTF8
}
	
$logEntry = "$timestamp,$LogLevel,$Source,$Id,$Message"
	
# Append to log file
Add-Content -Path $logFilePath -Value $logEntry

# Display log status
if ($LogLevel -eq "ERROR") {
    Write-Host "ERROR Logged: $logEntry to $logFilePath" -BackgroundColor "Red" -ForegroundColor "White"
}
elseif ($LogLevel -eq "WARN") {
    Write-Host "WARNING Logged: $logEntry to $logFilePath" -BackgroundColor "Yellow" -ForegroundColor "Black"
}
else {
    Write-Host "Logged: $logEntry to $logFilePath" -BackgroundColor "Blue" -ForegroundColor "White"
}
