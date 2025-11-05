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

function Invoke-OllamaRun {
	param([Parameter(Mandatory)][string]$Model)

	Write-LogMessage -Id 'InstallModel' -Message "Ensuring model '$Model' is available."
	$output = & ollama run $Model 2>&1
	$exit = $LASTEXITCODE

	if ($exit -ne 0) {
		$msg = "ollama run $Model failed: $($output -join ' | ')"
		Write-LogMessage -Level 'ERROR' -Id 'InstallModel' -Message $msg
	} else {
		Write-LogMessage -Id 'InstallModel' -Message "ollama run completed for $Model"
	}
}

# Deepseek
Write-LogMessage -Id 'InstallGroup' -Message 'Installing Deepseek models'
Invoke-OllamaRun -Model 'deepseek-r1:1.5b'
Invoke-OllamaRun -Model 'deepseek-r1:8b'

# Google models
Write-LogMessage -Id 'InstallGroup' -Message 'Installing Google models'
Invoke-OllamaRun -Model 'gemma2:latest'

# Microsoft models
Write-LogMessage -Id 'InstallGroup' -Message 'Installing Microsoft models'
Invoke-OllamaRun -Model 'phi4:latest'

# Apache models
Write-LogMessage -Id 'InstallGroup' -Message 'Installing Apache models'
Invoke-OllamaRun -Model 'mistral:latest'

# open-source models
Write-LogMessage -Id 'InstallGroup' -Message 'Installing open-source models'
Invoke-OllamaRun -Model 'llama3:latest'
Invoke-OllamaRun -Model 'codellama:13b'
Invoke-OllamaRun -Model 'dolphin-llama3:8b'

# uncensored open-source model
Write-LogMessage -Id 'InstallGroup' -Message 'Installing uncensored open-source models'
Invoke-OllamaRun -Model 'dolphin-phi:latest'
Invoke-OllamaRun -Model 'dolphin-mistral:latest'
