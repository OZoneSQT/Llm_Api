[CmdletBinding()]
param(
	[string]$EnvironmentName = 'llm_api',
	[string]$Token,
	[switch]$AddToGitCredential
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '..\scripts\conda_helpers.ps1')

$trainingRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$envFile = Join-Path $trainingRoot '.env'

function Convert-SecureStringToPlain {
	param([System.Security.SecureString]$Secure)

	if (-not $Secure) { return '' }
	$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
	try {
		return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
	}
	finally {
		[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
	}
}


function Get-TokenFromEnvFile {
	param([string]$Path)
	if (-not (Test-Path $Path)) {
		return ''
	}
	foreach ($line in Get-Content -Path $Path) {
		$trim = $line.Trim()
		if (-not $trim -or $trim.StartsWith('#')) { continue }
		if ($trim -match '^HF_TOKEN\s*=\s*(.*)$') {
			return $Matches[1].Trim()
		}
	}
	return ''
}

function Write-TokenToEnvFile {
	param(
		[string]$Path,
		[string]$Value
	)

	$lines = @()
	if (Test-Path $Path) {
		$lines = Get-Content -Path $Path
	}
	$updated = $false
	for ($i = 0; $i -lt $lines.Count; $i++) {
		if ($lines[$i].TrimStart().StartsWith('HF_TOKEN=')) {
			$lines[$i] = "HF_TOKEN=$Value"
			$updated = $true
			break
		}
	}
	if (-not $updated) {
		$lines += "HF_TOKEN=$Value"
	}
	$lines | Out-File -FilePath $Path -Encoding UTF8
}

$tokenSource = ''
if ($PSBoundParameters.ContainsKey('Token') -and $Token) {
	$tokenSource = 'parameter'
}

if (-not $Token) {
	$envToken = $env:HF_TOKEN
	if ($envToken) {
		$Token = $envToken
		$tokenSource = 'env'
	}
}

if (-not $Token) {
	$Token = Get-TokenFromEnvFile -Path $envFile
	if ($Token) {
		$tokenSource = 'env-file'
	}
}

if (-not $Token) {
	$secure = Read-Host -Prompt 'Enter your Hugging Face access token' -AsSecureString
	$Token = Convert-SecureStringToPlain -Secure $secure
	$tokenSource = 'prompt'
}

if (-not $Token) {
	throw 'Hugging Face token is required.'
}

$argsList = @('run', '-n', $EnvironmentName, 'huggingface-cli', 'login', '--token', $Token)
if ($AddToGitCredential) {
	$argsList += '--add-to-git-credential'
}

Invoke-CondaCommand -Arguments $argsList

if ($tokenSource -eq 'prompt') {
	try {
		Write-TokenToEnvFile -Path $envFile -Value $Token
		Write-Host "Saved token to $envFile for future use." -ForegroundColor Yellow
	} catch {
		Write-Warning "Failed to persist token to $envFile: $_"
	}
}

# Best effort to clear token from memory
$Token = $null
