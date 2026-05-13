[CmdletBinding()]
param(
    [string]$ProjectDir = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path -LiteralPath $ProjectDir).Path
$VenvPython = Join-Path $ProjectDir "venv\Scripts\python.exe"
$LogDir = Join-Path $ProjectDir "logs"
$LogFile = Join-Path $LogDir "uvicorn.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $ProjectDir

if (-not (Test-Path -LiteralPath $VenvPython)) {
    & (Join-Path $ProjectDir "scripts\server\ensure-runtime.ps1") -ProjectDir $ProjectDir
}

$env:PYTHONUNBUFFERED = "1"
$env:WELDON_ENV = "production"

Add-Content -Path $LogFile -Value ""
Add-Content -Path $LogFile -Value "=== WeldonTradeNeo start $(Get-Date -Format o) ==="

& $VenvPython -m uvicorn main:app --host $HostAddress --port $Port *>> $LogFile
exit $LASTEXITCODE
