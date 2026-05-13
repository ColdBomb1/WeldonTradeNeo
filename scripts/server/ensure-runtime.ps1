[CmdletBinding()]
param(
    [string]$ProjectDir = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
)

$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path -LiteralPath $ProjectDir).Path
$VenvDir = Join-Path $ProjectDir "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $ProjectDir "requirements.txt"

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "data") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "logs") | Out-Null

function Test-VenvPython {
    param([string]$PythonPath)
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        return $false
    }

    try {
        & $PythonPath -c "import sys; print(sys.executable)" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

if (-not (Test-VenvPython $VenvPython)) {
    Write-Host "Creating Python virtual environment at $VenvDir"
    $SystemPython = Get-Command python -ErrorAction Stop
    & $SystemPython.Source -m venv --clear $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment at $VenvDir"
    }
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip"
}

& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install requirements from $Requirements"
}

# MetaTrader5 is optional in this app and can lag new Python releases. Keep the
# runtime usable even when the package has no wheel for the installed Python.
& $VenvPython -m pip install MetaTrader5
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Optional MetaTrader5 package install failed; continuing without it."
}

Write-Host "Runtime ready: $VenvPython"
