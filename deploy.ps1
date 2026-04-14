# WeldonTrader Neo — Deployment Script for Windows Server
# Run this in PowerShell as Administrator on the remote server
# After copying the project to C:\WeldonTradeNeo

$ErrorActionPreference = "Stop"
$PROJECT_DIR = "C:\WeldonTradeNeo"
$PG_VERSION = "16"

Write-Host "=== WeldonTrader Neo Deployment ===" -ForegroundColor Cyan

# --- Step 1: Check Python ---
Write-Host "`n[1/6] Checking Python..." -ForegroundColor Yellow
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "ERROR: Python not found. Install Python 3.11+ first." -ForegroundColor Red
    exit 1
}
python --version

# --- Step 2: Install PostgreSQL ---
Write-Host "`n[2/6] Installing PostgreSQL..." -ForegroundColor Yellow
$pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue
if ($pgService) {
    Write-Host "PostgreSQL already installed: $($pgService.DisplayName)" -ForegroundColor Green
} else {
    Write-Host "Downloading PostgreSQL 16..."
    $pgInstaller = "$env:TEMP\postgresql-16-windows-x64.exe"
    if (-not (Test-Path $pgInstaller)) {
        Invoke-WebRequest -Uri "https://get.enterprisedb.com/postgresql/postgresql-16.6-1-windows-x64.exe" -OutFile $pgInstaller
    }
    Write-Host "Installing PostgreSQL (this may take a few minutes)..."
    Start-Process -Wait -FilePath $pgInstaller -ArgumentList `
        "--mode unattended",
        "--unattendedmodeui minimal",
        "--superpassword postgres",
        "--serverport 5432",
        "--prefix `"C:\Program Files\PostgreSQL\$PG_VERSION`""

    # Add to PATH
    $pgBin = "C:\Program Files\PostgreSQL\$PG_VERSION\bin"
    $env:Path += ";$pgBin"
    [Environment]::SetEnvironmentVariable("Path", $env:Path, "Machine")
    Write-Host "PostgreSQL installed." -ForegroundColor Green
}

# --- Step 3: Create Database ---
Write-Host "`n[3/6] Creating database..." -ForegroundColor Yellow
$env:PGPASSWORD = "postgres"
$pgBin = "C:\Program Files\PostgreSQL\$PG_VERSION\bin"
& "$pgBin\psql" -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'weldon_trader'" | Out-Null
if ($LASTEXITCODE -ne 0) {
    & "$pgBin\createdb" -U postgres weldon_trader
    Write-Host "Database 'weldon_trader' created." -ForegroundColor Green
} else {
    Write-Host "Database 'weldon_trader' already exists." -ForegroundColor Green
}

# --- Step 4: Install Python dependencies ---
Write-Host "`n[4/6] Installing Python packages..." -ForegroundColor Yellow
Set-Location $PROJECT_DIR
pip install -r requirements.txt
# Optional: MetaTrader5 (only works on Windows with MT5 installed)
pip install MetaTrader5 2>$null

# --- Step 5: Create data directory ---
Write-Host "`n[5/6] Setting up data directory..." -ForegroundColor Yellow
if (-not (Test-Path "$PROJECT_DIR\data")) {
    New-Item -ItemType Directory -Path "$PROJECT_DIR\data" | Out-Null
}

# --- Step 6: Create Windows service / startup script ---
Write-Host "`n[6/6] Creating startup script..." -ForegroundColor Yellow

$startScript = @"
@echo off
cd /d $PROJECT_DIR
echo Starting WeldonTrader Neo...
python -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
"@
$startScript | Out-File -FilePath "$PROJECT_DIR\start.bat" -Encoding ASCII

$serviceScript = @"
@echo off
cd /d $PROJECT_DIR
echo Starting WeldonTrader Neo as background service...
start /B python -m uvicorn main:app --host 0.0.0.0 --port 8000 > logs\server.log 2>&1
echo Server started. Check logs\server.log for output.
"@
if (-not (Test-Path "$PROJECT_DIR\logs")) {
    New-Item -ItemType Directory -Path "$PROJECT_DIR\logs" | Out-Null
}
$serviceScript | Out-File -FilePath "$PROJECT_DIR\start-background.bat" -Encoding ASCII

# Create a NSSM service installer (if NSSM is available)
$nssmScript = @"
@echo off
REM Install as Windows Service using NSSM
REM Download NSSM from https://nssm.cc/download if not installed
nssm install WeldonTrader python -m uvicorn main:app --host 0.0.0.0 --port 8000
nssm set WeldonTrader AppDirectory $PROJECT_DIR
nssm set WeldonTrader AppStdout $PROJECT_DIR\logs\service-stdout.log
nssm set WeldonTrader AppStderr $PROJECT_DIR\logs\service-stderr.log
nssm set WeldonTrader Start SERVICE_AUTO_START
nssm start WeldonTrader
echo WeldonTrader service installed and started.
"@
$nssmScript | Out-File -FilePath "$PROJECT_DIR\install-service.bat" -Encoding ASCII

Write-Host "`n=== Deployment Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To start manually:  .\start.bat" -ForegroundColor White
Write-Host "To start in background:  .\start-background.bat" -ForegroundColor White
Write-Host "To install as service:  .\install-service.bat (requires NSSM)" -ForegroundColor White
Write-Host ""
Write-Host "Dashboard: http://localhost:8000" -ForegroundColor Cyan
Write-Host "From outside: http://15.204.224.153:8000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Make sure Windows Firewall allows port 8000 inbound." -ForegroundColor Yellow
