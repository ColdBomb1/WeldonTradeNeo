[CmdletBinding()]
param(
    [string]$SshHost = "weldontrade",
    [string]$ProjectDir = "C:\WeldonTradeNeo",
    [string]$TaskName = "WeldonTradeNeo",
    [switch]$AllowDirty,
    [switch]$SkipRestart
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

function Invoke-RemotePowerShell {
    param(
        [Parameter(Mandatory)]
        [string]$Command
    )

    $Encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Command))
    & ssh $SshHost "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $Encoded"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote PowerShell command failed with exit code $LASTEXITCODE"
    }
}

if (-not $AllowDirty) {
    $Dirty = git status --porcelain
    if ($Dirty) {
        throw "Working tree is dirty. Commit or stash changes before deploy, or rerun with -AllowDirty."
    }
}

$Commit = (git rev-parse --short HEAD).Trim()
$Archive = Join-Path $env:TEMP "weldontrade-$Commit.tar"
$RemoteArchive = "C:\Windows\Temp\weldontrade-deploy-$Commit.tar"

Remove-Item -LiteralPath $Archive -ErrorAction SilentlyContinue
git archive --format=tar -o $Archive HEAD
if ($LASTEXITCODE -ne 0) {
    throw "git archive failed"
}

& scp $Archive "${SshHost}:$($RemoteArchive -replace '\\','/')"
if ($LASTEXITCODE -ne 0) {
    throw "scp upload failed"
}

$RestartLiteral = if ($SkipRestart) { '$true' } else { '$false' }
$RemoteCommand = @"
`$ErrorActionPreference = "Stop"
`$ProgressPreference = "SilentlyContinue"
`$ProjectDir = "$ProjectDir"
`$TaskName = "$TaskName"
`$Archive = "$RemoteArchive"
`$SkipRestart = $RestartLiteral
`$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
`$BackupRoot = "C:\WeldonTradeNeo_backups"
`$BackupDir = Join-Path `$BackupRoot "deploy_`$Stamp"
`$StagingDir = Join-Path "C:\Windows\Temp" "weldontrade_deploy_`$Stamp"

New-Item -ItemType Directory -Force -Path `$BackupDir | Out-Null
if (Test-Path -LiteralPath `$ProjectDir) {
    robocopy `$ProjectDir `$BackupDir /MIR /XD venv __pycache__ .pytest_cache .mypy_cache .ruff_cache /XF *.pyc *.pyo | Out-Null
    if (`$LASTEXITCODE -gt 7) { throw "Backup robocopy failed with exit code `$LASTEXITCODE" }
}

Remove-Item -LiteralPath `$StagingDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path `$StagingDir | Out-Null
tar -xf `$Archive -C `$StagingDir
if (`$LASTEXITCODE -ne 0) { throw "Failed to extract deploy archive" }

`$ExistingTask = Get-ScheduledTask -TaskName `$TaskName -ErrorAction SilentlyContinue
if (`$ExistingTask) {
    Stop-ScheduledTask -TaskName `$TaskName -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Get-CimInstance Win32_Process |
    Where-Object {
        `$_.ExecutablePath -like "`$ProjectDir\venv\Scripts\python.exe" -and
        `$_.CommandLine -match "uvicorn" -and
        `$_.CommandLine -match "main:app"
    } |
    ForEach-Object {
        Stop-Process -Id `$_.ProcessId -Force -ErrorAction SilentlyContinue
    }

New-Item -ItemType Directory -Force -Path `$ProjectDir | Out-Null
robocopy `$StagingDir `$ProjectDir /MIR /XD .git .vscode venv data logs __pycache__ .pytest_cache .mypy_cache .ruff_cache "config\keys" /XF apikey.txt .env *.pyc *.pyo | Out-Null
if (`$LASTEXITCODE -gt 7) { throw "Deploy robocopy failed with exit code `$LASTEXITCODE" }

& (Join-Path `$ProjectDir "scripts\server\ensure-runtime.ps1") -ProjectDir `$ProjectDir
& (Join-Path `$ProjectDir "scripts\server\install-startup-task.ps1") -ProjectDir `$ProjectDir -TaskName `$TaskName

if (-not `$SkipRestart) {
    Start-ScheduledTask -TaskName `$TaskName
    Start-Sleep -Seconds 5
}

`$Task = Get-ScheduledTask -TaskName `$TaskName -ErrorAction SilentlyContinue
`$Port = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -First 1
Write-Output "DEPLOYED_COMMIT=$Commit"
Write-Output "BACKUP_DIR=`$BackupDir"
if (`$Task) { Write-Output "TASK_STATE=`$(`$Task.State)" }
if (`$Port) { Write-Output "PORT_8000=`$(`$Port.State)" }
"@

Invoke-RemotePowerShell -Command $RemoteCommand
Remove-Item -LiteralPath $Archive -ErrorAction SilentlyContinue
