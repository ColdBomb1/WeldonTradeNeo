[CmdletBinding()]
param(
    [ValidateSet("deploy", "start", "stop", "restart", "status", "logs", "shell", "bootstrap")]
    [string]$Action = "status",
    [string]$SshHost = "weldontrade",
    [string]$ProjectDir = "C:\WeldonTradeNeo",
    [string]$TaskName = "WeldonTradeNeo",
    [int]$LogTail = 120
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

function Invoke-RemotePowerShell {
    param(
        [Parameter(Mandatory)]
        [string]$Command
    )

    $Command = "`$ProgressPreference = `"SilentlyContinue`"`n" + $Command
    $Encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Command))
    & ssh $SshHost "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $Encoded"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote PowerShell command failed with exit code $LASTEXITCODE"
    }
}

switch ($Action) {
    "deploy" {
        & (Join-Path $RepoRoot "scripts\deploy.ps1") -SshHost $SshHost -ProjectDir $ProjectDir -TaskName $TaskName
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    "shell" {
        & ssh $SshHost
        exit $LASTEXITCODE
    }
    "bootstrap" {
        $Command = @"
`$ErrorActionPreference = "Stop"
& "$ProjectDir\scripts\server\ensure-runtime.ps1" -ProjectDir "$ProjectDir"
& "$ProjectDir\scripts\server\install-startup-task.ps1" -ProjectDir "$ProjectDir" -TaskName "$TaskName"
"@
        Invoke-RemotePowerShell -Command $Command
    }
    "start" {
        Invoke-RemotePowerShell -Command "Start-ScheduledTask -TaskName `"$TaskName`""
    }
    "stop" {
        $Command = @"
Stop-ScheduledTask -TaskName "$TaskName" -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Get-CimInstance Win32_Process |
    Where-Object {
        `$_.ExecutablePath -like "$ProjectDir\venv\Scripts\python.exe" -and
        `$_.CommandLine -match "uvicorn" -and
        `$_.CommandLine -match "main:app"
    } |
    ForEach-Object {
        Stop-Process -Id `$_.ProcessId -Force -ErrorAction SilentlyContinue
    }
"@
        Invoke-RemotePowerShell -Command $Command
    }
    "restart" {
        & (Join-Path $RepoRoot "scripts\remote.ps1") -Action stop -SshHost $SshHost -ProjectDir $ProjectDir -TaskName $TaskName
        & (Join-Path $RepoRoot "scripts\remote.ps1") -Action start -SshHost $SshHost -ProjectDir $ProjectDir -TaskName $TaskName
    }
    "status" {
        $Command = @"
`$Task = Get-ScheduledTask -TaskName "$TaskName" -ErrorAction SilentlyContinue
if (`$Task) {
    `$Info = Get-ScheduledTaskInfo -TaskName "$TaskName"
    [pscustomobject]@{
        TaskName = `$Task.TaskName
        State = `$Task.State
        LastRunTime = `$Info.LastRunTime
        LastTaskResult = `$Info.LastTaskResult
        NextRunTime = `$Info.NextRunTime
    } | Format-List
} else {
    Write-Output "Scheduled task not installed: $TaskName"
}

Write-Output "--- port 8000 ---"
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress,LocalPort,State,OwningProcess |
    Format-Table -AutoSize

Write-Output "--- health ---"
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/healthz" -TimeoutSec 5
} catch {
    Write-Output `$_.Exception.Message
}
"@
        Invoke-RemotePowerShell -Command $Command
    }
    "logs" {
        $Command = @"
`$Uvicorn = Join-Path "$ProjectDir" "logs\uvicorn.log"
`$Stdout = Join-Path "$ProjectDir" "logs\uvicorn.stdout.log"
`$Stderr = Join-Path "$ProjectDir" "logs\uvicorn.stderr.log"
`$App = Join-Path "$ProjectDir" "logs\app.log"
if (Test-Path -LiteralPath `$Uvicorn) {
    Write-Output "--- uvicorn.log ---"
    Get-Content -LiteralPath `$Uvicorn -Tail $LogTail
}
if (Test-Path -LiteralPath `$Stdout) {
    Write-Output "--- uvicorn.stdout.log ---"
    Get-Content -LiteralPath `$Stdout -Tail $LogTail
}
if (Test-Path -LiteralPath `$Stderr) {
    Write-Output "--- uvicorn.stderr.log ---"
    Get-Content -LiteralPath `$Stderr -Tail $LogTail
}
if (Test-Path -LiteralPath `$App) {
    Write-Output "--- app.log ---"
    Get-Content -LiteralPath `$App -Tail $LogTail
}
"@
        Invoke-RemotePowerShell -Command $Command
    }
}
