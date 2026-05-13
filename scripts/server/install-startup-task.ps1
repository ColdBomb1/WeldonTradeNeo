[CmdletBinding()]
param(
    [string]$ProjectDir = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$TaskName = "WeldonTradeNeo"
)

$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path -LiteralPath $ProjectDir).Path
$RunScript = Join-Path $ProjectDir "scripts\server\run-app.ps1"
$RunUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

if (-not (Test-Path -LiteralPath $RunScript)) {
    throw "Missing run script: $RunScript"
}

$ActionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$RunScript`" -ProjectDir `"$ProjectDir`""
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $ActionArgs
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $RunUser
$Principal = New-ScheduledTaskPrincipal -UserId $RunUser -LogonType Interactive -RunLevel Highest
$Settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Description "Runs WeldonTradeNeo FastAPI app from $ProjectDir when $RunUser logs on" `
    -Force | Out-Null

Write-Output "Scheduled task installed: $TaskName ($RunUser logon)"
