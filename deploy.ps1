param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$Script = Join-Path $PSScriptRoot "scripts\deploy.ps1"
& $Script @RemainingArgs
exit $LASTEXITCODE
