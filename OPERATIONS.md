# WeldonTradeNeo Operations

This checkout is the Codex-managed source tree for the remote runtime at
`weldontrade:C:\WeldonTradeNeo`.

## Daily Workflow

```powershell
# from C:\WeldonTraderHub
git status
.\scripts\remote.ps1 -Action status
.\scripts\remote.ps1 -Action logs
```

Make code changes locally, run checks, commit, then deploy:

```powershell
.\.venv\Scripts\python.exe -m compileall .
git add .
git commit -m "Describe the change"
.\scripts\remote.ps1 -Action deploy
```

The deploy script uploads the current Git commit over SSH, creates a timestamped
server backup in `C:\WeldonTradeNeo_backups`, preserves runtime files, installs
Python dependencies into the server `venv`, installs/updates the startup task,
and restarts the app.

## Remote Runtime

- Project: `C:\WeldonTradeNeo`
- SSH alias: `weldontrade`
- Scheduled task: `WeldonTradeNeo` running under the interactive Administrator account at logon
- App URL on server: `http://127.0.0.1:8000`
- External URL: `http://15.204.224.153:8000`
- Uvicorn task log: `C:\WeldonTradeNeo\logs\uvicorn.log`
- Uvicorn stdout/stderr: `C:\WeldonTradeNeo\logs\uvicorn.stdout.log`, `C:\WeldonTradeNeo\logs\uvicorn.stderr.log`
- App log: `C:\WeldonTradeNeo\logs\app.log`

## Runtime Files Preserved During Deploy

These stay on the server and are not deployed from source:

- `data/`
- `logs/`
- `config/keys/`
- `venv/`
- `.env`
- `apikey.txt`

## Useful Commands

```powershell
.\scripts\remote.ps1 -Action deploy
.\scripts\remote.ps1 -Action restart
.\scripts\remote.ps1 -Action stop
.\scripts\remote.ps1 -Action start
.\scripts\remote.ps1 -Action status
.\scripts\remote.ps1 -Action logs
.\scripts\remote.ps1 -Action shell
```
