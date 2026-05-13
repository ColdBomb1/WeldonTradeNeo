@echo off
cd /d C:\WeldonTradeNeo
echo Starting WeldonTrader Neo...
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
