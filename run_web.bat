@echo off
echo Starting CaroNet Web Server...
cd /d "%~dp0backend"
uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
