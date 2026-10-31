@echo off
rem run.bat - start the Llama RPC Farm web UI (Windows).
rem - uses the project .venv (falls back to system python)
rem - frees port 8199 if a stale instance is bound
rem idempotent: safe to run repeatedly; it just (re)starts the app
rem stop with: taskkill /f /im python.exe (or the dashboard's Restart button)

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
set "PORT=8199"
set "LOG=%TEMP%\llama-web.log"

rem already running? report and exit
curl -s -m 2 "http://127.0.0.1:%PORT%/" 2>nul | findstr /i "Llama RPC Farm llama-swap" >nul
if %errorlevel%==0 (
  echo Already running at http://127.0.0.1:%PORT% ^(restart via dashboard^)
  exit /b 0
)

set "PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo ==^> No .venv found - run install.bat first ^(falling back to system python^)
  set "PY=python"
)

rem free a stale port: kill whatever owns :8199
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue; if ($c) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo ==^> Starting web UI ^(python: %PY%^)
rem cmd's `start` binds redirection to `start` itself — spawn via a
rem detached cmd /c so the python process actually owns the log file
start "" /b cmd /c ""%PY%" "%SCRIPT_DIR%src\app.py" > "%LOG%" 2>&1"

rem wait for it to come up (max ~10s)
for /l %%i in (1,1,10) do (
  timeout /t 1 /nobreak >nul
  curl -s -m 2 "http://127.0.0.1:%PORT%/" 2>nul | findstr /i "Llama RPC Farm llama-swap" >nul
  if !errorlevel!==0 (
    echo Web UI running at http://127.0.0.1:%PORT%
    echo Logs: %LOG%
    exit /b 0
  )
)
echo UI did not answer on :%PORT% yet - check %LOG%
endlocal
exit /b 1