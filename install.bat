@echo off
rem install.bat - one-shot installer for Llama RPC Farm (Windows).
rem - finds a Python 3.11+ (py launcher, then PATH pythons)
rem - creates .venv in the project root
rem - installs everything from requirements.txt
rem Safe to re-run; idempotent.

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "REQ=%SCRIPT_DIR%requirements.txt"
set "MIN_MINOR=11"

echo ==^> Checking for Python 3.%MIN_MINOR%+ ...

set "PY="
py -3.%MIN_MINOR% -c "import sys" >nul 2>&1 && set "PY=py -3.%MIN_MINOR%"
if not defined PY (
  for %%v in (3.13 3.12 3.11) do (
    if not defined PY (
      py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
    )
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, %MIN_MINOR%) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo ERROR: no Python 3.%MIN_MINOR%+ found. Install from https://python.org (check "Add to PATH").
  exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo ==^> Creating virtualenv at %VENV_DIR%
  %PY% -m venv "%VENV_DIR%"
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
"%VENV_PY%" -m pip --version >nul 2>&1 || "%VENV_PY%" -m ensurepip >nul 2>&1

echo ==^> Installing requirements
"%VENV_PY%" -m pip install --upgrade pip -q
"%VENV_PY%" -m pip install -r "%REQ%"

echo ==^> Done.
echo.
echo Next steps:
echo   %SCRIPT_DIR%run.bat        # start the web UI
echo   http://127.0.0.1:8199
endlocal