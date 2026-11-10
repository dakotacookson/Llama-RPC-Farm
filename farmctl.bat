@echo off
rem farmctl — CLI client for the running Llama RPC Farm app (:8199).
setlocal
set "ROOT=%~dp0"
if exist "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" "%ROOT%src\farmctl.py" %*
) else (
  python "%ROOT%src\farmctl.py" %*
)
exit /b %ERRORLEVEL%
