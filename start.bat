@echo off
REM One-click start on Windows: creates a venv on first run, then serves http://localhost:8000
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
call :findpy
if not defined PY (
  echo Python not found. Install Python 3.11+ from python.org and tick "Add to PATH".
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment with %PY% ...
  %PY% -m venv .venv || goto :error
  .venv\Scripts\python -m pip install -r requirements.txt -r requirements-providers.txt || goto :error
)
if not exist .env (
  copy .env.example .env >nul
  echo Created .env from .env.example. Add your GROQ_API_KEY there to enable the LLM and Whisper.
)
start "" http://localhost:8000
.venv\Scripts\python -m uvicorn app.main:app --port 8000
goto :eof

:error
echo Setup failed. Check that Python 3.11+ is installed.
pause
exit /b 1

:findpy
where py >nul 2>nul && (set "PY=py -3" & exit /b 0)
if exist "%USERPROFILE%\anaconda3\python.exe" (set "PY=%USERPROFILE%\anaconda3\python.exe" & exit /b 0)
if exist "%USERPROFILE%\miniconda3\python.exe" (set "PY=%USERPROFILE%\miniconda3\python.exe" & exit /b 0)
where python >nul 2>nul && (set "PY=python" & exit /b 0)
exit /b 0
