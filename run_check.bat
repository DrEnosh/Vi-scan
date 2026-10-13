@echo off
REM Double-click: sets up .venv (first run), makes spoken test clips, runs tests and the live API check.
REM Everything is written to check_output.txt next to this file.
cd /d "%~dp0"
set LOG=check_output.txt
set PYTHONDONTWRITEBYTECODE=1
echo === Viscan setup and live check (%DATE% %TIME%) === > %LOG%
call :findpy
if not defined PY (
  echo Python not found. Install Python 3.11+ from python.org and tick "Add to PATH". >> %LOG%
  exit /b 1
)
echo Using: %PY% >> %LOG%
%PY% --version >> %LOG% 2>&1
if not exist .venv\Scripts\python.exe (
  echo Creating .venv and installing packages, this takes a minute... >> %LOG%
  %PY% -m venv .venv >> %LOG% 2>&1
  .venv\Scripts\python -m pip install -q -r requirements.txt -r requirements-providers.txt >> %LOG% 2>&1
)
echo. >> %LOG%
echo === Spoken test clips (Windows text-to-speech) === >> %LOG%
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\make_speech_samples.ps1 >> %LOG% 2>&1
echo. >> %LOG%
echo === Tests === >> %LOG%
.venv\Scripts\python -m pytest -q -p no:cacheprovider >> %LOG% 2>&1
echo. >> %LOG%
echo === Live check === >> %LOG%
.venv\Scripts\python scripts\check_llm.py --audio samples\speech_open_priya.wav --audio samples\speech_summarize.wav --audio samples\speech_dictation.wav >> %LOG% 2>&1
echo. >> %LOG%
echo === Live API tests (pytest -m live) === >> %LOG%
timeout /t 5 /nobreak >nul
set RUN_LIVE=1
.venv\Scripts\python -m pytest -q -p no:cacheprovider -m live >> %LOG% 2>&1
set RUN_LIVE=
echo. >> %LOG%
echo === Finished (%TIME%) === >> %LOG%
exit /b 0

:findpy
where py >nul 2>nul && (set "PY=py -3" & exit /b 0)
if exist "%USERPROFILE%\anaconda3\python.exe" (set "PY=%USERPROFILE%\anaconda3\python.exe" & exit /b 0)
if exist "%USERPROFILE%\miniconda3\python.exe" (set "PY=%USERPROFILE%\miniconda3\python.exe" & exit /b 0)
where python >nul 2>nul && (set "PY=python" & exit /b 0)
exit /b 0
