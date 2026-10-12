@echo off
REM Double-click to run. First run creates a local Python environment and installs the packages.
cd /d "%~dp0"

set "PY="
python --version >nul 2>&1 && set "PY=python"
if not defined PY (py -3 --version >nul 2>&1 && set "PY=py -3")
if not defined PY (
    echo Python 3.9 or newer is required.
    echo Download it from https://www.python.org/downloads/ and tick "Add Python to PATH" during setup.
    pause
    exit /b 1
)

if not exist ".venv\setup_done" (
    echo First run: setting up the environment. This takes a few minutes...
    if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet || goto :error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet || goto :error
    echo done> ".venv\setup_done"
)

".venv\Scripts\python.exe" main.py %*
echo.
echo Tables are in "reports\results" and figures in "reports\figures".
pause
exit /b 0

:error
echo.
echo Setup failed. Check your internet connection and try again.
pause
exit /b 1
