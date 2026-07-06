@echo off
REM Simple launcher: opens a console so you can see any error messages.
cd /d "%~dp0"

REM Pick a Python interpreter
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")

echo Checking dependencies...
%PY% -c "import pandas, numpy, sklearn, openpyxl" 2>nul || %PY% -m pip install --upgrade -r requirements.txt

echo Starting Pharma Pack Dimension Predictor...
start "" http://127.0.0.1:8000
%PY% app.py

echo.
echo The app has stopped. Press any key to close this window.
pause >nul
