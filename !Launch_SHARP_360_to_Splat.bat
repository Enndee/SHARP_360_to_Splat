@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

set "GUI_SCRIPT=%SCRIPT_DIR%\Easy_360_SHARP_GUI.py"
set "SHARP_PY=%SCRIPT_DIR%\.venv\Scripts\python.exe"

if not exist "%GUI_SCRIPT%" (
    echo ERROR: Could not find Easy_360_SHARP_GUI.py next to this launcher.
    pause
    exit /b 1
)

if not exist "%SHARP_PY%" (
    echo ERROR: Could not find the local .venv Python interpreter.
    echo Run Setup_NewPC.bat first, then try again.
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo Starting SHARP_360_to_Splat...
"%SHARP_PY%" "%GUI_SCRIPT%" %*
if errorlevel 1 (
    echo.
    echo The GUI exited with an error.
    pause
    exit /b 1
)

exit /b 0