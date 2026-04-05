@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "SETUP_TARGET=%SCRIPT_DIR%\Setup_Advanced_Full.bat"
if not exist "%SETUP_TARGET%" set "SETUP_TARGET=%SCRIPT_DIR%\Setup_NewPC.bat"

if /I "%~f0"=="%SETUP_TARGET%" (
    echo ERROR: The beginner setup wrapper cannot target itself.
    pause
    exit /b 1
)

echo Launching the beginner setup profile...
echo   - CPU-only torch
echo   - no SeedVR2 install
echo   - no automatic DA360 checkpoint download
echo.
call "%SETUP_TARGET%" --cpu-only --skip-seedvr2 --skip-checkpoint %*
exit /b %errorlevel%