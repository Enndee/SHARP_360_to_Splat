@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "GUI_SCRIPT=%SCRIPT_DIR%Easy_360_SHARP_GUI.py"
if not exist "%GUI_SCRIPT%" (
    echo ERROR: Could not find Easy_360_SHARP_GUI.py next to this launcher.
    pause
    exit /b 1
)

set "SHARP_PY="
for %%B in (
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\Anaconda3"
    "%USERPROFILE%\Miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "%LOCALAPPDATA%\Anaconda3"
    "%LOCALAPPDATA%\Miniconda3"
    "%APPDATA%\anaconda3"
    "%APPDATA%\miniconda3"
    "%ProgramData%\anaconda3"
    "%ProgramData%\miniconda3"
    "%ProgramData%\Anaconda3"
    "%ProgramData%\Miniconda3"
    "C:\anaconda3"
    "C:\miniconda3"
    "D:\anaconda3"
    "D:\miniconda3"
) do (
    if exist "%%~B\envs\sharp\python.exe" (
        set "SHARP_PY=%%~B\envs\sharp\python.exe"
        goto :launch_gui
    )
)

for /f "usebackq delims=" %%B in (`conda info --base 2^>nul`) do (
    if exist "%%~B\envs\sharp\python.exe" (
        set "SHARP_PY=%%~B\envs\sharp\python.exe"
        goto :launch_gui
    )
)

echo ERROR: Could not find the 'sharp' conda environment.
echo Run Setup_NewPC.bat first, then try again.
pause
exit /b 1

:launch_gui
echo Starting SHARP_360_to_Splat...
"%SHARP_PY%" "%GUI_SCRIPT%" %*
if errorlevel 1 (
    echo.
    echo The GUI exited with an error.
    pause
    exit /b 1
)

exit /b 0