@echo off
setlocal EnableDelayedExpansion

set "APP_NAME=SHARP_360_to_Splat"
set "RELEASE_VERSION=1.5.1"
set "PACKAGE_DIR=release_pkg\%APP_NAME%_v%RELEASE_VERSION%_Windows"
set "ZIP_PATH=release_pkg\%APP_NAME%_v%RELEASE_VERSION%_Windows.zip"
set "VIEWER_SOURCE=splatapult\build\Release"

echo =====================================================
echo  SHARP_360_to_Splat  --  Build EXE
echo =====================================================
echo.

REM ── Locate the 'sharp' conda environment ─────────────────────────────────
set "SHARP_PY="
for %%B in (
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%LOCALAPPDATA%\miniconda3"
) do (
    if exist "%%~B\envs\sharp\python.exe" (
        set "SHARP_PY=%%~B\envs\sharp\python.exe"
        set "SHARP_SCRIPTS=%%~B\envs\sharp\Scripts"
        goto :found_env
    )
)

echo ERROR: Could not find the 'sharp' conda environment.
echo        Run Setup_NewPC.bat first to create it.
pause
exit /b 1

:found_env
echo [1/5] Found sharp env: %SHARP_PY%
echo.

REM ── Install PyInstaller into the sharp env ────────────────────────────────
echo [2/5] Installing PyInstaller...
"%SHARP_PY%" -m pip install pyinstaller --quiet
if errorlevel 1 (
    echo ERROR: pip install pyinstaller failed.
    pause
    exit /b 1
)

REM ── Locate Tcl/Tk libraries (must match the version _tkinter.pyd was built with)
set "CONDA_BASE="
for %%B in (
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%LOCALAPPDATA%\miniconda3"
) do (
    if exist "%%~B\envs\sharp\python.exe" (
        set "CONDA_BASE=%%~B"
        goto :found_conda
    )
)
:found_conda
set "TCL_LIBRARY=%CONDA_BASE%\envs\sharp\Library\lib\tcl8.6"
set "TK_LIBRARY=%CONDA_BASE%\envs\sharp\Library\lib\tk8.6"
echo       TCL_LIBRARY=%TCL_LIBRARY%
echo       TK_LIBRARY=%TK_LIBRARY%

REM Put the sharp env's Library\bin FIRST on PATH so PyInstaller picks up the
REM correct tcl86t.dll (8.6.15) rather than the base conda's (8.6.14).
set "PATH=%CONDA_BASE%\envs\sharp\Library\bin;%PATH%"

REM ── Build the exe ─────────────────────────────────────────────────────────
echo.
echo [3/5] Building %APP_NAME%.exe ...
"%SHARP_SCRIPTS%\pyinstaller.exe" ^
    --onefile ^
    --windowed ^
    --name "%APP_NAME%" ^
    --hidden-import plyfile ^
    --hidden-import numpy ^
     --hidden-import PIL ^
     --hidden-import PIL.Image ^
     --hidden-import PIL.ImageOps ^
     --hidden-import PIL.ImageTk ^
    --collect-all plyfile ^
     --collect-all PIL ^
    --add-data "%TCL_LIBRARY%;tcl" ^
    --add-data "%TK_LIBRARY%;tk" ^
    "Easy_360_SHARP_GUI.py"

if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)

REM ── Assemble distribution folder ──────────────────────────────────────────
echo.
echo [4/5] Assembling release package...

if exist "release_pkg" rmdir /S /Q "release_pkg"
mkdir "%PACKAGE_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create release package folder.
    pause
    exit /b 1
)

copy /Y "dist\%APP_NAME%.exe" "%PACKAGE_DIR%\" >nul
copy /Y "Launch_SHARP_360_to_Splat.bat" "%PACKAGE_DIR%\" >nul
copy /Y "Setup_NewPC.bat" "%PACKAGE_DIR%\" >nul
copy /Y "gsbox.exe" "%PACKAGE_DIR%\" >nul

if not exist "%PACKAGE_DIR%\third_party\DA360" mkdir "%PACKAGE_DIR%\third_party\DA360"
xcopy /E /I /Y "third_party\DA360\*" "%PACKAGE_DIR%\third_party\DA360\" >nul

if exist "checkpoints\DA360_large.pth" (
    if not exist "%PACKAGE_DIR%\checkpoints" mkdir "%PACKAGE_DIR%\checkpoints"
    copy /Y "checkpoints\DA360_large.pth" "%PACKAGE_DIR%\checkpoints\" >nul
) else (
    echo WARNING: DA360_large.pth was not found under checkpoints\.
    echo          The packaged build will require the user to browse to a DA360 checkpoint manually.
)

if not exist "%PACKAGE_DIR%\splatapult\build\Release" mkdir "%PACKAGE_DIR%\splatapult\build\Release"
if exist "%VIEWER_SOURCE%\*" (
    xcopy /E /I /Y "%VIEWER_SOURCE%\*" "%PACKAGE_DIR%\splatapult\build\Release\" >nul
) else (
    echo WARNING: Could not find splatapult viewer files in %VIEWER_SOURCE%.
    echo          The v%RELEASE_VERSION% package will be built without the bundled viewer payload.
    rmdir /S /Q "%PACKAGE_DIR%\splatapult" >nul 2>&1
)

if exist "easy_360_sharp_gui_settings.json" del /Q "%PACKAGE_DIR%\easy_360_sharp_gui_settings.json" >nul 2>&1

REM Write a quick README
(
    echo SHARP_360_to_Splat
    echo ====================
    echo.
    echo REQUIREMENTS FOR A NEW PC
    echo   1. Windows 11, NVIDIA GPU (RTX 20/30/40/50 series with CUDA)
    echo   2. Anaconda or Miniconda - https://www.anaconda.com/download
    echo   3. Git for Windows      - https://git-scm.com/download/win
    echo.
    echo SETUP (first time only^)
    echo   Run Setup_NewPC.bat  -- installs the SHARP conda environment.
    echo.
    echo USAGE
    echo   - Double-click SHARP_360_to_Splat.exe and use the file picker, OR
    echo   - Right-click any JPEG/PNG -^> Send To -^> SHARP_360_to_Splat
    echo     (Setup_NewPC.bat creates the Send To shortcut automatically^)
    echo   - DA360 depth alignment is enabled by default when checkpoints\DA360_large.pth is present
    echo   - Double-click splat files in the GUI to open them in the configured viewer
    echo.
    echo FOLDER STRUCTURE
    echo   SHARP_360_to_Splat.exe    - launcher
    echo   gsbox.exe                 - format conversion helper
    echo   Launch_SHARP_360_to_Splat.bat - script launcher
    echo   Setup_NewPC.bat           - one-time environment installer
    echo   third_party\DA360\       - vendored DA360 inference code
    echo   checkpoints\DA360_large.pth - default DA360 checkpoint
) > "%PACKAGE_DIR%\README.txt"

if exist "%PACKAGE_DIR%\splatapult\build\Release" (
    >> "%PACKAGE_DIR%\README.txt" echo   splatapult\build\Release\ - bundled viewer files
) else (
    >> "%PACKAGE_DIR%\README.txt" echo   Viewer payload not bundled in this package.
)

echo.
echo [5/5] Creating zip archive...
powershell -NoProfile -Command "Compress-Archive -Path '%PACKAGE_DIR%\*' -DestinationPath '%ZIP_PATH%' -Force"
if errorlevel 1 (
    echo ERROR: Could not create zip archive.
    pause
    exit /b 1
)

echo.
echo =====================================================
echo  Build complete!
echo.
echo  Folder: %PACKAGE_DIR%
echo  Zip:    %ZIP_PATH%
echo =====================================================
