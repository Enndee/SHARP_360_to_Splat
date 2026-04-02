@echo off
setlocal EnableDelayedExpansion

set "APP_NAME=SHARP_360_to_Splat"
set "RELEASE_VERSION=1.5.2"
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

set "VENV_DIR=%SCRIPT_DIR%\.venv"
set "SHARP_PY=%VENV_DIR%\Scripts\python.exe"
set "PACKAGE_DIR=release_pkg\%APP_NAME%_v%RELEASE_VERSION%_Windows"
set "ZIP_PATH=release_pkg\%APP_NAME%_v%RELEASE_VERSION%_Windows.zip"
set "VIEWER_SOURCE=splatapult\build\Release"

echo =====================================================
echo  SHARP_360_to_Splat  --  Build EXE
echo =====================================================
echo.

if not exist "%SHARP_PY%" (
    echo ERROR: Could not find the local .venv interpreter.
    echo        Run Setup_NewPC.bat first to create it.
    pause
    exit /b 1
)

echo [1/5] Found local venv: %SHARP_PY%
echo.

echo [2/5] Installing PyInstaller...
"%SHARP_PY%" -m pip install pyinstaller --quiet
if errorlevel 1 (
    echo ERROR: pip install pyinstaller failed.
    pause
    exit /b 1
)

set "PYTHON_BASE="
set "TCL_LIBRARY="
set "TK_LIBRARY="
set "PYINFO_FILE=%TEMP%\sharp_pyinfo_%RANDOM%_%RANDOM%.txt"
"%SHARP_PY%" -c "import sys,pathlib; base=pathlib.Path(sys.base_prefix); print(base); print(base / 'tcl' / 'tcl8.6'); print(base / 'tcl' / 'tk8.6')" > "%PYINFO_FILE%"
if errorlevel 1 (
    echo ERROR: Could not determine the base Python installation for the venv.
    if exist "%PYINFO_FILE%" del /Q "%PYINFO_FILE%" >nul 2>&1
    pause
    exit /b 1
)
set /a PYINFO_INDEX=0
for /f "usebackq delims=" %%A in ("%PYINFO_FILE%") do (
    if !PYINFO_INDEX! EQU 0 set "PYTHON_BASE=%%~A"
    if !PYINFO_INDEX! EQU 1 set "TCL_LIBRARY=%%~A"
    if !PYINFO_INDEX! EQU 2 set "TK_LIBRARY=%%~A"
    set /a PYINFO_INDEX+=1
)
if exist "%PYINFO_FILE%" del /Q "%PYINFO_FILE%" >nul 2>&1

if not defined PYTHON_BASE (
    echo ERROR: Could not determine the base Python installation for the venv.
    pause
    exit /b 1
)

echo       Base Python = %PYTHON_BASE%
echo       TCL_LIBRARY=%TCL_LIBRARY%
echo       TK_LIBRARY=%TK_LIBRARY%
set "PATH=%PYTHON_BASE%;%PYTHON_BASE%\DLLs;%PATH%"

echo.
echo [3/5] Building lightweight %APP_NAME%.exe launcher ...
"%SHARP_PY%" -m PyInstaller ^
    --onefile ^
    --windowed ^
    --name "%APP_NAME%" ^
    --add-data "%TCL_LIBRARY%;tcl" ^
    --add-data "%TK_LIBRARY%;tk" ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    --exclude-module triton ^
    "bootstrap_launcher.py"

if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)

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
copy /Y "!Launch_SHARP_360_to_Splat.bat" "%PACKAGE_DIR%\" >nul
copy /Y "Setup_NewPC.bat" "%PACKAGE_DIR%\" >nul
copy /Y "Easy_360_SHARP_GUI.py" "%PACKAGE_DIR%\" >nul
copy /Y "insp_to_splat.py" "%PACKAGE_DIR%\" >nul
copy /Y "insp_settings.json" "%PACKAGE_DIR%\" >nul
copy /Y "seedvr2_settings.json" "%PACKAGE_DIR%\" >nul
copy /Y "README.md" "%PACKAGE_DIR%\" >nul
copy /Y "gsbox.exe" "%PACKAGE_DIR%\" >nul

if not exist "%PACKAGE_DIR%\ml-sharp" mkdir "%PACKAGE_DIR%\ml-sharp"
xcopy /E /I /Y "ml-sharp\*" "%PACKAGE_DIR%\ml-sharp\" >nul

if not exist "%PACKAGE_DIR%\third_party\DA360" mkdir "%PACKAGE_DIR%\third_party\DA360"
xcopy /E /I /Y "third_party\DA360\*" "%PACKAGE_DIR%\third_party\DA360\" >nul

if exist "third_party\ImageMagick\magick.exe" (
    if not exist "%PACKAGE_DIR%\third_party\ImageMagick" mkdir "%PACKAGE_DIR%\third_party\ImageMagick"
    xcopy /E /I /Y "third_party\ImageMagick\*" "%PACKAGE_DIR%\third_party\ImageMagick\" >nul
    echo       Included repo-managed ImageMagick from third_party\ImageMagick.
) else (
    echo       ImageMagick was not found under third_party\ImageMagick.
    echo       The release package will rely on Setup_NewPC.bat to install it on the target machine.
)

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

(
    echo SHARP_360_to_Splat
    echo ====================
    echo.
    echo REQUIREMENTS FOR A NEW PC
    echo   1. Windows 11, NVIDIA GPU ^(RTX 20/30/40/50 series with CUDA^)
    echo   2. Python 3.13          - https://www.python.org/downloads/windows/
    echo   3. Git for Windows      - https://git-scm.com/download/win
    echo.
    echo SETUP ^(first time only^)
    echo   Run Setup_NewPC.bat  -- creates the local .venv, clones SeedVR2, and installs dependencies.
    echo.
    echo USAGE
    echo   - Double-click SHARP_360_to_Splat.exe and it will launch the GUI or offer to run Setup_NewPC.bat on first use, OR
    echo   - Right-click any JPEG/PNG -^> Send To -^> SHARP_360_to_Splat
    echo     ^(Setup_NewPC.bat creates the Send To shortcut automatically^)
    echo   - DA360 depth alignment is enabled by default when checkpoints\DA360_large.pth is present
    echo   - If third_party\ImageMagick\ is bundled, the GUI uses that local copy first
    echo   - SeedVR2 supports equalizing non-square faces before upscale and an optional pre-upscale downscale factor
    echo   - Double-click splat files in the GUI to open them in the configured viewer
    echo.
    echo IMPORTANT
    echo   - This EXE is a lightweight launcher only.
    echo   - Torch, CUDA runtime packages, and other heavy Python dependencies are NOT bundled into the EXE package.
    echo   - Setup_NewPC.bat installs those dependencies into .venv on the target machine.
    echo.
    echo FOLDER STRUCTURE
    echo   SHARP_360_to_Splat.exe    - lightweight launcher
    echo   gsbox.exe                 - format conversion helper
    echo   !Launch_SHARP_360_to_Splat.bat - script launcher
    echo   Setup_NewPC.bat           - one-time environment installer
    echo   Easy_360_SHARP_GUI.py     - main GUI source
    echo   insp_to_splat.py          - main pipeline source
    echo   insp_settings.json        - pipeline defaults
    echo   seedvr2_settings.json     - SeedVR2 defaults
    echo   ml-sharp\                - vendored SHARP source used by setup/runtime
    echo   third_party\DA360\       - vendored DA360 inference code
    echo   third_party\ImageMagick\ - optional bundled ImageMagick runtime
    echo   checkpoints\DA360_large.pth - default DA360 checkpoint
) > "%PACKAGE_DIR%\README.txt"

if exist "%PACKAGE_DIR%\splatapult\build\Release" (
    >> "%PACKAGE_DIR%\README.txt" echo   splatapult\build\Release\ - bundled viewer files
) else (
    >> "%PACKAGE_DIR%\README.txt" echo   Viewer payload not bundled in this package.
)

if not exist "%PACKAGE_DIR%\third_party\ImageMagick\magick.exe" (
    >> "%PACKAGE_DIR%\README.txt" echo   ImageMagick not bundled; Setup_NewPC.bat will install it into third_party\ImageMagick on the target PC.
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
