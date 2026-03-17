@echo off
setlocal EnableDelayedExpansion

echo =====================================================
echo  SHARP_360_to_Splat  --  New PC Setup
echo =====================================================
echo.
echo This script will:
echo   1. Locate Anaconda or Miniconda reliably on Windows
echo   2. Verify git is available
echo   3. Create the 'sharp' conda environment (Python 3.13)
echo   4. Install PyTorch with CUDA 12.8 BEFORE ml-sharp
echo   5. Install Apple's ml-sharp package from GitHub
echo   6. Download the default DA360 checkpoint for depth alignment
echo   7. Create a Send To shortcut for SHARP_360_to_Splat
echo.
echo Press any key to continue, or Ctrl+C to cancel.
pause >nul

set "SCRIPT_DIR=%~dp0"
set "CONDA_BASE="
set "CONDA_CMD="

echo.
echo [1/7] Locating conda...
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
    if exist "%%~B\condabin\conda.bat" (
        set "CONDA_BASE=%%~B"
        set "CONDA_CMD=%%~B\condabin\conda.bat"
        goto :found_conda
    )
)

for /f "usebackq delims=" %%B in (`conda info --base 2^>nul`) do (
    if exist "%%~B\condabin\conda.bat" (
        set "CONDA_BASE=%%~B"
        set "CONDA_CMD=%%~B\condabin\conda.bat"
        goto :found_conda
    )
)

echo.
echo ERROR: Could not find Anaconda or Miniconda.
echo.
echo Please install one of them first:
echo   https://www.anaconda.com/download
echo.
echo After installing, open a fresh terminal and run this script again.
pause
exit /b 1

:found_conda
echo       OK - conda found at %CONDA_BASE%

echo.
echo [2/7] Checking for git...
where git >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: git not found in PATH.
    echo.
    echo Please install Git for Windows:
    echo   https://git-scm.com/download/win
    echo.
    echo After installing, open a fresh terminal and run this script again.
    pause
    exit /b 1
)
echo       OK - git found.

echo.
echo [3/7] Creating conda environment 'sharp' with Python 3.13...
call "%CONDA_CMD%" env list | findstr /R /C:"^[* ]*sharp[ ]" >nul 2>&1
if not errorlevel 1 (
    echo       Environment 'sharp' already exists -- skipping creation.
    goto :install_torch
)
call "%CONDA_CMD%" create -n sharp python=3.13 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment.
    pause
    exit /b 1
)

:install_torch
echo.
echo [4/7] Installing PyTorch with CUDA 12.8 support...
echo       IMPORTANT: This must happen before ml-sharp to avoid CPU-only torch.
echo       This may take a few minutes...
call "%CONDA_CMD%" run -n sharp pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo.
    echo WARNING: PyTorch CUDA install failed.
    echo          Trying CPU-only version as fallback...
    call "%CONDA_CMD%" run -n sharp pip install torch torchvision
    if errorlevel 1 (
        echo ERROR: Could not install PyTorch at all.
        pause
        exit /b 1
    )
    echo       WARNING: CPU-only PyTorch installed. Conversions will be slow.
) else (
    echo       OK - PyTorch with CUDA installed.
)

echo.
echo [5/7] Installing ml-sharp from GitHub...
echo       Step 1: Cloning repository (this may take 1-2 minutes, please wait)...
echo       Step 2: Installing dependencies (another 1-2 minutes)...
echo       If it appears stuck after the git clone line, it is still working.
call "%CONDA_CMD%" run -n sharp pip install "git+https://github.com/apple/ml-sharp.git"
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install ml-sharp.
    echo        Make sure git is installed and you have internet access.
    pause
    exit /b 1
)
echo       OK - ml-sharp installed.

echo.
echo [6/7] Installing gdown and downloading the default DA360 checkpoint...
call "%CONDA_CMD%" run -n sharp pip install gdown
if errorlevel 1 (
    echo.
    echo WARNING: Could not install gdown automatically.
    echo          DA360 depth alignment will stay unavailable until you install gdown
    echo          and download checkpoints\DA360_large.pth manually.
    goto :create_shortcut
)

if not exist "%SCRIPT_DIR%checkpoints" mkdir "%SCRIPT_DIR%checkpoints"
if exist "%SCRIPT_DIR%checkpoints\DA360_large.pth" (
    echo       DA360 checkpoint already exists -- skipping download.
) else (
    call "%CONDA_CMD%" run -n sharp python -m gdown https://drive.google.com/uc?id=1NYF4yJR83HEtxzOURLdmONeUe413auP6 -O "%SCRIPT_DIR%checkpoints\DA360_large.pth"
    if errorlevel 1 (
        echo.
        echo WARNING: Could not download the DA360 checkpoint automatically.
        echo          You can still run the app, but DA360 alignment must be disabled
        echo          or the checkpoint must be downloaded manually later.
    ) else (
        echo       OK - DA360 checkpoint downloaded.
    )
)

:create_shortcut
echo.
echo.
echo [7/7] Creating Send To shortcut...
set "EXE_PATH=%SCRIPT_DIR%SHARP_360_to_Splat.exe"
if not exist "%EXE_PATH%" (
    echo       WARNING: SHARP_360_to_Splat.exe was not found next to this script.
    echo       The shortcut will target Launch_SHARP_360_to_Splat.bat instead.
    set "EXE_PATH=%SCRIPT_DIR%Launch_SHARP_360_to_Splat.bat"
)
powershell -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell; " ^
    "$lnk = $ws.CreateShortcut([Environment]::GetFolderPath('SendTo') + '\SHARP_360_to_Splat.lnk'); " ^
    "$lnk.TargetPath = '%EXE_PATH:\=\\%'; " ^
    "$lnk.WorkingDirectory = '%SCRIPT_DIR:\=\\%'; " ^
    "$lnk.Save()"
if errorlevel 1 (
    echo       WARNING: Could not create Send To shortcut automatically.
    echo       You can create it manually and place it in:
    echo       %%APPDATA%%\Microsoft\Windows\SendTo\
) else (
    echo       OK - shortcut created.
    echo       Right-click any JPEG/PNG -^> Send To -^> SHARP_360_to_Splat
)

echo.
echo =====================================================
echo  Setup complete!
echo.
echo  FIRST RUN NOTE:
echo    On the first conversion, SHARP will download its model
echo    weights (~500MB) from Apple's servers automatically.
echo    This only happens once.
echo =====================================================
pause
