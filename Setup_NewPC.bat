@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "VENV_DIR=%SCRIPT_DIR%\.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128"
set "TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu"
set "TORCH_PINNED=torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0"
set "TRITON_PINNED=triton-windows<3.5"
set "SEEDVR2_REPO=https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
set "SEEDVR2_DIR=%SCRIPT_DIR%\seedvr2_videoupscaler"
set "THIRD_PARTY_DIR=%SCRIPT_DIR%\third_party"
set "IMAGEMAGICK_DIR=%THIRD_PARTY_DIR%\ImageMagick"
set "IMAGEMAGICK_BOOTSTRAP_DIR=%THIRD_PARTY_DIR%\_downloads"
set "IMAGEMAGICK_INSTALLER=%IMAGEMAGICK_BOOTSTRAP_DIR%\ImageMagick-installer.exe"
set "CHECKPOINT_DIR=%SCRIPT_DIR%\checkpoints"
set "CHECKPOINT_PATH=%CHECKPOINT_DIR%\DA360_large.pth"
set "CHECKPOINT_URL=https://drive.google.com/uc?id=1NYF4yJR83HEtxzOURLdmONeUe413auP6"
set "PYTHON_BOOTSTRAP="
set "DRY_RUN=0"
set "SKIP_CHECKPOINT=0"
set "INSTALL_SEEDVR2=ask"
set "CPU_ONLY=ask"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--dry-run" set "DRY_RUN=1"
if /I "%~1"=="--skip-checkpoint" set "SKIP_CHECKPOINT=1"
if /I "%~1"=="--skip-models" set "SKIP_CHECKPOINT=1"
if /I "%~1"=="--skip-seedvr2" set "INSTALL_SEEDVR2=0"
if /I "%~1"=="--with-seedvr2" set "INSTALL_SEEDVR2=1"
if /I "%~1"=="--cpu-only" set "CPU_ONLY=1"
if /I "%~1"=="--no-cuda" set "CPU_ONLY=1"
if /I "%~1"=="--with-cuda" set "CPU_ONLY=0"
if /I "%~1"=="--help" goto usage
if /I "%~1"=="/?" goto usage
shift
goto parse_args

:args_done

echo =====================================================
echo  SHARP_360_to_Splat  --  New PC Setup
echo =====================================================
echo.
echo This script will:
echo   1. Detect Python 3.13
echo   2. Create or reuse a local .venv
echo   3. Install PyTorch ^(CUDA 12.8 or CPU-only^)
echo   4. Install the vendored ml-sharp package into the venv
echo   5. Install core runtime extras and optionally SeedVR2
echo   6. Install ImageMagick into third_party\ImageMagick
echo   7. Download the default DA360 checkpoint for depth alignment
echo   8. Create a Send To shortcut for SHARP_360_to_Splat
echo.
echo The environment layout matches the portable SeedVR2 setup style:
echo   .venv\Scripts\python.exe
echo.
echo Install root: "%SCRIPT_DIR%"
if "%DRY_RUN%"=="1" echo Dry-run mode enabled. No changes will be written.
if "%SKIP_CHECKPOINT%"=="1" echo Checkpoint download disabled.

if not "%DRY_RUN%"=="1" (
    echo.
    echo Press any key to continue, or Ctrl+C to cancel.
    pause >nul
)

call :detect_python
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.13 was not found.
    echo.
    echo Install Python 3.13 first:
    echo   https://www.python.org/downloads/windows/
    echo.
    echo During setup, enable "Add python.exe to PATH".
    echo Then rerun this script.
    pause
    exit /b 1
)

call :resolve_install_profile
if errorlevel 1 goto fail

echo.
echo [1/8] Checking for git...
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
echo [2/8] Creating or reusing local virtual environment...
if exist "%PYTHON_EXE%" (
    echo       Reusing existing virtual environment: "%VENV_DIR%"
) else (
    if "%DRY_RUN%"=="1" (
        echo       [dry-run] %PYTHON_BOOTSTRAP% -m venv "%VENV_DIR%"
    ) else (
        call %PYTHON_BOOTSTRAP% -m venv "%VENV_DIR%"
        if errorlevel 1 goto fail
        echo       OK - created %VENV_DIR%
    )
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo.
echo [3/8] Upgrading pip tooling...
if "%DRY_RUN%"=="1" (
    echo       [dry-run] "%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel
) else (
    "%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel
    if errorlevel 1 goto fail
)

echo.
echo [4/8] Installing PyTorch with CUDA 12.8 support...
if "%CPU_ONLY%"=="1" (
    echo [4/8] Installing PyTorch CPU-only wheels...
) else (
    echo [4/8] Installing PyTorch with CUDA 12.8 support...
)
echo       IMPORTANT: This must happen before ml-sharp so the intended torch build is selected.
echo       This may take a few minutes...
if "%CPU_ONLY%"=="1" (
    if "%DRY_RUN%"=="1" (
        echo       [dry-run] "%PYTHON_EXE%" -m pip install %TORCH_PINNED% --index-url "%TORCH_CPU_INDEX_URL%"
        echo       [dry-run] fallback: "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url "%TORCH_CPU_INDEX_URL%"
    ) else (
        "%PYTHON_EXE%" -m pip install %TORCH_PINNED% --index-url "%TORCH_CPU_INDEX_URL%"
        if errorlevel 1 (
            echo.
            echo WARNING: Pinned Torch 2.8.0 CPU wheels not available.
            echo          Trying the latest CPU wheels instead...
            "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url "%TORCH_CPU_INDEX_URL%"
            if errorlevel 1 goto fail
        )
    )
) else (
    if "%DRY_RUN%"=="1" (
        echo       [dry-run] "%PYTHON_EXE%" -m pip install %TORCH_PINNED% --index-url "%TORCH_CUDA_INDEX_URL%"
        echo       [dry-run] fallback: "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url "%TORCH_CUDA_INDEX_URL%"
    ) else (
        "%PYTHON_EXE%" -m pip install %TORCH_PINNED% --index-url "%TORCH_CUDA_INDEX_URL%"
        if errorlevel 1 (
            echo.
            echo WARNING: Pinned Torch 2.8.0 CUDA 12.8 wheels not available.
            echo          Trying the latest CUDA 12.8 wheels instead...
            "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url "%TORCH_CUDA_INDEX_URL%"
            if errorlevel 1 goto fail
        )
    )
)

echo.
echo [5/8] Installing SHARP_360_to_Splat runtime dependencies...
echo       Installing local vendored ml-sharp in editable mode...
if "%DRY_RUN%"=="1" (
    echo       [dry-run] "%PYTHON_EXE%" -m pip install -e "%SCRIPT_DIR%\ml-sharp"
) else (
    "%PYTHON_EXE%" -m pip install -e "%SCRIPT_DIR%\ml-sharp"
    if errorlevel 1 goto fail
)

echo       Installing core runtime extras used by this repo...
if "%DRY_RUN%"=="1" (
    echo       [dry-run] "%PYTHON_EXE%" -m pip install opencv-python gdown
) else (
    "%PYTHON_EXE%" -m pip install opencv-python gdown
    if errorlevel 1 goto fail
)

if "%INSTALL_SEEDVR2%"=="1" (
    echo       Installing optional SeedVR2 support...
    if "%DRY_RUN%"=="1" (
        echo       [dry-run] "%PYTHON_EXE%" -m pip install "%TRITON_PINNED%"
        echo       [dry-run] git clone "%SEEDVR2_REPO%" "%SEEDVR2_DIR%"
        echo       [dry-run] "%PYTHON_EXE%" -m pip install -r "%SEEDVR2_DIR%\requirements.txt"
    ) else (
        "%PYTHON_EXE%" -m pip install "%TRITON_PINNED%"
        if errorlevel 1 goto fail

        if exist "%SEEDVR2_DIR%\.git" (
            echo       Updating SeedVR2 checkout...
            git -C "%SEEDVR2_DIR%" pull --ff-only
            if errorlevel 1 goto fail
        ) else if exist "%SEEDVR2_DIR%" (
            echo ERROR: "%SEEDVR2_DIR%" exists but is not a Git checkout.
            echo        Delete or rename it, then rerun Setup_NewPC.bat.
            goto fail
        ) else (
            echo       Cloning SeedVR2 from %SEEDVR2_REPO% ...
            git clone "%SEEDVR2_REPO%" "%SEEDVR2_DIR%"
            if errorlevel 1 goto fail
        )

        echo       Installing SeedVR2 Python requirements...
        "%PYTHON_EXE%" -m pip install -r "%SEEDVR2_DIR%\requirements.txt"
        if errorlevel 1 goto fail
    )
) else (
    echo       Skipping optional SeedVR2 install.
)
echo       OK - runtime dependencies installed.

echo.
echo [6/8] Ensuring ImageMagick is available...
if exist "%IMAGEMAGICK_DIR%\magick.exe" (
    echo       OK - using repo-managed ImageMagick at "%IMAGEMAGICK_DIR%".
) else if "%DRY_RUN%"=="1" (
    echo       [dry-run] mkdir "%THIRD_PARTY_DIR%"
    echo       [dry-run] mkdir "%IMAGEMAGICK_BOOTSTRAP_DIR%"
    echo       [dry-run] download latest ImageMagick x64 installer from the official GitHub release API
    echo       [dry-run] install silently into "%IMAGEMAGICK_DIR%"
) else (
    if not exist "%THIRD_PARTY_DIR%" mkdir "%THIRD_PARTY_DIR%"
    if not exist "%IMAGEMAGICK_BOOTSTRAP_DIR%" mkdir "%IMAGEMAGICK_BOOTSTRAP_DIR%"
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$ProgressPreference = 'SilentlyContinue'; " ^
        "$release = Invoke-RestMethod 'https://api.github.com/repos/ImageMagick/ImageMagick/releases/latest'; " ^
        "$asset = $release.assets | Where-Object { $_.name -match 'Q16-HDRI-x64-dll\.exe$|Q16-x64-dll\.exe$|Q8-x64-dll\.exe$' } | Select-Object -First 1; " ^
        "if (-not $asset) { throw 'Could not find the latest ImageMagick x64 installer asset.' }; " ^
        "Invoke-WebRequest -UseBasicParsing -Uri $asset.browser_download_url -OutFile '%IMAGEMAGICK_INSTALLER%'"
    if errorlevel 1 (
        echo       WARNING: Could not download the latest ImageMagick installer.
        echo                Panorama optimization will fall back to PATH or manual selection.
    ) else (
        echo       Installing ImageMagick into "%IMAGEMAGICK_DIR%"...
        "%IMAGEMAGICK_INSTALLER%" /SP- /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR="%IMAGEMAGICK_DIR%"
        if errorlevel 1 (
            echo       WARNING: ImageMagick installation into third_party failed.
            echo                The GUI can still browse to magick.exe manually later.
        ) else if exist "%IMAGEMAGICK_DIR%\magick.exe" (
            echo       OK - repo-managed ImageMagick installed.
        ) else (
            echo       WARNING: ImageMagick installer completed, but magick.exe was not found in "%IMAGEMAGICK_DIR%".
        )
    )
)

echo.
echo [7/8] Downloading the default DA360 checkpoint...
if "%SKIP_CHECKPOINT%"=="1" (
    echo       Skipping checkpoint download.
) else if exist "%CHECKPOINT_PATH%" (
    echo       DA360 checkpoint already exists -- skipping download.
) else if "%DRY_RUN%"=="1" (
    echo       [dry-run] mkdir "%CHECKPOINT_DIR%"
    echo       [dry-run] "%PYTHON_EXE%" -m gdown %CHECKPOINT_URL% -O "%CHECKPOINT_PATH%"
) else (
    if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"
    "%PYTHON_EXE%" -m gdown %CHECKPOINT_URL% -O "%CHECKPOINT_PATH%"
    if errorlevel 1 (
        echo.
        echo WARNING: Could not download the DA360 checkpoint automatically.
        echo          You can still run the app, but DA360 alignment must be disabled
        echo          or the checkpoint must be downloaded manually later.
    ) else (
        echo       OK - DA360 checkpoint downloaded.
    )
)

echo.
echo [8/8] Creating Send To shortcut...
set "EXE_PATH=%SCRIPT_DIR%\SHARP_360_to_Splat.exe"
if not exist "%EXE_PATH%" (
    echo       WARNING: SHARP_360_to_Splat.exe was not found next to this script.
    echo       The shortcut will target !Launch_SHARP_360_to_Splat.bat instead.
    set "EXE_PATH=%SCRIPT_DIR%\!Launch_SHARP_360_to_Splat.bat"
)
if "%DRY_RUN%"=="1" (
    echo       [dry-run] create Send To shortcut for "%EXE_PATH%"
) else (
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
)

echo.
echo =====================================================
echo  Setup complete!
echo.
echo  Environment: %PYTHON_EXE%
echo.
echo  FIRST RUN NOTE:
echo    On the first conversion, SHARP will download its model
echo    weights (~500MB) from Apple's servers automatically.
echo    This only happens once.
echo =====================================================
if not "%DRY_RUN%"=="1" pause
exit /b 0

:usage
echo.
echo Usage: %~n0 [--dry-run] [--skip-checkpoint]
echo.
echo   --dry-run           Show what would happen without changing anything.
echo   --skip-checkpoint   Skip downloading checkpoints\DA360_large.pth.
echo   --skip-models       Alias for --skip-checkpoint.
echo   --skip-seedvr2      Do not clone or install SeedVR2 support.
echo   --with-seedvr2      Force installation of SeedVR2 support without prompting.
echo   --cpu-only          Install CPU-only PyTorch wheels.
echo   --no-cuda           Alias for --cpu-only.
echo   --with-cuda         Force CUDA PyTorch wheels without prompting.
exit /b 0

:resolve_install_profile
if /I "%CPU_ONLY%"=="ask" (
    if "%DRY_RUN%"=="1" (
        set "CPU_ONLY=0"
        echo       [dry-run] Defaulting to CUDA-enabled torch.
    ) else (
        echo.
        choice /C GC /N /M "Install GPU CUDA torch or CPU-only torch? [G/C]"
        if errorlevel 2 (
            set "CPU_ONLY=1"
        ) else (
            set "CPU_ONLY=0"
        )
    )
)

if /I "%INSTALL_SEEDVR2%"=="ask" (
    if "%CPU_ONLY%"=="1" (
        set "INSTALL_SEEDVR2=0"
        echo       CPU-only profile selected; SeedVR2 install will be skipped.
    ) else if "%DRY_RUN%"=="1" (
        set "INSTALL_SEEDVR2=0"
        echo       [dry-run] Defaulting to no SeedVR2 install.
    ) else (
        echo.
        choice /C YN /N /M "Install optional SeedVR2 support now? [Y/N]"
        if errorlevel 2 (
            set "INSTALL_SEEDVR2=0"
        ) else (
            set "INSTALL_SEEDVR2=1"
        )
    )
)

if "%CPU_ONLY%"=="1" if "%INSTALL_SEEDVR2%"=="1" (
    echo       WARNING: SeedVR2 is GPU-oriented and will be skipped in CPU-only mode.
    set "INSTALL_SEEDVR2=0"
)

if "%CPU_ONLY%"=="1" (
    echo       Install profile: CPU-only torch.
) else (
    echo       Install profile: CUDA torch.
)
if "%INSTALL_SEEDVR2%"=="1" (
    echo       SeedVR2 support: enabled.
) else (
    echo       SeedVR2 support: skipped.
)
exit /b 0

:detect_python
where py >nul 2>&1
if not errorlevel 1 (
    py -3.13 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_BOOTSTRAP=py -3.13"
)

if defined PYTHON_BOOTSTRAP exit /b 0

where python >nul 2>&1
if errorlevel 1 exit /b 1

for /f "delims=" %%V in ('python -c "import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))"') do set "PYTHON_VERSION=%%V"
if "%PYTHON_VERSION%"=="3.13" (
    set "PYTHON_BOOTSTRAP=python"
    exit /b 0
)

exit /b 1

:fail
echo.
echo Setup failed.
if not "%DRY_RUN%"=="1" pause
exit /b 1
