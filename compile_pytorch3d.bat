@echo off
:: Rebuild pytorch3d FROM SOURCE with CUDA + sm_120 (Blackwell / RTX 5090).
:: The default pip install of pytorch3d on Windows gives a CPU-only build
:: which raises ``RuntimeError: Not compiled with GPU support`` from
:: rasterize_points — kills worldgen Stage 2 (traj_render's
:: multi_gpu_point_rendering).
::
:: Same vcvars + TORCH_CUDA_ARCH_LIST setup as compile_gsplat.bat.
::
:: Usage:
::   compile_pytorch3d.bat            build + install
::   compile_pytorch3d.bat --clean    uninstall first, then rebuild

setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"

set CLEAN=0
if /I "%~1"=="--clean" set CLEAN=1

:: --- MSVC env ---
set "INCLUDE="
set "LIB="
set "LIBPATH="
set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
    echo ERROR: vcvars64.bat not found at %VCVARS%
    exit /b 2
)
call %VCVARS% >nul

:: --- CUDA arch + memory throttle (matches compile_gsplat.bat) ---
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "TORCH_CUDA_ARCH_LIST=8.6;8.9;9.0;12.0+PTX"
set "MAX_JOBS=2"
:: pytorch3d's setup.py looks for CUB; CUDA 12+ bundles it but the flag
:: tells setup to enable CUDA explicitly.
set "FORCE_CUDA=1"

echo ============================================================
echo pytorch3d source build (CUDA + sm_120)
echo ============================================================
echo   CUDA_HOME            : %CUDA_HOME%
echo   TORCH_CUDA_ARCH_LIST : %TORCH_CUDA_ARCH_LIST%
echo   MAX_JOBS             : %MAX_JOBS%
echo   FORCE_CUDA           : %FORCE_CUDA%
echo   cl.exe               :
where cl.exe 2>nul
echo ============================================================

set "PIP=%~dp0.venv\Scripts\pip.exe"

if "%CLEAN%"=="1" (
    echo --- uninstalling existing pytorch3d ---
    "%PIP%" uninstall -y pytorch3d
)

:: Clone with git directly instead of letting pip's VCS backend handle it.
:: On Python 3.12 Windows, pip's `git --version` probe trips the stdlib
:: subprocess.Popen.communicate() thread race:
::   RuntimeError: cannot join thread before it is started
:: Doing the clone ourselves with a normal git invocation avoids pip's
:: subprocess thread path entirely; pip then sees a local dir to install.
set "PTSRC=%TEMP%\pytorch3d-src"
if exist "%PTSRC%" (
    echo --- removing stale clone at %PTSRC% ---
    rmdir /s /q "%PTSRC%"
)
echo --- cloning pytorch3d (depth=1) into %PTSRC% ---
git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git "%PTSRC%"
set RC=%ERRORLEVEL%
if not %RC%==0 (
    echo === FAIL clone rc=%RC% ===
    exit /b %RC%
)

echo --- installing pytorch3d from source (10-20 min compile) ---
:: --no-build-isolation so the build sees our torch + CUDA env.
"%PIP%" install --no-build-isolation --no-deps "%PTSRC%"
set RC=%ERRORLEVEL%
if not %RC%==0 (
    echo === FAIL rc=%RC% ===
    echo Check the pip log above for the actual error.
    exit /b %RC%
)

echo.
echo --- verifying CUDA support ---
"%~dp0.venv\Scripts\python.exe" -c "import torch; from pytorch3d import _C; print('pytorch3d CUDA OK:', hasattr(_C, 'PackedToPaddedCuda'))"
set RC=%ERRORLEVEL%
if not %RC%==0 (
    echo === FAIL: pytorch3d imported but CUDA path not present ===
    exit /b %RC%
)

echo.
echo === OK ===
echo Now retry:  .\run_with_log.bat --module worldgen
exit /b 0
