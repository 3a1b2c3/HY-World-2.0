@echo off
:: Pre-compile gsplat_maskgaussian's CUDA extension OUTSIDE the viewer/training
:: process, so errors are visible and the result gets cached at
:: ~\AppData\Local\torch_extensions\. After this succeeds, run_example.bat
:: --module viewer loads the cached extension instantly.
::
:: Why we do it here instead of pip install -e .:
::   gsplat_maskgaussian's setup.py installs ONLY the Python sources; the CUDA
::   backend is JIT-compiled at first import via torch.utils.cpp_extension.load.
::   The same JIT machinery used inside the viewer, but here we can inspect
::   errors instead of having them swallowed by the viewer process.
::
:: Usage:
::   compile_gsplat.bat              compile (~5-10 min on first run)
::   compile_gsplat.bat --clean      wipe the existing cache first

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

:: --- CUDA arch + memory throttle ---
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "TORCH_CUDA_ARCH_LIST=8.6;8.9;9.0;12.0+PTX"
set "MAX_JOBS=2"

echo ============================================================
echo gsplat_maskgaussian pre-compile
echo ============================================================
echo   CUDA_HOME            : %CUDA_HOME%
echo   TORCH_CUDA_ARCH_LIST : %TORCH_CUDA_ARCH_LIST%
echo   MAX_JOBS             : %MAX_JOBS% ^(nvcc uses ~7 GB per job^)
echo   cl.exe               :
where cl.exe 2>nul
echo ============================================================

if "%CLEAN%"=="1" (
    set "CACHE_DIR=%LOCALAPPDATA%\torch_extensions\Cache"
    if exist "!CACHE_DIR!\gsplat_cuda" (
        echo --- wiping !CACHE_DIR!\gsplat_cuda ---
        rmdir /s /q "!CACHE_DIR!\gsplat_cuda"
    )
)

:: Trigger JIT compile by importing + calling a function from each module.
.venv\Scripts\python.exe -X utf8 -c "import sys; sys.path.insert(0, r'hyworld2/worldgen/third_party/gsplat_maskgaussian'); print('Importing gsplat.cuda._backend (this triggers the JIT compile)...'); import gsplat.cuda._backend as _b; print('Forcing _C load ...'); _ = _b._C; print(); print('=== gsplat compiled and importable ==='); print('  cached at:', _b.__file__)"
set RC=%ERRORLEVEL%
echo.
if not %RC%==0 (
    echo === FAIL rc=%RC% ===
    echo The full ninja build log is at:
    echo   %LOCALAPPDATA%\torch_extensions\Cache\gsplat_cuda\<hash>\build.ninja
    echo Open it + look for the lines just before "ninja: build stopped".
    exit /b %RC%
)
echo === OK ===
echo Now run:  .\run_example.bat --module viewer
exit /b 0
