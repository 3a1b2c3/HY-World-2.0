@echo off
:: HY-World 2.0 install pipeline. Run from the repo root inside the .venv.
::
:: Steps:
::   1. requirements.txt
::   2. utils3d (not pinned upstream — needed by worldgen/traj_generate.py)
::   3. activate MSVC + CUDA (so the source builds in step 4 actually compile)
::   4. requirements_git.txt (pytorch3d, MoGe, nerfview, spz, fused-ssim from git)
::   5. third_party/gsplat_maskgaussian (custom CUDA gsplat fork — worldgen only)
::   6. third_party/navmesh + recastnavigation submodule (worldgen only)
::   7. python download_models.py --module all
::
:: Skip with: setup.bat --skip-worldgen   (stops after step 4 — enough for worldrecon + panogen)
::
:: Re-runnable. Each step is idempotent except step 4 (rebuilds git deps).

setlocal enableextensions

set SKIP_WORLDGEN=0
if /I "%~1"=="--skip-worldgen" set SKIP_WORLDGEN=1

cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found at %~dp0.venv. Create it first:  python -m venv .venv
    exit /b 2
)
call .venv\Scripts\activate.bat

set PY=%~dp0.venv\Scripts\python.exe
set PIP=%PY% -m pip
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
if not exist "%CUDA_HOME%" set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
if not exist "%CUDA_HOME%" (
    echo ERROR: no CUDA toolkit found under "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\".
    exit /b 2
)
set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
    echo ERROR: MSVC vcvars64.bat not found at %VCVARS%. Install VS 2022 Community + C++ workload.
    exit /b 2
)

echo ============================================================
echo HY-World 2.0 setup
echo ============================================================
echo   venv      : %~dp0.venv
echo   CUDA_HOME : %CUDA_HOME%
echo   MSVC      : %VCVARS%
echo   worldgen  : %SKIP_WORLDGEN% (1 = skip steps 5-6)
echo ============================================================

echo.
echo --- 1/7  pip install -r requirements.txt ---
%PIP% install -r requirements.txt
if errorlevel 1 ( echo FAIL step1 & exit /b 1 )

echo.
echo --- 2/7  pip install utils3d (worldgen helper, missing from reqs) ---
:: --no-deps: older utils3d (<=0.1.2) pins open3d, which has no Python 3.12 + Windows
:: wheel and breaks pip resolution. utils3d 0.1.3 doesn't need open3d at import time
:: for our worldgen usage (traj_generate.py only uses light utilities).
%PIP% install --no-deps utils3d
if errorlevel 1 ( echo FAIL step2 & exit /b 1 )

echo.
echo --- 3/7  activate MSVC + CUDA for source builds ---
:: Clear user-level INCLUDE/LIB before vcvars. On this machine the user env has
:: stale racer-x\_build\host-deps\winsdk\include\* paths that REPLACE the proper
:: Windows SDK headers — cl.exe then can't find corecrt.h / windows.h and dies
:: with "cl.exe failed: None" during torch CUDAExtension builds. vcvars64.bat
:: appends to INCLUDE/LIB, it doesn't reset them — so we wipe first.
set "INCLUDE="
set "LIB="
set "LIBPATH="
call %VCVARS% >nul
if errorlevel 1 ( echo FAIL step3 (vcvars64.bat) & exit /b 1 )
:: Pin CUDA_HOME / CUDA_PATH to the cu128 toolkit (matches torch 2.7+cu128 wheels).
:: Required by torch.utils.cpp_extension.CUDAExtension during step-4 source builds.
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if not defined CUDA_PATH set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
where cl >nul 2>nul
if errorlevel 1 ( echo FAIL step3: cl.exe still not on PATH after vcvars & exit /b 1 )
echo cl   : OK
where nvcc >nul 2>nul
if errorlevel 1 ( echo WARN: nvcc not on PATH — some CUDA builds may fail )

echo.
echo --- 4/7  install_git_deps.py  (manual clone + pip install per dep; ~10 min) ---
:: Why not pip / uv pip directly:
::   - pip's VCS handler hits subprocess._communicate thread race on Py3.12+Win
::   - uv pip --no-build-isolation runs builds in subprocesses that don't inherit
::     CUDA_HOME on Windows (torch CUDAExtension fails)
:: install_git_deps.py: git-clone manually, then `pip install` from local path.
"%PY%" "%~dp0install_git_deps.py"
if errorlevel 1 ( echo FAIL step4 & exit /b 1 )

if "%SKIP_WORLDGEN%"=="1" goto :download_models

echo.
echo --- 5/7  third_party/gsplat_maskgaussian (custom CUDA gsplat fork) ---
if not exist "hyworld2\worldgen\third_party\gsplat_maskgaussian" (
    echo ERROR: hyworld2\worldgen\third_party\gsplat_maskgaussian not found.
    echo Did you clone the repo with --recursive? Run: git submodule update --init --recursive
    exit /b 2
)
pushd "hyworld2\worldgen\third_party\gsplat_maskgaussian"
:: --no-build-isolation so setup.py can `import torch` (build env has no torch)
:: --no-deps so pip doesn't recurse into git+ deps + hit the Py3.12 thread race
%PIP% install --no-build-isolation --no-deps -e .
set RC=%ERRORLEVEL%
popd
if not %RC%==0 ( echo FAIL step5 rc=%RC% & exit /b %RC% )

echo.
echo --- 6/7  third_party/navmesh + recastnavigation submodule ---
if not exist "hyworld2\worldgen\third_party\navmesh" (
    echo ERROR: hyworld2\worldgen\third_party\navmesh not found.
    exit /b 2
)
pushd "hyworld2\worldgen\third_party\navmesh"
if not exist "recastnavigation\.git" (
    echo Cloning recastnavigation submodule...
    git submodule update --init --recursive
    if errorlevel 1 ( echo FAIL step6 (submodule init) & popd & exit /b 1 )
)
:: --no-build-isolation so setup.py can `import torch`/use venv packages.
%PIP% install --no-build-isolation --no-deps -e .
set RC=%ERRORLEVEL%
popd
if not %RC%==0 ( echo FAIL step6 rc=%RC% & exit /b %RC% )

:download_models
echo.
echo --- 7/7  download_models.py --all (per-file, dodges Windows thread_map crash) ---
%PY% download_models.py --all
if errorlevel 1 ( echo FAIL step7 & exit /b 1 )

echo.
echo ============================================================
echo HY-World 2.0 setup complete.
echo Next:  run_example.bat                  (all modules)
echo        run_example.bat --module worldrecon
echo ============================================================
exit /b 0
