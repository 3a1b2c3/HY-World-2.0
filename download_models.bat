@echo off
:: Download HY-World 2.0 model weights via download_models.py.
::
:: Resolution order for python (no hard venv assumption):
::   1. HYWORLD_PY env var (absolute path or command name)
::   2. .venv\Scripts\python.exe if present
::   3. plain `python` on PATH
::
:: Usage:
::   download_models.bat                          ERROR: pick at least one flag
::   download_models.bat --mirror                 WorldMirror-2 only (~1.2B, recommended start)
::   download_models.bat --pano-lora              HY-Pano-2 LoRA only (~850 MB)
::   download_models.bat --qwen                   Qwen3-VL-8B-Instruct (~16 GB)
::   download_models.bat --all                    mirror + pano-lora + qwen (excludes the huge full pano)
::   download_models.bat --pano                   HY-Pano-2 full model (~150 GB — only if you really need it)
::
:: Env overrides:
::   set HYWORLD_CKPT_DIR=D:\models\hyworld    alternate destination
::   set HF_TOKEN=hf_xxx                       for gated repos
::   set HYWORLD_PY=C:\path\python.exe         pick a specific interpreter

setlocal enableextensions
cd /d "%~dp0"

if defined HYWORLD_PY (
    set PY=%HYWORLD_PY%
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set PY=%~dp0.venv\Scripts\python.exe
) else (
    set PY=python
)

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo ============================================================
echo HY-World 2.0 model download
echo ============================================================
echo   python      : %PY%
echo   args        :%*
if defined HYWORLD_CKPT_DIR (
    echo   ckpt_dir    : %HYWORLD_CKPT_DIR% ^(env override^)
) else (
    echo   ckpt_dir    : %~dp0checkpoint ^(default^)
)
echo ============================================================

"%PY%" "%~dp0download_models.py" %*
exit /b %ERRORLEVEL%
