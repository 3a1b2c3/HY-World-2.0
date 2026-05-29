@echo off
:: Launch the minimal OpenAI-compat Qwen3-VL server (vLLM stand-in).
:: Run this in a SEPARATE shell, then run run_example.bat --module worldgen here.
setlocal enableextensions
cd /d "%~dp0"

:: Windows kernel-object exhaustion guard. HF transformers' async loader spawns
:: a ThreadPoolExecutor that leaks threads on Windows, eventually tripping
:: `RuntimeError: can't allocate lock` in anyio worker threads.
set HF_DEACTIVATE_ASYNC_LOAD=1
set TOKENIZERS_PARALLELISM=false

call .venv\Scripts\activate.bat
python vlm_server.py %*
