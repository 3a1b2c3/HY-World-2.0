@echo off
:: Launch the minimal OpenAI-compat Qwen3-VL server (vLLM stand-in).
:: Run this in a SEPARATE shell, then run run_example.bat --module worldgen here.
setlocal enableextensions
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python vlm_server.py %*
