@echo off
:: Open a .ply gaussian splat file in the show_gs.py web viewer.
::
:: Thin wrapper around `run_example.bat --module viewer` that just pins
:: HY_VIEW_PLY to whatever you pass. Everything else (position_meta_info.json
:: synthesis, browser auto-open, viewer port) is handled there so behavior
:: stays in one place.
::
:: Usage:
::   view_splat.bat                                 latest gaussians.ply under output\
::   view_splat.bat path\to\file.ply                specific file
::   view_splat.bat path\to\file.ply 8090           pin port (default 8081)

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

if not "%~1"=="" (
    if not exist "%~1" (
        echo ERROR: file not found: %~1
        exit /b 2
    )
    set "HY_VIEW_PLY=%~1"
)
if not "%~2"=="" set "HY_VIEW_PORT=%~2"

call "%~dp0run_example.bat" --module viewer
exit /b %ERRORLEVEL%
