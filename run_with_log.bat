@echo off
:: Wrapper around run_example.bat that captures all stdout+stderr to a
:: timestamped log file using cmd's native redirect ('>'). This grabs output
:: from the ENTIRE process tree (torchrun children, VLM server, etc.) — a
:: PowerShell-Tee pipe only captures direct stdout and misses subprocesses.
::
:: Trade-off: terminal stays mostly quiet during the run (only stderr line
:: buffering may sneak through). The log is the authoritative output.
::
:: Usage:
::   run_with_log.bat                       all modules
::   run_with_log.bat --module worldgen
::   run_with_log.bat --module viewer
::
:: To watch the log live IN ANOTHER terminal:
::   Get-Content C:\workspace\world\HY-World-2.0\logs\run_LATEST.log -Wait -Tail 50

setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"
if not exist "logs" mkdir "logs"

:: yyyymmdd_hhmmss timestamp (locale-independent via wmic)
for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value ^| find "="') do set "LDT=%%a"
set "LOG=%~dp0logs\run_!LDT:~0,8!_!LDT:~8,6!.log"

echo ============================================================
echo run_with_log.bat
echo   args : %*
echo   log  : !LOG!
echo.
echo Tail it in another shell:
echo   Get-Content "!LOG!" -Wait -Tail 50
echo ============================================================
echo.

:: cmd /c with > captures stdout from every child process the bat spawns,
:: including torchrun-launched workers. Errors land in the same file because
:: 2>&1 merges streams before the redirect.
cmd /c ""%~dp0run_example.bat" %* >"!LOG!" 2>&1"
set "RC=!ERRORLEVEL!"

echo ============================================================
echo run_with_log.bat finished
echo   rc  : !RC!
echo   log : !LOG!
echo.
echo Inspect the tail:
echo   Get-Content "!LOG!" -Tail 80
echo.
echo --- result artifacts (under output\) ---
:: Show the most-recently-modified gaussians.ply (what the viewer would pick
:: up) plus the dir it lives in. PowerShell is the cleanest way to mtime-sort
:: a recursive walk on Windows.
for /f "delims=" %%P in ('powershell -NoProfile -Command "Get-ChildItem -Path '%~dp0output' -Recurse -Filter gaussians.ply -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set "LATEST_PLY=%%P"
if defined LATEST_PLY (
    for %%I in ("!LATEST_PLY!") do (
        echo   latest .ply : !LATEST_PLY!
        echo   result dir  : %%~dpI
    )
    echo.
    echo View it:  .\run_example.bat --module viewer
) else (
    echo   no gaussians.ply found under %~dp0output\
)
echo ============================================================
exit /b !RC!
