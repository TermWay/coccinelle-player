@echo off
title FSR4 upscaler
setlocal
rem Opens the window, with the file dropped on this script (if any) open in it. The full path is taken before
rem changing folder, so a relative path given on a command line works too.
set "INPUT=%~f1"
cd /d "%~dp0"

if exist "dist\fsr4\FSR4 Upscaler.exe" (
    if defined INPUT (
        start "" "dist\fsr4\FSR4 Upscaler.exe" "%INPUT%"
    ) else (
        start "" "dist\fsr4\FSR4 Upscaler.exe"
    )
    exit /b 0
)

where python >nul 2>nul
if errorlevel 1 (
    echo Python 3.12 is not installed or not on PATH.
    pause
    exit /b 1
)

python tools\fetch_sdk.py --check >nul
if errorlevel 1 (
    echo.
    echo The FSR DLLs in bin\ are missing or damaged. Downloading AMD's signed FSR 4 DLLs and their license
    echo into bin\ from AMD's FidelityFX SDK on GitHub, at the tag pinned in tools\fetch_sdk.py:
    python tools\fetch_sdk.py
    if errorlevel 1 (
        echo Download failed.
        pause
        exit /b 1
    )
)

if defined INPUT (
    python -m fsr4 ui "%INPUT%"
) else (
    python -m fsr4 ui
)
if errorlevel 1 (
    echo.
    echo The window closed with an error, see the messages above. If a package is missing:
    echo     python -m pip install -r requirements.txt
    pause
)
