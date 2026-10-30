@echo off
title FSR4 upscaler
cd /d "%~dp0"

if exist "dist\fsr4\FSR4 Upscaler.exe" (
    start "" "dist\fsr4\FSR4 Upscaler.exe" %1
    exit /b 0
)

where python >nul 2>nul
if errorlevel 1 (
    echo Python 3.12 is not installed or not on PATH.
    pause
    exit /b 1
)

if not exist "bin\amd_fidelityfx_upscaler_vk.dll" (
    echo The FSR SDK DLLs are missing. Downloading them into bin\ ...
    python tools\fetch_sdk.py
    if errorlevel 1 (
        echo Download failed.
        pause
        exit /b 1
    )
)

python -m fsr4 ui %1
if errorlevel 1 (
    echo.
    echo The window closed with an error, see the messages above.
    pause
)
