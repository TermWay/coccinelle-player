@echo off
title FSR4 upscaler
cd /d "%~dp0"

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

if "%~1"=="" goto prompt
set "INPUT=%~1"
goto run

:prompt
echo FSR4 upscaler: drop a file on upscale.bat, or type a path below.
set /p "INPUT=Image or video to upscale: "
if "%INPUT%"=="" exit /b 0
set "INPUT=%INPUT:"=%"

:run
python -m fsr4 upscale "%INPUT%" %2 %3 %4 %5 %6 %7 %8 %9
if errorlevel 1 (
    echo.
    echo Upscaling failed, see the messages above.
)
pause
