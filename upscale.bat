@echo off
title Upscale a file
rem Drag and drop an image or a video onto this file: it is upscaled 2x right away, no questions asked.
rem Extra arguments are passed to `fsr4 upscale` (see README).
cd /d "%~dp0"
if "%~1"=="" (
    echo Drop an image or a video onto this file.
    pause
    exit /b 1
)
if exist "dist\fsr4\fsr4.exe" (
    "dist\fsr4\fsr4.exe" upscale %*
) else (
    python -m fsr4 upscale %*
)
if errorlevel 1 (
    echo.
    echo Upscaling failed, see the messages above.
)
pause
