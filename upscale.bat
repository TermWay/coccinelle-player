@echo off
title Upscale a file
rem Drag and drop images or videos onto this file: each is upscaled 2x right away, no questions asked.
rem For other settings run `fsr4 upscale` (see docs\command-line.md).
setlocal
set "HERE=%~dp0"
if "%~1"=="" (
    echo Drop an image or a video onto this file.
    pause
    exit /b 1
)
set "FAILED="

:next
if "%~1"=="" goto done
rem The full path is taken before changing folder, so a relative path given on a command line works too.
set "INPUT=%~f1"
echo Upscaling "%INPUT%"
pushd "%HERE%"
if exist "dist\fsr4\fsr4.exe" (
    "dist\fsr4\fsr4.exe" upscale "%INPUT%" || set "FAILED=1"
) else (
    python -m fsr4 upscale "%INPUT%" || set "FAILED=1"
)
popd
shift
goto next

:done
if defined FAILED (
    echo.
    echo Upscaling failed, see the messages above.
)
pause
if defined FAILED exit /b 1
