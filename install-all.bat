@echo off
title Install everything
setlocal
rem Installs what Coccinelle Player uses but does not include: ffmpeg, which it needs, then Real-ESRGAN, an
rem optional upscaler. Each step skips what is already installed.

echo === 1 of 2: ffmpeg (needed) ===
call "%~dp0install-ffmpeg.bat" nopause
set "FFMPEG_FAILED=%errorlevel%"
echo.

echo === 2 of 2: Real-ESRGAN (optional) ===
call "%~dp0install-realesrgan.bat" nopause
set "REALESRGAN_FAILED=%errorlevel%"
echo.

title Install everything
if not "%FFMPEG_FAILED%"=="0" (
    echo ffmpeg is NOT installed, and Coccinelle Player cannot open files without it. See the messages above.
) else if not "%REALESRGAN_FAILED%"=="0" (
    echo ffmpeg is ready. Real-ESRGAN is not installed; the player works without it.
) else (
    echo Everything is installed. Start Coccinelle Player.exe.
)
pause
