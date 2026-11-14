@echo off
title Install ffmpeg
setlocal
rem Installs ffmpeg and ffprobe, which Coccinelle Player needs to read and write every video and image. It uses
rem winget, Windows' own package manager, to install the "essentials" build from gyan.dev, and does nothing when
rem ffmpeg is already there. "install-ffmpeg.bat nopause" closes without waiting for a key.
cd /d "%~dp0"

if exist "ffmpeg.exe" (
    echo ffmpeg is already here, next to this script:
    ffmpeg.exe -hide_banner -version | findstr /b "ffmpeg"
    goto done
)
where ffmpeg >nul 2>nul
if not errorlevel 1 (
    echo ffmpeg is already installed:
    ffmpeg -hide_banner -version | findstr /b "ffmpeg"
    goto done
)

where winget >nul 2>nul
if errorlevel 1 (
    echo winget is not on this computer. Either install "App Installer" from the Microsoft Store and run this
    echo script again, or download ffmpeg from https://www.gyan.dev/ffmpeg/builds/ and copy ffmpeg.exe and
    echo ffprobe.exe next to this script.
    goto failed
)

echo Installing ffmpeg with winget...
winget install --id Gyan.FFmpeg.Essentials --exact --source winget
if errorlevel 1 (
    echo winget could not install ffmpeg, see the messages above.
    goto failed
)
echo.
echo ffmpeg is installed. Programs started from now on find it; a terminal window already open does not.

:done
if /i not "%~1"=="nopause" pause
exit /b 0

:failed
if /i not "%~1"=="nopause" pause
exit /b 1
