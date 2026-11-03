@echo off
title Build exe
setlocal
rem Builds dist\fsr4: "Coccinelle Player.exe" and fsr4.exe, the FSR DLLs and their license in bin\, the
rem third-party notices in licenses\, README.md and LICENSE.txt. Optional arguments, to build elsewhere: the
rem folder that receives fsr4\ (default dist) and PyInstaller's work folder (default build).
set "DIST_ROOT=dist"
set "WORK_ROOT=build"
if not "%~1"=="" set "DIST_ROOT=%~f1"
if not "%~2"=="" set "WORK_ROOT=%~f2"
cd /d "%~dp0"
set "DIST=%DIST_ROOT%\fsr4"

where python >nul 2>nul
if errorlevel 1 (
    echo Python 3.12 is not installed or not on PATH.
    pause
    exit /b 1
)

echo Checking the FSR DLLs in bin\ against their pinned SHA-256 ...
python tools\fetch_sdk.py --check
if errorlevel 1 (
    echo Downloading them again from AMD's FidelityFX SDK on GitHub ...
    python tools\fetch_sdk.py
    if errorlevel 1 (
        echo The FSR DLLs could not be downloaded and verified.
        pause
        exit /b 1
    )
)

echo Building "%DIST%" with PyInstaller ...
python -m PyInstaller --noconfirm --clean --distpath "%DIST_ROOT%" --workpath "%WORK_ROOT%" fsr4.spec
if errorlevel 1 (
    echo PyInstaller failed, see the messages above.
    pause
    exit /b 1
)

python tools\collect_licenses.py "%DIST%"
if errorlevel 1 (
    echo The third-party notices could not be written, see the message above.
    pause
    exit /b 1
)

if not exist "%DIST%\bin" mkdir "%DIST%\bin"
for %%F in (amd_fidelityfx_loader_vk.dll amd_fidelityfx_upscaler_vk.dll license.md) do (
    copy /y "bin\%%F" "%DIST%\bin\%%F" >nul
    if errorlevel 1 (
        echo Could not copy bin\%%F into "%DIST%\bin".
        pause
        exit /b 1
    )
)
copy /y "README.md" "%DIST%\README.md" >nul
if errorlevel 1 (
    echo Could not copy README.md into "%DIST%".
    pause
    exit /b 1
)
copy /y "LICENSE" "%DIST%\LICENSE.txt" >nul
if errorlevel 1 (
    echo Could not copy LICENSE into "%DIST%".
    pause
    exit /b 1
)

echo.
echo Done: "%DIST%\Coccinelle Player.exe"  (ffmpeg.exe and ffprobe.exe must be on PATH or copied next to it)
pause
