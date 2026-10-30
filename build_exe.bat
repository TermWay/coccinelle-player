@echo off
title Build exe
cd /d "%~dp0"
echo Building dist\fsr4\fsr4.exe with PyInstaller ...
python -m PyInstaller --noconfirm --clean --onedir --name fsr4 --console ^
    --collect-binaries cv2 --exclude-module torch --exclude-module torchvision ^
    --exclude-module matplotlib --exclude-module PIL ^
    run_fsr4.py
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)
if not exist "bin\amd_fidelityfx_upscaler_vk.dll" python tools\fetch_sdk.py
if not exist "dist\fsr4\bin" mkdir "dist\fsr4\bin"
copy /y "bin\amd_fidelityfx_loader_vk.dll" "dist\fsr4\bin\" >nul
copy /y "bin\amd_fidelityfx_upscaler_vk.dll" "dist\fsr4\bin\" >nul
copy /y "bin\license.md" "dist\fsr4\bin\" >nul
copy /y "README.md" "dist\fsr4\" >nul
echo.
echo Done: dist\fsr4\fsr4.exe  (ffmpeg.exe and ffprobe.exe must be on PATH or copied next to it)
pause
