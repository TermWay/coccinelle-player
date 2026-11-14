@echo off
title Install Real-ESRGAN
setlocal
rem Installs Real-ESRGAN, the optional upscaler that redraws pictures (realesrgan-ncnn-vulkan, BSD 3-Clause, by
rem Xintao Wang). It downloads the program from its GitHub release v0.2.5.0, refuses the download unless it matches
rem the SHA-256 below, and unzips it into bin\realesrgan\ next to this script, where Coccinelle Player looks for it.
rem "install-realesrgan.bat nopause" closes without waiting for a key.
cd /d "%~dp0"

set "URL=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
set "SHA256=abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d"
set "ZIP=%TEMP%\realesrgan-ncnn-vulkan-20220424-windows.zip"
set "TARGET=bin\realesrgan"

if exist "%TARGET%\realesrgan-ncnn-vulkan.exe" (
    echo Real-ESRGAN is already installed in %TARGET%.
    goto done
)

echo Downloading Real-ESRGAN, 45 MB...
curl.exe --fail --location --progress-bar --output "%ZIP%" "%URL%"
if errorlevel 1 (
    echo The download failed. Check the internet connection and try again.
    goto failed
)

echo Checking the download...
set "DOWNLOADED="
for /f %%H in ('powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath $env:ZIP).Hash"') do set "DOWNLOADED=%%H"
if /i not "%DOWNLOADED%"=="%SHA256%" (
    echo The download is not the expected file, so it was not installed.
    echo     expected SHA-256 %SHA256%
    echo     downloaded       %DOWNLOADED%
    del "%ZIP%"
    goto failed
)

echo Unzipping into %TARGET%...
if not exist "%TARGET%" mkdir "%TARGET%"
tar -xf "%ZIP%" -C "%TARGET%" realesrgan-ncnn-vulkan.exe vcomp140.dll models README_windows.md
if errorlevel 1 (
    echo Could not unzip into %TARGET%.
    del "%ZIP%"
    goto failed
)
del "%ZIP%"
echo.
echo Real-ESRGAN is installed. Its upscalers appear in Coccinelle Player's list the next time it starts.

:done
if /i not "%~1"=="nopause" pause
exit /b 0

:failed
if /i not "%~1"=="nopause" pause
exit /b 1
