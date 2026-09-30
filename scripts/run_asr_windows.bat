@echo off
title Nexus Edge Voice Activator - Windows Gateway
echo ============================================================
echo   NEXUS : Low-Latency Edge Voice Activator (Windows Gateway)
echo ============================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH!
    echo Please install Python 3.10+ from python.org and check "Add Python to PATH".
    pause
    exit /b 1
)

echo [1/2] Verifying dependencies (pyserial, vosk)...
python -c "import serial, vosk" >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing required packages: pyserial, vosk...
    pip install pyserial vosk
)

echo [2/2] Launching Gateway Server...
echo.
python scripts\serial_asr_server_windows.py %*

if %errorlevel% neq 0 (
    echo.
    echo Server exited with an error.
    pause
)
