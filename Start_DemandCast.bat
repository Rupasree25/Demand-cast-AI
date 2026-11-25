
@echo off
title DemandCast AI Automatic Setup

echo =====================================
echo       DEMANDCAST AI SETUP
echo =====================================

set "PYTHON=%LocalAppData%\Programs\Python\Python311\python.exe"
set "INSTALLER=%TEMP%\python-3.11.9-amd64.exe"

if not exist "%PYTHON%" (
    echo Python is not installed. Installing Python...
    curl -L "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -o "%INSTALLER%"
    if not exist "%INSTALLER%" (
        echo Python download failed.
        pause
        exit /b 1
    )
    "%INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
)

if not exist "%PYTHON%" (
    echo Python installation was unsuccessful.
    pause
    exit /b 1
)

echo Python is ready!
echo Starting DemandCast AI...

cd /d "%~dp0"
"%PYTHON%" demandcast_setup.py

pause
