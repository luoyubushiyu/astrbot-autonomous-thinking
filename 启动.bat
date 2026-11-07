@echo off
setlocal
cd /d "%~dp0"
title Autonomous Agent

set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY python --version >nul 2>nul && set "PY=python"
if not defined PY (
  echo.
  echo   [!] Python not found.
  echo       Install Python 3.11 or newer: https://www.python.org/downloads/
  echo       Remember to tick "Add python.exe to PATH" during setup.
  echo.
  pause
  exit /b 1
)

if not exist "config.json" (
  copy /y "config.example.json" "config.json" >nul
  echo   [i] Created config.json from the example. Default is mock mode: offline, no API key.
)

echo   [i] Starting autonomous agent. Press Ctrl+C to stop.
echo.
%PY% -u -m autonomous_agent
echo.
echo   [i] Exited with code %errorlevel%
pause
