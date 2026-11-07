@echo off
setlocal
cd /d "%~dp0"
title 自主思考 Agent

set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY python --version >nul 2>nul && set "PY=python"
if not defined PY (
  echo.
  echo   [!] 没找到 Python。
  echo       去 https://www.python.org/downloads/ 装 3.11 或更新的版本，
  echo       安装时记得勾上 "Add python.exe to PATH"。
  echo.
  pause
  exit /b 1
)

if not exist "config.json" (
  copy /y "config.example.json" "config.json" >nul
  echo   [i] 已从示例生成 config.json（默认 mock 模式：不联网、不用 API key）
)

echo   [i] 启动中，按 Ctrl+C 停止
echo.
%PY% -u -m autonomous_agent
echo.
echo   [i] 已退出，代码 %errorlevel%
pause
