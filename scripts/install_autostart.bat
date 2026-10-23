@echo off
chcp 65001 >nul
cd /d "%~dp0\.."

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_autostart.ps1"
pause
