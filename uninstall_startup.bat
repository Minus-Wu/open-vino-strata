@echo off
chcp 65001 >nul
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\uninstall_autostart.ps1"
if "%1" neq "--no-pause" pause
