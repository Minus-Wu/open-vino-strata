@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PYTHON_EXE=.\.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    echo [Notice] 找不到 .venv 目錄，將使用系統 Python...
    set PYTHON_EXE=python
)

echo =====================================================================
echo      OpenVINO-Strata: 單一執行檔自動化打包 (PyInstaller One-File)    
echo =====================================================================

"%PYTHON_EXE%" scripts\build_package.py %*
if "%1" neq "--no-pause" pause
