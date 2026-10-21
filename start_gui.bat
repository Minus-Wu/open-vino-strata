@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [Notice] .venv not found, falling back to system python...
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" "%~dp0run_gui.py" %*
