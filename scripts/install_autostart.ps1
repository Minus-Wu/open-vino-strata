# OpenVINO-Strata Windows Startup Registration (PowerShell)
# Configures OpenVINO-Strata to launch automatically in background on Windows login

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "     OpenVINO-Strata: Windows Login Auto-Start Installation         " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

& $PythonExe (Join-Path $ProjectRoot "run_tray.py") --install-startup

Write-Host "[OK] Configuration completed. OpenVINO-Strata will now start silently in background upon user login." -ForegroundColor Green
