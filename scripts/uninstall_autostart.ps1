# OpenVINO-Strata Windows Startup Unregistration (PowerShell)
# Removes OpenVINO-Strata from Windows login auto-start

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "     OpenVINO-Strata: Remove Windows Login Auto-Start               " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

& $PythonExe (Join-Path $ProjectRoot "run_tray.py") --uninstall-startup

Write-Host "[OK] Auto-start registration has been removed." -ForegroundColor Green
