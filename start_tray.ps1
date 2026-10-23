# OpenVINO-Strata Windows System Tray Launcher (PowerShell)
# Starts OpenVINO-Strata running in the Windows System Tray

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[Notice] .venv not found in project folder, falling back to system python..." -ForegroundColor Yellow
    $PythonExe = "python"
}

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "     OpenVINO-Strata: Windows System Tray Resident Mode             " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "[*] Launching OpenVINO-Strata in System Tray mode..." -ForegroundColor Green
Write-Host "[*] Tip: Click the tray icon to open Chat UI, or right-click to switch models / unload VRAM." -ForegroundColor Yellow

& $PythonExe (Join-Path $ScriptDir "run_tray.py") @args
