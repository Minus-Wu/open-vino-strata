# OpenVINO-Strata API Server Launcher (PowerShell)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[Notice] .venv not found in project folder, falling back to system python..." -ForegroundColor Yellow
    $PythonExe = "python"
}

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  OpenVINO-Strata LLM Server (Intel OpenVINO 2026 + PolyStrata MoE)  " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

& $PythonExe (Join-Path $ScriptDir "run_server.py") @args
