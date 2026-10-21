# OpenVINO-Strata Desktop GUI Launcher (PowerShell)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[Notice] .venv not found in project folder, falling back to system python..." -ForegroundColor Yellow
    $PythonExe = "python"
}

& $PythonExe (Join-Path $ScriptDir "run_gui.py") @args
