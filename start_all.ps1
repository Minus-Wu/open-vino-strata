# OpenVINO-Strata All-in-One Launcher (PowerShell)
# Starts background OpenVINO-Strata API Server (Intel GPU) and launches the Desktop GUI

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[Notice] .venv not found in project folder, falling back to system python..." -ForegroundColor Yellow
    $PythonExe = "python"
}

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "     OpenVINO-Strata: Low-VRAM 35B-A3B Server & Desktop Studio       " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# 1. Start Server in background
Write-Host "[*] Step 1: Starting OpenVINO-Strata API Server (GPU) in background..." -ForegroundColor Green
$ServerProc = Start-Process -FilePath $PythonExe -ArgumentList "run_server.py --device GPU --port 12223" -WorkingDirectory $ScriptDir -WindowStyle Minimized -PassThru

# 2. Poll /health until ready
Write-Host "[*] Step 2: Waiting for OpenVINO-Strata server to initialize..." -ForegroundColor Yellow
$maxRetries = 25
for ($i = 0; $i -lt $maxRetries; $i++) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:12223/health" -UseBasicParsing -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($resp -and $resp.StatusCode -eq 200) {
            Write-Host "[*] Server is healthy and ready on http://127.0.0.1:12223 !" -ForegroundColor Green
            break
        }
    } catch {
        # Keep waiting while model compiles/loads
    }
}

# 3. Launch Desktop GUI in foreground
Write-Host "[*] Step 3: Launching OpenVINO-Strata Desktop Chat GUI..." -ForegroundColor Green
& $PythonExe (Join-Path $ScriptDir "run_gui.py") --url "http://127.0.0.1:12223" @args

# 4. Clean up background server when GUI exits
Write-Host "[*] Desktop GUI closed. Releasing VRAM & stopping background server..." -ForegroundColor Yellow
if ($ServerProc -and -not $ServerProc.HasExited) {
    Stop-Process -Id $ServerProc.Id -Force -ErrorAction SilentlyContinue
}
Write-Host "[OK] All OpenVINO-Strata services stopped cleanly." -ForegroundColor Cyan
