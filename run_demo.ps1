# ==============================================================================
# ATTUNE: Context-Aware Preference Gating Engine - Demo Launcher (PowerShell)
# ==============================================================================

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "   ATTUNE: Context-Aware Preference Gating Engine (Phase 5 Demo)   " -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

$RootPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootPath

# Test if port 8000 is already active
$PortInUse = $false
try {
    $TcpClient = New-Object System.Net.Sockets.TcpClient
    $TcpClient.Connect("127.0.0.1", 8000)
    $TcpClient.Close()
    $PortInUse = $true
} catch {
    $PortInUse = $false
}

if ($PortInUse) {
    Write-Host "[INFO] ATTUNE Server is already running on port 8000." -ForegroundColor Green
} else {
    Write-Host "[INFO] Launching ATTUNE FastAPI Server on http://127.0.0.1:8000 ..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "python app/server.py" -WindowStyle Normal
    Start-Sleep -Seconds 3
}

Write-Host "[INFO] Opening demonstration dashboard in default browser..." -ForegroundColor Green
Start-Process "http://127.0.0.1:8000"

Write-Host ""
Write-Host "Interactive UI available at: http://127.0.0.1:8000" -ForegroundColor White
Write-Host "To stop the server, close the spawned PowerShell window or kill the python process." -ForegroundColor DarkGray
Write-Host ""
