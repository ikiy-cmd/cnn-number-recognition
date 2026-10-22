#requires -Version 5.1
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
    & $python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    & $python -m pip install -r requirements.txt
}

if (-not (Test-Path -LiteralPath "artifacts\model.pt")) {
    Write-Host "Downloading MNIST and training the first model..." -ForegroundColor Cyan
    & $python -m app.mnist
    & $python -m app.train --epochs 8 --train-size 55000
}

try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 2
    Write-Host "Server is already running at http://127.0.0.1:8000" -ForegroundColor Green
    Start-Process "http://127.0.0.1:8000"
    exit 0
} catch {
    # no server yet
}

Write-Host "Layer Trace is starting at http://127.0.0.1:8000" -ForegroundColor Green
Start-Process powershell -WindowStyle Hidden -ArgumentList @(
    "-NoProfile",
    "-Command",
    "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:8000'"
)

& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
