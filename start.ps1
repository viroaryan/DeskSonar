# DeskSonar PowerShell Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Launching DeskSonar Acoustic Radar Engine" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment and installing dependencies..." -ForegroundColor Yellow
    python -m venv .venv
    .\.venv\Scripts\pip.exe install -r requirements.txt
}

$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m src.cli run --port 8765
