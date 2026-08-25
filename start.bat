@echo off
title DeskSonar - Ultrasonic Acoustic Radar
echo ========================================================
echo   Launching DeskSonar Acoustic Radar Engine
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)

python -m src.cli run --port 8765
pause
