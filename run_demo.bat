@echo off
title ATTUNE Demo Launcher
echo ====================================================================
echo    ATTUNE: Context-Aware Preference Gating Engine (Phase 5 Demo)   
echo ====================================================================
echo.

cd /d "%~dp0"

echo [INFO] Starting ATTUNE server...
start "ATTUNE Server" cmd /k "python app/server.py"

echo [INFO] Waiting for server initialization...
timeout /t 3 /nobreak >nul

echo [INFO] Opening browser at http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"

echo.
echo ATTUNE Demonstration Dashboard launched successfully!
echo Press any key to exit this launcher window (server will remain running).
pause >nul
