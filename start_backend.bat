@echo off
title BoneTest Backend Server (FastAPI - Port 8000)
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found at "%~dp0.venv"
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo ============================================================================
echo Starting BoneTest FastAPI Backend Server...
echo   - API Base URL      : http://localhost:8000
echo   - Swagger Docs UI   : http://localhost:8000/docs
echo   - Health Endpoint   : http://localhost:8000/api/health
echo ============================================================================
echo.

python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
