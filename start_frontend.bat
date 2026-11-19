@echo off
title BoneTest Frontend Dev Server (Vite - Port 5173)
cd /d "%~dp0frontend"

if exist "C:\Users\Manya\nodejs\node-v22.13.1-win-x64\node.exe" (
    set "PATH=C:\Users\Manya\nodejs\node-v22.13.1-win-x64;%PATH%"
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js / npm not found in PATH.
    pause
    exit /b 1
)

echo ============================================================================
echo Starting BoneTest React/Vite Frontend Dev Server...
echo   - Frontend App URL  : http://localhost:5173
echo   - Proxying /api to  : http://localhost:8000
echo ============================================================================
echo.

call npm.cmd run dev
