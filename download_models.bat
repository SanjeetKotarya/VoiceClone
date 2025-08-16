@echo off
echo Starting Model Download...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo Python found, downloading models...
python download_models.py

echo.
echo Model download completed!
pause

