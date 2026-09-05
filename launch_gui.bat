@echo off
REM Launch the StructureForge web GUI (Windows).
REM Opens http://127.0.0.1:8000 in the default browser once the server is ready.

cd /d "%~dp0"

REM --- ensure the [api] extras are installed ---------------------------------
python -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [structureforge] installing API dependencies...
    pip install -e ".[api]" --quiet
)

REM --- open browser after a short delay --------------------------------------
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000"

echo [structureforge] starting server at http://127.0.0.1:8000  (Ctrl+C to stop)
python -m structureforge.api.cli %*
