@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "COMPOSE_PROJECT_NAME=oncall"

if /I "%~1"=="--encoding-check" (
    echo START_SCRIPT_CHECK_OK
    exit /b 0
)

echo ====================================
echo Starting SuperBizAgent services
echo ====================================
echo.

REM Fail fast when the Docker CLI or Docker Desktop engine is unavailable.
where docker >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker CLI was not found. Install Docker Desktop first.
    exit /b 1
)
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Desktop engine is not running.
    echo [TIP] Start Docker Desktop and wait until the engine is ready, then run this script again.
    exit /b 1
)
if /I "%~1"=="--preflight-check" (
    echo START_PREFLIGHT_CHECK_OK
    exit /b 0
)

REM Check for uv; fall back to pip when it is unavailable.
echo [1/8] Checking package manager...
where uv >nul 2>&1
if errorlevel 1 (
    echo [INFO] uv was not found; pip will be used.
    echo [TIP] Install uv for faster setup: pip install uv
    set USE_UV=0
) else (
    echo [OK] uv package manager detected.
    set USE_UV=1
)
echo.

REM Ensure a compatible Python version is configured.
echo [2/8] Configuring Python version...
if exist .python-version (
    set /p PYTHON_VERSION=<.python-version
    echo [INFO] Configured version: !PYTHON_VERSION!
    
    REM Python 3.10 is not supported by this project.
    echo !PYTHON_VERSION! | findstr /C:"3.10" >nul
    if not errorlevel 1 (
        echo [WARN] Python 3.10 is unsupported; changing configuration to 3.13...
        echo 3.13> .python-version
        echo [OK] Python configuration updated to 3.13.
    )
) else (
    echo [INFO] Creating .python-version...
    echo 3.13> .python-version
)
echo.

REM Create or synchronize the virtual environment.
echo [3/8] Preparing virtual environment...
if exist .venv\Scripts\python.exe (
    echo [INFO] Existing virtual environment found; checking dependencies...
    
    REM Prefer uv sync when uv is available.
    if "%USE_UV%"=="1" (
        uv sync 2>nul
        if errorlevel 1 (
            echo [WARN] uv sync failed; updating with pip...
            .venv\Scripts\python.exe -m pip install -e . -q
        ) else (
            echo [OK] Dependencies synchronized with uv.
        )
    ) else (
        echo [INFO] Updating dependencies with pip...
        .venv\Scripts\python.exe -m pip install -e . -q
    )
) else (
    echo [INFO] Creating a new virtual environment...
    
    REM Prefer uv sync when uv is available.
    if "%USE_UV%"=="1" (
        echo [INFO] Trying uv sync...
        uv sync 2>nul
        if not errorlevel 1 (
            echo [OK] Virtual environment created with uv.
            goto :venv_created
        )
        echo [WARN] uv sync failed; falling back to Python venv...
    )
    
    REM Create the environment with the standard venv module.
    echo [INFO] Running python -m venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        echo [TIP] Make sure Python 3.11 or newer is installed.
        pause
        exit /b 1
    )
    
    REM Install project dependencies.
    echo [INFO] Installing project dependencies; this may take a few minutes...
    .venv\Scripts\python.exe -m pip install --upgrade pip -q
    .venv\Scripts\python.exe -m pip install -e . -q
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

:venv_created
echo [OK] Virtual environment is ready.
echo.

REM Configure the project Python executable.
set PYTHON_CMD=.venv\Scripts\python.exe

REM Start only the infrastructure owned by this Compose project.
REM Compose will reject unrelated containers that reuse the same fixed names.
echo [4/8] Starting Milvus and Prometheus infrastructure...
docker network inspect milvus >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating the shared Milvus Docker network...
    docker network create milvus >nul
    if errorlevel 1 (
        echo [ERROR] Failed to create the Milvus Docker network.
        pause
        exit /b 1
    )
)
docker compose -f vector-database.yml up -d
if errorlevel 1 (
    echo [ERROR] Docker startup failed.
    echo [TIP] Check Docker Desktop, image access, and fixed-name container conflicts.
    pause
    exit /b 1
)
echo [INFO] Waiting for etcd, MinIO, Milvus, and Prometheus health checks...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\wait-docker-infrastructure.ps1" -ComposeProject "%COMPOSE_PROJECT_NAME%"
if errorlevel 1 (
    echo [ERROR] Docker infrastructure did not become healthy.
    echo [TIP] Run: docker compose -f vector-database.yml ps
    pause
    exit /b 1
)
echo [OK] Milvus and Prometheus infrastructure is healthy.
echo.

REM Start the grid data collection and synchronization simulator.
echo [5/8] Starting grid service simulator...
start "Grid Service Simulator" /min %PYTHON_CMD% -m uvicorn grid_simulator.service:app --host 0.0.0.0 --port 9105
timeout /t 3 /nobreak >nul
curl -s -f http://localhost:9105/health >nul 2>&1
if errorlevel 1 (
    echo [WARN] Grid simulator is not ready yet; check its process window.
) else (
    echo [OK] Grid simulator started.
)
echo.

REM Start the CLS MCP server.
echo [6/8] Starting CLS MCP server...
start "CLS MCP Server" /min %PYTHON_CMD% mcp_servers/cls_server.py
timeout /t 2 /nobreak >nul
echo [OK] CLS MCP server started.
echo.

REM Start the Monitor MCP server.
echo [7/8] Starting Monitor MCP server...
start "Monitor MCP Server" /min %PYTHON_CMD% mcp_servers/monitor_server.py
timeout /t 2 /nobreak >nul
echo [OK] Monitor MCP server started.
echo.

REM Start the FastAPI server.
echo [8/8] Starting FastAPI server...
curl -s -f http://localhost:9900/health >nul 2>&1
if errorlevel 1 (
    start "SuperBizAgent API" %PYTHON_CMD% -m uvicorn app.main:app --host 0.0.0.0 --port 9900
) else (
    echo [INFO] FastAPI is already running; reusing the healthy process.
)
echo [INFO] Waiting for the FastAPI health endpoint...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\wait-http-endpoint.ps1" -Uri "http://localhost:9900/health" -ServiceName "FastAPI" -TimeoutSeconds 60
if errorlevel 1 (
    echo [ERROR] FastAPI failed to become healthy. Review logs\app_*.log.
    pause
    exit /b 1
)

REM Check API status. Knowledge documents persist in Milvus and are uploaded from the Web UI.
echo.
echo [INFO] Checking API status...
curl -s http://localhost:9900/health >nul 2>&1
if errorlevel 1 (
    echo [ERROR] FastAPI health verification failed unexpectedly.
    pause
    exit /b 1
) else (
    echo [OK] FastAPI server is healthy.
    echo.
    echo [INFO] Knowledge documents are persisted in Milvus.
    echo [INFO] Upload new or updated .md/.txt files from the Web UI when needed.
)

echo.
echo ====================================
echo Service startup completed.
echo ====================================
echo Web UI: http://localhost:9900
echo API docs: http://localhost:9900/docs
echo Grid simulator: http://localhost:9105
echo Grid control: http://localhost:9105/control
echo Scenario status: http://localhost:9105/api/status
echo Prometheus: http://localhost:9090
echo Prometheus Targets: http://localhost:9090/targets
echo.
echo Logs:
echo   - FastAPI: logs\app_*.log
echo   - CLS MCP: type mcp_cls.log
echo   - Monitor: type mcp_monitor.log
echo Stop services: stop-windows.bat
echo ====================================
pause
