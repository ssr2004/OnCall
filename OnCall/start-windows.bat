@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

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
echo [1/9] Checking package manager...
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
echo [2/9] Configuring Python version...
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
echo [3/9] Preparing virtual environment...
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

REM Reuse fixed-name Milvus containers from an older Compose project when present.
REM This avoids container_name conflicts; the current project starts only Prometheus.
echo [4/9] Starting Milvus and Prometheus infrastructure...
set MILVUS_STACK_EXISTS=1
for %%c in (milvus-etcd milvus-minio milvus-standalone milvus-attu) do (
    docker container inspect %%c >nul 2>&1
    if errorlevel 1 set MILVUS_STACK_EXISTS=0
)

if "!MILVUS_STACK_EXISTS!"=="1" (
    echo [INFO] Existing Milvus containers detected; reusing them...
    docker start milvus-etcd milvus-minio milvus-standalone milvus-attu >nul
    if errorlevel 1 (
        echo [ERROR] Failed to start the existing Milvus containers.
        pause
        exit /b 1
    )
    echo [INFO] Starting only Prometheus with the current Compose project...
    docker compose -f vector-database.yml up -d prometheus
) else (
    echo [INFO] No complete existing Milvus stack found; creating the infrastructure with Compose...
    docker compose -f vector-database.yml up -d
)
if errorlevel 1 (
    echo [ERROR] Docker startup failed. Check Docker Desktop and image network access.
    pause
    exit /b 1
)
echo [INFO] Waiting 10 seconds for the infrastructure...
timeout /t 10 /nobreak >nul
echo [OK] Milvus and Prometheus infrastructure started.
echo.

REM Start the grid data collection and synchronization simulator.
echo [5/9] Starting grid service simulator...
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
echo [6/9] Starting CLS MCP server...
start "CLS MCP Server" /min %PYTHON_CMD% mcp_servers/cls_server.py
timeout /t 2 /nobreak >nul
echo [OK] CLS MCP server started.
echo.

REM Start the Monitor MCP server.
echo [7/9] Starting Monitor MCP server...
start "Monitor MCP Server" /min %PYTHON_CMD% mcp_servers/monitor_server.py
timeout /t 2 /nobreak >nul
echo [OK] Monitor MCP server started.
echo.

REM Start the FastAPI server.
echo [8/9] Starting FastAPI server...
start "SuperBizAgent API" %PYTHON_CMD% -m uvicorn app.main:app --host 0.0.0.0 --port 9900
echo [INFO] Waiting 15 seconds for the API...
timeout /t 15 /nobreak >nul
echo.

REM Check API status and upload runbooks.
echo.
echo [INFO] Checking API status...
curl -s http://localhost:9900/health >nul 2>&1
if errorlevel 1 (
    echo [WARN] The API may still be starting. Wait a little longer and check the process window.
) else (
    echo [OK] FastAPI server is healthy.
    echo.
    
    REM Upload aiops-docs files through the API.
    echo [9/9] Uploading runbooks to the vector database...
    for %%f in (aiops-docs\*.md) do (
        echo   Uploading: %%~nxf
        curl -s -X POST http://localhost:9900/api/upload -F "file=@%%f" >nul 2>&1
    )
    echo [OK] Runbook upload completed.
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
