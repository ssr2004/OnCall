@echo off
chcp 65001 >nul
cd /d "%~dp0"

if /I "%~1"=="--encoding-check" (
    echo STOP_SCRIPT_CHECK_OK
    exit /b 0
)

set "STOP_ERRORS=0"

echo ====================================
echo Stopping SuperBizAgent services
echo ====================================
echo.

REM Stop the FastAPI server.
echo [1/6] Stopping FastAPI server...
taskkill /FI "WINDOWTITLE eq SuperBizAgent API*" /T /F >nul 2>&1
if errorlevel 1 (
    echo [INFO] FastAPI server is not running or is already stopped.
) else (
    echo [OK] FastAPI server stopped.
)
echo.

REM Stop the CLS MCP server.
echo [2/6] Stopping CLS MCP server...
taskkill /FI "WINDOWTITLE eq CLS MCP Server*" /T /F >nul 2>&1
if errorlevel 1 (
    echo [INFO] CLS MCP server is not running or is already stopped.
) else (
    echo [OK] CLS MCP server stopped.
)
echo.

REM Stop the Monitor MCP server.
echo [3/6] Stopping Monitor MCP server...
taskkill /FI "WINDOWTITLE eq Monitor MCP Server*" /T /F >nul 2>&1
if errorlevel 1 (
    echo [INFO] Monitor MCP server is not running or is already stopped.
) else (
    echo [OK] Monitor MCP server stopped.
)
echo.

REM Stop the grid simulator.
echo [4/6] Stopping grid simulator...
taskkill /FI "WINDOWTITLE eq Grid Service Simulator*" /T /F >nul 2>&1
if errorlevel 1 (
    echo [INFO] Grid simulator is not running or is already stopped.
) else (
    echo [OK] Grid simulator stopped.
)
echo.

REM Clean up project-owned orphan processes and verify that service ports are free.
echo [5/6] Verifying project service ports...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop-project-processes.ps1" -ProjectRoot "%~dp0"
if errorlevel 1 (
    echo [WARN] Some service ports are still occupied. Review the messages above.
    set "STOP_ERRORS=1"
) else (
    echo [OK] Project service ports are free.
)
echo.

REM Stop the current project's Docker infrastructure.
echo [6/6] Stopping Docker infrastructure...
docker ps --format "{{.Names}}" | findstr "milvus" >nul 2>&1
if not errorlevel 1 (
    docker compose -f vector-database.yml down
    if errorlevel 1 (
        echo [ERROR] Failed to stop Docker containers.
        set "STOP_ERRORS=1"
    ) else (
        echo [OK] Docker infrastructure stopped.
    )
) else (
    echo [INFO] Milvus containers are not running.
)
echo.

echo ====================================
if "%STOP_ERRORS%"=="0" (
    echo All project services have been stopped.
) else (
    echo Stop completed with warnings. Review the messages above.
)
echo ====================================
echo.
echo Optional cleanup command for project-owned Docker volumes:
echo     docker compose -f vector-database.yml down -v
echo.
pause
exit /b %STOP_ERRORS%
