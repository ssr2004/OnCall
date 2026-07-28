"""面向甲方演示的电网数据采集与同步服务模拟器。

该服务只模拟只读的数据采集、处理和同步链路，不包含任何真实电网控制能力。
Prometheus 抓取 ``/metrics``；演示人员通过场景接口注入和恢复故障。
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse


SERVICE_NAME = "grid-data-sync-service"
REGION = "demo-grid-region"
STATION = "demo-control-center"
TOTAL_STATIONS = 12
CONTROL_PAGE = Path(__file__).with_name("control.html")

SCENARIOS: dict[str, dict[str, Any]] = {
    "normal": {
        "description": "电网遥测数据采集与同步正常",
        "level": "INFO",
        "message": "遥测数据采集、处理和主站同步均正常",
        "online_stations": TOTAL_STATIONS,
        "queue_depth": 20.0,
        "sync_failure_rate": 0.5,
        "data_freshness_seconds": 2.0,
        "processing_latency_seconds": 0.18,
    },
    "service_down": {
        "description": "数据同步服务不可用",
        "level": "ERROR",
        "message": "电网数据同步服务健康检查失败，指标端点不可用",
        "online_stations": 0,
        "queue_depth": 0.0,
        "sync_failure_rate": 100.0,
        "data_freshness_seconds": 120.0,
        "processing_latency_seconds": 0.0,
    },
    "communication_interruption": {
        "description": "部分站点通信中断",
        "level": "ERROR",
        "message": "检测到部分变电站采集通道中断，遥测数据无法按时上送",
        "online_stations": 7,
        "queue_depth": 260.0,
        "sync_failure_rate": 12.0,
        "data_freshness_seconds": 18.0,
        "processing_latency_seconds": 0.8,
    },
    "queue_backlog": {
        "description": "遥测消息严重积压",
        "level": "WARN",
        "message": "遥测消息消费速度低于采集速度，待处理队列持续增长",
        "online_stations": TOTAL_STATIONS,
        "queue_depth": 1500.0,
        "sync_failure_rate": 3.0,
        "data_freshness_seconds": 12.0,
        "processing_latency_seconds": 1.2,
    },
    "sync_failure": {
        "description": "数据同步失败率过高",
        "level": "ERROR",
        "message": "主站数据同步连续失败，请检查下游接口和数据校验结果",
        "online_stations": TOTAL_STATIONS,
        "queue_depth": 420.0,
        "sync_failure_rate": 45.0,
        "data_freshness_seconds": 20.0,
        "processing_latency_seconds": 1.0,
    },
    "data_delay": {
        "description": "遥测数据处理与更新延迟",
        "level": "WARN",
        "message": "最新遥测数据已超过允许的新鲜度窗口，处理耗时明显升高",
        "online_stations": TOTAL_STATIONS,
        "queue_depth": 680.0,
        "sync_failure_rate": 8.0,
        "data_freshness_seconds": 90.0,
        "processing_latency_seconds": 4.5,
    },
}


class GridSimulationState:
    """线程安全的模拟状态、指标和日志存储。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self.logs: deque[dict[str, Any]] = deque(maxlen=1000)
        self.started_at = time.time()
        self.total_messages = 100_000.0
        self.total_failed_messages = 120.0
        self.scenario = "normal"
        self.scenario_started_at = time.time()
        self.last_periodic_log_at = 0.0
        self._apply_profile("normal", record_log=True)

    def _append_log(self, level: str, message: str) -> None:
        now = datetime.now(timezone.utc)
        self.logs.append(
            {
                "timestamp": now.isoformat(),
                "timestamp_ms": int(now.timestamp() * 1000),
                "level": level,
                "service": SERVICE_NAME,
                "region": REGION,
                "station": STATION,
                "scenario": self.scenario,
                "message": message,
            }
        )

    def _apply_profile(self, scenario: str, record_log: bool) -> None:
        profile = SCENARIOS[scenario]
        self.scenario = scenario
        self.scenario_started_at = time.time()
        self.online_stations = float(profile["online_stations"])
        self.queue_depth = float(profile["queue_depth"])
        self.sync_failure_rate = float(profile["sync_failure_rate"])
        self.data_freshness_seconds = float(profile["data_freshness_seconds"])
        self.processing_latency_seconds = float(profile["processing_latency_seconds"])
        if record_log:
            self._append_log(str(profile["level"]), str(profile["message"]))
            self.last_periodic_log_at = time.time()

    def set_scenario(self, scenario: str) -> dict[str, Any]:
        if scenario not in SCENARIOS:
            raise ValueError(f"未知场景 {scenario!r}")
        with self._lock:
            self._apply_profile(scenario, record_log=True)
            return self.snapshot()

    def tick(self) -> None:
        """按固定节奏推进指标，使 Prometheus 能观察到连续趋势。"""
        with self._lock:
            if self.scenario == "normal":
                self.total_messages += 240
                self.total_failed_messages += 1
                self.queue_depth = max(10.0, self.queue_depth - 30.0)
                self.data_freshness_seconds = 2.0
                self.processing_latency_seconds = 0.18
                self.sync_failure_rate = 0.5
            elif self.scenario == "communication_interruption":
                self.total_messages += 80
                self.total_failed_messages += 12
                self.queue_depth += 35
                self.data_freshness_seconds = min(29.0, self.data_freshness_seconds + 1)
            elif self.scenario == "queue_backlog":
                self.total_messages += 120
                self.total_failed_messages += 3
                self.queue_depth += 80
            elif self.scenario == "sync_failure":
                self.total_messages += 180
                self.total_failed_messages += 81
                self.queue_depth += 25
            elif self.scenario == "data_delay":
                self.total_messages += 100
                self.total_failed_messages += 8
                self.data_freshness_seconds += 5
                self.processing_latency_seconds = min(8.0, self.processing_latency_seconds + 0.2)

            # 故障持续期间每 30 秒补充一条同场景日志，既符合持续异常的运行特征，也
            # 保证在告警触发后执行诊断时能查询到时间窗口内的证据。
            now = time.time()
            if self.scenario != "normal" and now - self.last_periodic_log_at >= 30:
                profile = SCENARIOS[self.scenario]
                self._append_log(str(profile["level"]), str(profile["message"]))
                self.last_periodic_log_at = now

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            profile = SCENARIOS[self.scenario]
            return {
                "service": SERVICE_NAME,
                "region": REGION,
                "station": STATION,
                "scenario": self.scenario,
                "scenario_description": profile["description"],
                "scenario_started_at": datetime.fromtimestamp(
                    self.scenario_started_at, tz=timezone.utc
                ).isoformat(),
                "healthy": self.scenario != "service_down",
                "metrics": {
                    "grid_service_health": 0 if self.scenario == "service_down" else 1,
                    "grid_station_online": self.online_stations,
                    "grid_station_total": float(TOTAL_STATIONS),
                    "grid_telemetry_queue_depth": self.queue_depth,
                    "grid_data_sync_failure_rate": self.sync_failure_rate,
                    "grid_data_freshness_seconds": self.data_freshness_seconds,
                    "grid_telemetry_processing_latency_seconds": self.processing_latency_seconds,
                    "grid_telemetry_messages_total": self.total_messages,
                    "grid_telemetry_failed_messages_total": self.total_failed_messages,
                },
            }

    def query_logs(
        self,
        start_time: int | None,
        end_time: int | None,
        level: str | None,
        query: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self.logs)
        if start_time is not None:
            items = [item for item in items if item["timestamp_ms"] >= start_time]
        if end_time is not None:
            items = [item for item in items if item["timestamp_ms"] <= end_time]
        if level:
            items = [item for item in items if item["level"].upper() == level.upper()]
        if query:
            needle = query.lower()
            items = [
                item
                for item in items
                if needle in item["message"].lower()
                or needle in item["scenario"].lower()
            ]
        return items[-limit:]


state = GridSimulationState()


async def _simulation_loop() -> None:
    while True:
        state.tick()
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(_simulation_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Grid Data Sync Simulator",
    version="1.0.0",
    description="电网运行数据采集与同步服务模拟器",
    lifespan=lifespan,
)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _metric_line(name: str, value: float, labels: str) -> str:
    return f"{name}{{{labels}}} {value}"


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "message": "电网数据采集与同步模拟服务正在运行",
        "control_page": "/control",
        "scenario_api": "/api/scenario/{scenario}",
        "available_scenarios": list(SCENARIOS),
    }


@app.get("/control", response_class=FileResponse)
async def control_page() -> FileResponse:
    """返回面向演示人员的模拟场景控制页面。"""
    return FileResponse(CONTROL_PAGE, media_type="text/html; charset=utf-8")


@app.get("/health")
async def health():
    snapshot = state.snapshot()
    status_code = 200 if snapshot["healthy"] else 503
    return JSONResponse(status_code=status_code, content=snapshot)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    snapshot = state.snapshot()
    if not snapshot["healthy"]:
        return PlainTextResponse(
            "# grid data sync service is unavailable\n",
            status_code=503,
            media_type="text/plain; version=0.0.4",
        )

    metrics_data = snapshot["metrics"]
    labels = ",".join(
        [
            f'region="{_escape_label(REGION)}"',
            f'station="{_escape_label(STATION)}"',
            f'service="{_escape_label(SERVICE_NAME)}"',
        ]
    )
    scenario_labels = f'{labels},scenario="{_escape_label(state.scenario)}"'
    lines = [
        "# HELP grid_service_health Whether the grid data sync service is healthy.",
        "# TYPE grid_service_health gauge",
        _metric_line("grid_service_health", metrics_data["grid_service_health"], labels),
        "# HELP grid_station_online Number of stations with active telemetry channels.",
        "# TYPE grid_station_online gauge",
        _metric_line("grid_station_online", metrics_data["grid_station_online"], labels),
        "# HELP grid_station_total Total number of stations managed by the service.",
        "# TYPE grid_station_total gauge",
        _metric_line("grid_station_total", metrics_data["grid_station_total"], labels),
        "# HELP grid_telemetry_queue_depth Number of telemetry messages waiting to be processed.",
        "# TYPE grid_telemetry_queue_depth gauge",
        _metric_line("grid_telemetry_queue_depth", metrics_data["grid_telemetry_queue_depth"], labels),
        "# HELP grid_data_sync_failure_rate Percentage of failed data synchronization operations.",
        "# TYPE grid_data_sync_failure_rate gauge",
        _metric_line("grid_data_sync_failure_rate", metrics_data["grid_data_sync_failure_rate"], labels),
        "# HELP grid_data_freshness_seconds Age of the latest synchronized telemetry data.",
        "# TYPE grid_data_freshness_seconds gauge",
        _metric_line("grid_data_freshness_seconds", metrics_data["grid_data_freshness_seconds"], labels),
        "# HELP grid_telemetry_processing_latency_seconds Telemetry processing latency.",
        "# TYPE grid_telemetry_processing_latency_seconds gauge",
        _metric_line(
            "grid_telemetry_processing_latency_seconds",
            metrics_data["grid_telemetry_processing_latency_seconds"],
            labels,
        ),
        "# HELP grid_telemetry_messages_total Total telemetry messages processed.",
        "# TYPE grid_telemetry_messages_total counter",
        _metric_line("grid_telemetry_messages_total", metrics_data["grid_telemetry_messages_total"], labels),
        "# HELP grid_telemetry_failed_messages_total Total telemetry messages that failed processing.",
        "# TYPE grid_telemetry_failed_messages_total counter",
        _metric_line(
            "grid_telemetry_failed_messages_total",
            metrics_data["grid_telemetry_failed_messages_total"],
            labels,
        ),
        "# HELP grid_simulator_scenario Active simulator scenario.",
        "# TYPE grid_simulator_scenario gauge",
        _metric_line("grid_simulator_scenario", 1.0, scenario_labels),
    ]
    return PlainTextResponse(
        "\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4",
    )


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    return state.snapshot()


@app.post("/api/scenario/{scenario}")
async def set_scenario(scenario: str) -> dict[str, Any]:
    try:
        snapshot = state.set_scenario(scenario)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "available_scenarios": list(SCENARIOS)},
        ) from exc
    return {"success": True, "message": f"已切换到 {scenario} 场景", "data": snapshot}


@app.post("/api/recover")
async def recover() -> dict[str, Any]:
    snapshot = state.set_scenario("normal")
    return {"success": True, "message": "模拟服务已恢复正常", "data": snapshot}


@app.get("/api/logs")
async def get_logs(
    start_time: int | None = None,
    end_time: int | None = None,
    level: str | None = None,
    query: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    logs = state.query_logs(start_time, end_time, level, query, limit)
    return {
        "success": True,
        "service": SERVICE_NAME,
        "scenario": state.scenario,
        "total": len(logs),
        "logs": logs,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9105)
