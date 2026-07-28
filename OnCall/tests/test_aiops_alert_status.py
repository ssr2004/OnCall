"""AIOps 顶部告警状态摘要测试。"""

import json

import pytest

from app.services.aiops_service import AIOpsService


class FakeAlertTool:
    def __init__(self, payload=None, error: Exception | None = None):
        self.payload = payload
        self.error = error

    async def ainvoke(self, _arguments):
        if self.error is not None:
            raise self.error
        return json.dumps(self.payload, ensure_ascii=False)


def make_service(tool: FakeAlertTool) -> AIOpsService:
    service = object.__new__(AIOpsService)
    service._alert_status_tool = tool
    return service


@pytest.mark.asyncio
async def test_alert_status_prioritizes_firing_and_returns_safe_summaries() -> None:
    service = make_service(
        FakeAlertTool(
            {
                "success": True,
                "total": 2,
                "alerts": [
                    {
                        "alert_name": "GridCommunicationInterrupted",
                        "severity": "critical",
                        "service_name": "grid-data-sync-service",
                        "state": "firing",
                        "summary": "部分电网站点通信中断",
                    },
                    {
                        "alert_name": "GridDataFreshnessDelayHigh",
                        "severity": "warning",
                        "service_name": "grid-data-sync-service",
                        "state": "pending",
                    },
                ],
            }
        )
    )

    result = await service.get_alert_status()

    assert result["success"] is True
    assert result["status"] == "firing"
    assert result["total"] == 2
    assert result["firing"] == 1
    assert result["pending"] == 1
    assert result["alerts"][0]["alert_name"] == "GridCommunicationInterrupted"


@pytest.mark.asyncio
async def test_alert_status_distinguishes_pending_and_healthy() -> None:
    pending_service = make_service(
        FakeAlertTool(
            {
                "success": True,
                "alerts": [{"alert_name": "GridTelemetryQueueBacklog", "state": "pending"}],
            }
        )
    )
    healthy_service = make_service(FakeAlertTool({"success": True, "alerts": []}))

    pending = await pending_service.get_alert_status()
    healthy = await healthy_service.get_alert_status()

    assert pending["status"] == "pending"
    assert pending["pending"] == 1
    assert healthy["status"] == "healthy"
    assert healthy["total"] == 0


@pytest.mark.asyncio
async def test_alert_status_reports_monitor_connection_failure() -> None:
    service = make_service(FakeAlertTool(error=RuntimeError("monitor unavailable")))

    result = await service.get_alert_status()

    assert result["success"] is False
    assert result["status"] == "unavailable"
    assert result["total"] == 0
    assert "monitor unavailable" in result["message"]
    assert service._alert_status_tool is None
