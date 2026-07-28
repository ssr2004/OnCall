"""AIOps Executor 权威参数校正测试。"""

from datetime import datetime, timezone
import json

from app.agent.aiops.executor import _normalize_grid_tool_calls


def test_grid_tool_calls_use_prefetched_alert_context() -> None:
    prefetched = {
        "alerts": [
            {
                "alert_name": "GridTelemetryQueueBacklog",
                "service_name": "grid-data-sync-service",
                "active_at": "2026-07-26T11:38:48.434943837Z",
            }
        ]
    }
    input_text = (
        "PREFETCHED_ACTIVE_ALERTS_JSON:\n"
        + json.dumps(prefetched)
        + "\n\n上述预取告警是权威事实。"
    )
    tool_calls = [
        {
            "name": "query_grid_data_sync_metrics",
            "args": {
                "service_name": "grid-data-slide-service",
                "end_time": "timestamp_placeholder",
            },
        },
        {
            "name": "search_topic_by_service_name",
            "args": {"service_name": "grid-data-slide-service"},
        },
        {
            "name": "search_log",
            "args": {
                "topic_id": "topic_placeholder",
                "start_time": "alert_time",
                "end_time": "current_time",
            },
        },
    ]

    _normalize_grid_tool_calls(
        tool_calls,
        input_text,
        now=datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc),
    )

    metric_args = tool_calls[0]["args"]
    assert metric_args == {
        "service_name": "grid-data-sync-service",
        "start_time": "2026-07-26T11:38:48.434943837Z",
        "end_time": "2026-07-26T12:00:00Z",
        "interval": "5s",
    }
    assert tool_calls[1]["args"]["service_name"] == "grid-data-sync-service"
    assert tool_calls[2]["args"]["topic_id"] == "grid-topic-001"
    assert tool_calls[2]["args"]["start_time"] == prefetched["alerts"][0]["active_at"]
    assert tool_calls[2]["args"]["end_time"] == "2026-07-26T12:00:00Z"
