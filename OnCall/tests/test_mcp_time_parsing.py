"""MCP 时间参数容错测试。"""

from datetime import datetime
import time

from mcp_servers.cls_server import normalize_log_time
from mcp_servers.monitor_server import parse_time_or_default


def test_cls_current_time_placeholder_becomes_millisecond_timestamp() -> None:
    before = int(time.time() * 1000)
    result = normalize_log_time("timestamp_placeholder")
    after = int(time.time() * 1000)

    assert before <= result <= after


def test_monitor_current_time_placeholder_respects_default_offset() -> None:
    end_time = parse_time_or_default("timestamp_placeholder")
    start_time = parse_time_or_default(
        "timestamp_placeholder", default_offset_hours=-1
    )

    assert end_time.tzinfo is not None
    assert start_time.tzinfo is not None
    assert 3590 <= (end_time - start_time).total_seconds() <= 3610


def test_rfc3339_remains_supported_by_both_mcp_servers() -> None:
    timestamp = "2026-07-26T11:38:48.434943Z"

    assert normalize_log_time(timestamp) == 1785065928434
    assert parse_time_or_default(timestamp) == datetime.fromisoformat(
        "2026-07-26T11:38:48.434943+00:00"
    )
