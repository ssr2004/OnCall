"""只读验证 Prometheus、Monitor MCP 与 CLS MCP 的电网演示数据是否一致。"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient


SERVERS = {
    "cls": {
        "transport": "streamable-http",
        "url": "http://127.0.0.1:8003/mcp",
    },
    "monitor": {
        "transport": "streamable-http",
        "url": "http://127.0.0.1:8004/mcp",
    },
}


def parse_tool_result(result: Any) -> dict[str, Any]:
    """兼容 MCP adapter 返回的 TextContent 列表或 JSON 字符串。"""
    if isinstance(result, list) and result:
        block = result[0]
        if isinstance(block, dict):
            result = block.get("text", "")
        else:
            result = getattr(block, "text", str(block))
    if isinstance(result, str):
        return json.loads(result)
    if isinstance(result, dict):
        return result
    raise TypeError(f"无法解析 MCP 工具结果: {type(result)!r}")


async def main() -> None:
    client = MultiServerMCPClient(SERVERS)
    tools = {tool.name: tool for tool in await client.get_tools()}
    required = {
        "list_active_alerts",
        "query_grid_service_status",
        "query_grid_data_sync_metrics",
        "query_grid_telemetry_metrics",
        "search_topic_by_service_name",
        "search_log",
    }
    missing = required - set(tools)
    if missing:
        raise RuntimeError(f"MCP 缺少工具: {sorted(missing)}")

    service_name = "grid-data-sync-service"
    alerts = parse_tool_result(
        await tools["list_active_alerts"].ainvoke({"service_name": service_name})
    )
    sync_metrics = parse_tool_result(
        await tools["query_grid_data_sync_metrics"].ainvoke(
            {"service_name": service_name, "interval": "5s"}
        )
    )
    topic = parse_tool_result(
        await tools["search_topic_by_service_name"].ainvoke(
            {"service_name": service_name}
        )
    )
    now_ms = int(time.time() * 1000)
    logs = parse_tool_result(
        await tools["search_log"].ainvoke(
            {
                "topic_id": "grid-topic-001",
                # 故意传入相同起止时间，验证 CLS 适配器会自动扩展为有效告警窗口。
                "start_time": now_ms,
                "end_time": now_ms,
                "query": "level:WARN OR level:ERROR",
                "limit": 100,
            }
        )
    )

    assert alerts["success"] is True
    assert sync_metrics["success"] is True
    assert topic["total"] == 1
    assert logs["success"] is True

    summary = {
        "tool_count": len(tools),
        "tool_names": sorted(tools),
        "alert_names": [item["alert_name"] for item in alerts["alerts"]],
        "sync_anomalies": sync_metrics["anomalies"],
        "topic_id": topic["topics"][0]["topic_id"],
        "log_count": logs["total"],
        "log_scenarios": sorted({item["scenario"] for item in logs["logs"]}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
