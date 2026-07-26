"""智能运维监控 MCP Server。

Monitor MCP 作为 Agent 与监控平台之间的适配层，对外提供稳定的业务工具，
内部通过 Prometheus HTTP API 查询活动告警和时序指标。
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Monitor_MCP_Server")

mcp = FastMCP("Monitor")

PROMETHEUS_BASE_URL = os.getenv(
    "PROMETHEUS_BASE_URL", "http://127.0.0.1:9090"
).rstrip("/")
PROMETHEUS_REQUEST_TIMEOUT = float(os.getenv("PROMETHEUS_REQUEST_TIMEOUT", "10"))
MONITOR_EXPORTER_TYPE = os.getenv("MONITOR_EXPORTER_TYPE", "windows").strip().lower()
PROMETHEUS_SERVICE_LABEL = os.getenv("PROMETHEUS_SERVICE_LABEL", "service").strip()

QUERY_API_PATH = "/api/v1/query"
QUERY_RANGE_API_PATH = "/api/v1/query_range"
ALERTS_API_PATH = "/api/v1/alerts"

CPU_ALERT_THRESHOLD = 80.0
MEMORY_ALERT_THRESHOLD = 70.0


class PrometheusError(RuntimeError):
    """Prometheus 请求或响应错误。"""


def log_tool_call(func):
    """记录工具调用参数、状态和结果摘要。"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__
        logger.info("=" * 80)
        logger.info("调用方法: %s", method_name)

        try:
            params_str = json.dumps(kwargs, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            params_str = str(kwargs)
        logger.info("参数信息:\n%s", params_str if kwargs else "无")

        try:
            result = func(*args, **kwargs)
            logger.info("返回状态: %s", "SUCCESS" if result.get("success", True) else "FAILED")
            if isinstance(result, dict):
                summary = {
                    key: value
                    if not isinstance(value, (list, dict))
                    else f"<{type(value).__name__} with {len(value)} items>"
                    for key, value in list(result.items())[:6]
                }
                logger.info("返回结果摘要: %s", json.dumps(summary, ensure_ascii=False))
            logger.info("=" * 80)
            return result
        except Exception as exc:
            logger.exception("工具执行异常: %s", exc)
            logger.info("=" * 80)
            raise

    return wrapper


def _validate_configuration() -> None:
    if MONITOR_EXPORTER_TYPE not in {"windows", "node", "auto"}:
        raise ValueError(
            "MONITOR_EXPORTER_TYPE 必须是 windows、node 或 auto，"
            f"当前值为 {MONITOR_EXPORTER_TYPE!r}"
        )
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", PROMETHEUS_SERVICE_LABEL):
        raise ValueError(
            "PROMETHEUS_SERVICE_LABEL 不是合法的 Prometheus 标签名: "
            f"{PROMETHEUS_SERVICE_LABEL!r}"
        )


def _local_timezone():
    return datetime.now().astimezone().tzinfo or timezone.utc


def parse_time_or_default(time_str: Optional[str], default_offset_hours: int = 0) -> datetime:
    """解析本地时间或 RFC3339 时间；未提供时使用当前时间偏移。"""
    if not time_str:
        return datetime.now().astimezone() + timedelta(hours=default_offset_hours)

    value = time_str.strip()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError as exc:
            raise ValueError(
                "时间格式必须是 YYYY-MM-DD HH:MM:SS 或 RFC3339"
            ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_local_timezone())
    return parsed


def _validate_interval(interval: str) -> str:
    value = interval.strip().lower()
    if len(value) < 2 or value[-1] not in {"s", "m", "h"}:
        raise ValueError("interval 必须使用秒、分钟或小时格式，例如 30s、1m、5m、1h")
    try:
        amount = int(value[:-1])
    except ValueError as exc:
        raise ValueError("interval 数值部分必须是正整数") from exc
    if amount <= 0:
        raise ValueError("interval 必须大于 0")
    return value


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _service_matcher(service_name: str) -> str:
    return f'{PROMETHEUS_SERVICE_LABEL}="{_escape_label_value(service_name)}"'


def _prometheus_get(path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    url = f"{PROMETHEUS_BASE_URL}{path}"
    logger.info("请求 Prometheus: %s params=%s", url, params or {})

    try:
        with httpx.Client(
            timeout=PROMETHEUS_REQUEST_TIMEOUT,
            trust_env=False,
        ) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            body = response.json()
    except httpx.TimeoutException as exc:
        raise PrometheusError(f"Prometheus 请求超时: {url}") from exc
    except httpx.HTTPStatusError as exc:
        raise PrometheusError(
            f"Prometheus 返回 HTTP {exc.response.status_code}: {exc.response.text[:300]}"
        ) from exc
    except httpx.HTTPError as exc:
        raise PrometheusError(f"无法连接 Prometheus: {exc}") from exc
    except ValueError as exc:
        raise PrometheusError("Prometheus 返回的内容不是合法 JSON") from exc

    if body.get("status") != "success":
        error_type = body.get("errorType", "unknown")
        error = body.get("error", "Prometheus returned non-success status")
        raise PrometheusError(f"Prometheus 查询失败 ({error_type}): {error}")

    return body


def _cpu_query(service_name: str, exporter_type: str) -> str:
    matcher = _service_matcher(service_name)
    if exporter_type == "node":
        return (
            "100 - (avg by (instance, job, service, system, station) "
            f"(rate(node_cpu_seconds_total{{{matcher},mode=\"idle\"}}[5m])) * 100)"
        )
    if exporter_type == "windows":
        return (
            "100 - (avg by (instance, job, service, system, station) "
            f"(rate(windows_cpu_time_total{{{matcher},mode=\"idle\"}}[5m])) * 100)"
        )
    return f"({_cpu_query(service_name, 'windows')}) or ({_cpu_query(service_name, 'node')})"


def _memory_query(service_name: str, exporter_type: str) -> str:
    matcher = _service_matcher(service_name)
    if exporter_type == "node":
        return (
            "(1 - ("
            f"node_memory_MemAvailable_bytes{{{matcher}}} / "
            f"node_memory_MemTotal_bytes{{{matcher}}}"
            ")) * 100"
        )
    if exporter_type == "windows":
        return (
            "(1 - ("
            f"windows_os_physical_memory_free_bytes{{{matcher}}} / "
            f"windows_cs_physical_memory_bytes{{{matcher}}}"
            ")) * 100"
        )
    return f"({_memory_query(service_name, 'windows')}) or ({_memory_query(service_name, 'node')})"


def _parse_matrix(data: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get("resultType") != "matrix":
        raise PrometheusError(
            f"期望 Prometheus 返回 matrix，实际为 {data.get('resultType')!r}"
        )

    series: list[dict[str, Any]] = []
    for item in data.get("result", []):
        if not isinstance(item, dict):
            continue
        points: list[dict[str, Any]] = []
        for raw_point in item.get("values", []):
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                continue
            try:
                timestamp = float(raw_point[0])
                value = float(raw_point[1])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            points.append(
                {
                    "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                    "value": round(value, 2),
                }
            )
        series.append({"labels": item.get("metric", {}), "data_points": points})
    return series


def _statistics(series: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        point["value"]
        for item in series
        for point in item.get("data_points", [])
        if isinstance(point.get("value"), (int, float))
    ]
    if not values:
        return {}

    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, int((len(sorted_values) - 1) * 0.95))
    latest_values = [
        item["data_points"][-1]["value"]
        for item in series
        if item.get("data_points")
    ]
    current = sum(latest_values) / len(latest_values) if latest_values else values[-1]

    return {
        "current": round(current, 2),
        "avg": round(sum(values) / len(values), 2),
        "max": round(max(values), 2),
        "min": round(min(values), 2),
        "p95": round(sorted_values[p95_index], 2),
        "sample_count": len(values),
    }


def _metric_error(
    service_name: str,
    metric_name: str,
    message: str,
    error_type: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "source": "prometheus",
        "service_name": service_name,
        "metric_name": metric_name,
        "error_type": error_type,
        "message": message,
    }


def _query_metric(
    service_name: str,
    metric_name: str,
    query_builder,
    threshold: float,
    start_time: Optional[str],
    end_time: Optional[str],
    interval: str,
) -> dict[str, Any]:
    try:
        start_dt = parse_time_or_default(start_time, default_offset_hours=-1)
        end_dt = parse_time_or_default(end_time)
        step = _validate_interval(interval)
        if start_dt >= end_dt:
            raise ValueError("start_time 必须早于 end_time")

        promql = query_builder(service_name, MONITOR_EXPORTER_TYPE)
        body = _prometheus_get(
            QUERY_RANGE_API_PATH,
            {
                "query": promql,
                "start": start_dt.timestamp(),
                "end": end_dt.timestamp(),
                "step": step,
            },
        )
        series = _parse_matrix(body.get("data", {}))
        stats = _statistics(series)

        if not stats:
            return {
                "success": True,
                "source": "prometheus",
                "service_name": service_name,
                "metric_name": metric_name,
                "exporter_type": MONITOR_EXPORTER_TYPE,
                "interval": step,
                "time_range": {
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                },
                "data_points": [],
                "series": series,
                "statistics": {},
                "alert_info": {
                    "triggered": False,
                    "threshold": threshold,
                    "message": "Prometheus 查询成功，但没有匹配的时序数据",
                },
                "query_metadata": {
                    "service_label": PROMETHEUS_SERVICE_LABEL,
                    "promql": promql,
                },
                "message": (
                    f"未找到服务 {service_name!r} 的 {metric_name} 数据，请检查服务标签、"
                    "Exporter 抓取状态和时间范围"
                ),
            }

        triggered = stats["max"] > threshold
        return {
            "success": True,
            "source": "prometheus",
            "service_name": service_name,
            "metric_name": metric_name,
            "exporter_type": MONITOR_EXPORTER_TYPE,
            "interval": step,
            "time_range": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
            },
            # 保留旧工具的 data_points 字段，默认呈现第一条目标序列。
            "data_points": series[0]["data_points"] if series else [],
            "series": series,
            "statistics": stats,
            "alert_info": {
                "triggered": triggered,
                "threshold": threshold,
                "message": (
                    f"{metric_name} 最大值 {stats['max']}% 超过 {threshold}% 阈值"
                    if triggered
                    else f"{metric_name} 未超过 {threshold}% 阈值"
                ),
            },
            "query_metadata": {
                "service_label": PROMETHEUS_SERVICE_LABEL,
                "promql": promql,
            },
            "message": f"已从 Prometheus 获取 {len(series)} 条目标序列",
        }
    except ValueError as exc:
        return _metric_error(service_name, metric_name, str(exc), "invalid_argument")
    except PrometheusError as exc:
        return _metric_error(service_name, metric_name, str(exc), "prometheus_error")


def _parse_active_at(active_at: str) -> Optional[datetime]:
    if not active_at:
        return None
    try:
        parsed = datetime.fromisoformat(active_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _duration(active_at: str) -> str:
    parsed = _parse_active_at(active_at)
    if parsed is None:
        return "unknown"
    seconds = max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes}m{seconds}s"
    if minutes:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


@mcp.tool()
@log_tool_call
def list_active_alerts(
    severity: Optional[str] = None,
    service_name: Optional[str] = None,
    instance: Optional[str] = None,
    alert_name: Optional[str] = None,
    state: Optional[str] = None,
) -> dict[str, Any]:
    """查询 Prometheus 当前活动告警。

    当任务需要判断系统当前是否存在 firing/pending 告警，或需要确定后续指标查询目标时，
    应优先调用本工具。过滤参数均为可选，不传参数时返回全部活动告警。
    """
    try:
        body = _prometheus_get(ALERTS_API_PATH)
        raw_alerts = (body.get("data") or {}).get("alerts") or []
        alerts: list[dict[str, Any]] = []
        state_counts: dict[str, int] = {}

        for raw_alert in raw_alerts:
            if not isinstance(raw_alert, dict):
                continue
            labels = raw_alert.get("labels") or {}
            annotations = raw_alert.get("annotations") or {}
            alert_state = str(raw_alert.get("state", ""))

            if severity and labels.get("severity") != severity:
                continue
            if service_name and labels.get(PROMETHEUS_SERVICE_LABEL) != service_name:
                continue
            if instance and labels.get("instance") != instance:
                continue
            if alert_name and labels.get("alertname") != alert_name:
                continue
            if state and alert_state != state:
                continue

            active_at = str(raw_alert.get("activeAt", ""))
            state_counts[alert_state] = state_counts.get(alert_state, 0) + 1
            alerts.append(
                {
                    "alert_name": labels.get("alertname", ""),
                    "state": alert_state,
                    "severity": labels.get("severity", ""),
                    "service_name": labels.get(PROMETHEUS_SERVICE_LABEL, ""),
                    "instance": labels.get("instance", ""),
                    "job": labels.get("job", ""),
                    "active_at": active_at,
                    "duration": _duration(active_at),
                    "summary": annotations.get("summary", ""),
                    "description": annotations.get("description", ""),
                    "labels": labels,
                }
            )

        alerts.sort(
            key=lambda item: _parse_active_at(item["active_at"])
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return {
            "success": True,
            "source": "prometheus",
            "alerts": alerts,
            "state_counts": state_counts,
            "total": len(alerts),
            "filters": {
                "severity": severity,
                "service_name": service_name,
                "instance": instance,
                "alert_name": alert_name,
                "state": state,
            },
            "message": f"已从 Prometheus 获取 {len(alerts)} 条活动告警",
        }
    except PrometheusError as exc:
        return {
            "success": False,
            "source": "prometheus",
            "error_type": "prometheus_error",
            "message": str(exc),
            "alerts": [],
            "total": 0,
        }


@mcp.tool()
@log_tool_call
def query_cpu_metrics(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m",
) -> dict[str, Any]:
    """查询指定服务最近一段时间的 CPU 使用率。

    当告警、日志或用户描述涉及高负载、响应缓慢、任务积压时调用。service_name 应对应
    Prometheus 中配置的 service 标签；默认查询最近一小时，可指定本地时间或 RFC3339 时间。
    """
    return _query_metric(
        service_name=service_name,
        metric_name="cpu_usage_percent",
        query_builder=_cpu_query,
        threshold=CPU_ALERT_THRESHOLD,
        start_time=start_time,
        end_time=end_time,
        interval=interval,
    )


@mcp.tool()
@log_tool_call
def query_memory_metrics(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m",
) -> dict[str, Any]:
    """查询指定服务最近一段时间的内存使用率。

    当告警、日志或用户描述涉及内存压力、OOM、进程被终止或持续资源增长时调用。
    service_name 应对应 Prometheus 中配置的 service 标签；默认查询最近一小时。
    """
    return _query_metric(
        service_name=service_name,
        metric_name="memory_usage_percent",
        query_builder=_memory_query,
        threshold=MEMORY_ALERT_THRESHOLD,
        start_time=start_time,
        end_time=end_time,
        interval=interval,
    )


_validate_configuration()
logger.info(
    "Monitor MCP configured: prometheus=%s exporter=%s service_label=%s",
    PROMETHEUS_BASE_URL,
    MONITOR_EXPORTER_TYPE,
    PROMETHEUS_SERVICE_LABEL,
)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8004, path="/mcp")
