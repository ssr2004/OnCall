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
PROMETHEUS_SERVICE_LABEL = os.getenv("PROMETHEUS_SERVICE_LABEL", "service").strip()

QUERY_RANGE_API_PATH = "/api/v1/query_range"
ALERTS_API_PATH = "/api/v1/alerts"

QUEUE_BACKLOG_THRESHOLD = 1000.0
SYNC_FAILURE_RATE_THRESHOLD = 20.0
DATA_FRESHNESS_THRESHOLD = 30.0
PROCESSING_LATENCY_THRESHOLD = 2.0


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
    if value.lower() in {
        "now",
        "current_time",
        "current_timestamp",
        "timestamp_placeholder",
        "当前时间",
    }:
        # LLM 可能同时发起“取当前时间”和“查指标”两个工具调用，后一个调用
        # 无法消费前一个调用的结果。把常见占位值按该参数的默认偏移解析，
        # start_time 会得到一小时前，end_time 会得到当前时间。
        return datetime.now().astimezone() + timedelta(hours=default_offset_hours)
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


def _metric_query(service_name: str, metric_name: str) -> str:
    return f"{metric_name}{{{_service_matcher(service_name)}}}"


def _message_rate_query(service_name: str) -> str:
    return (
        "rate(grid_telemetry_messages_total"
        f"{{{_service_matcher(service_name)}}}[1m])"
    )


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


def _metric_group_error(
    service_name: str,
    group_name: str,
    message: str,
    error_type: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "source": "prometheus",
        "service_name": service_name,
        "metric_group": group_name,
        "error_type": error_type,
        "message": message,
    }


def _query_metric_group(
    service_name: str,
    group_name: str,
    metric_queries: dict[str, dict[str, Any]],
    start_time: Optional[str],
    end_time: Optional[str],
    interval: str,
) -> dict[str, Any]:
    """查询一组电网业务指标并给出统一的统计与异常判断。"""
    try:
        start_dt = parse_time_or_default(start_time, default_offset_hours=-1)
        end_dt = parse_time_or_default(end_time)
        step = _validate_interval(interval)
        if start_dt >= end_dt:
            raise ValueError("start_time 必须早于 end_time")

        metrics: dict[str, Any] = {}
        anomalies: list[dict[str, Any]] = []
        for metric_key, spec in metric_queries.items():
            promql = str(spec["promql"])
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
            metric_result = {
                "metric_name": spec["metric_name"],
                "description": spec["description"],
                "unit": spec["unit"],
                "series": series,
                "statistics": stats,
                "query_metadata": {
                    "service_label": PROMETHEUS_SERVICE_LABEL,
                    "promql": promql,
                },
            }
            metrics[metric_key] = metric_result

            threshold = spec.get("threshold")
            comparator = spec.get("comparator", "gt")
            if stats and threshold is not None:
                observed = stats["min"] if comparator == "lt" else stats["max"]
                triggered = observed < threshold if comparator == "lt" else observed > threshold
                if triggered:
                    anomalies.append(
                        {
                            "metric": metric_key,
                            "observed": observed,
                            "threshold": threshold,
                            "comparator": comparator,
                            "message": spec["alert_message"],
                        }
                    )

        # 在线站点数量需要与同一时刻的站点总数比较，而不是固定阈值。
        if "station_online" in metrics and "station_total" in metrics:
            online_stats = metrics["station_online"]["statistics"]
            total_stats = metrics["station_total"]["statistics"]
            if online_stats and total_stats and online_stats["current"] < total_stats["current"]:
                anomalies.append(
                    {
                        "metric": "station_online",
                        "observed": online_stats["current"],
                        "threshold": total_stats["current"],
                        "comparator": "lt",
                        "message": "在线站点数量低于应接入站点总数",
                    }
                )

        available_count = sum(1 for item in metrics.values() if item["statistics"])
        return {
            "success": True,
            "source": "prometheus",
            "service_name": service_name,
            "metric_group": group_name,
            "interval": step,
            "time_range": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
            },
            "metrics": metrics,
            "anomalies": anomalies,
            "message": (
                f"已查询 {len(metrics)} 项电网业务指标，其中 {available_count} 项包含数据，"
                f"发现 {len(anomalies)} 项异常"
            ),
        }
    except ValueError as exc:
        return _metric_group_error(service_name, group_name, str(exc), "invalid_argument")
    except PrometheusError as exc:
        return _metric_group_error(service_name, group_name, str(exc), "prometheus_error")


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
def query_grid_service_status(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m",
) -> dict[str, Any]:
    """查询电网数据服务健康状态和站点在线情况。

    当告警涉及服务不可用、站点离线或采集通道中断时调用。默认查询最近一小时。
    """
    return _query_metric_group(
        service_name=service_name,
        group_name="grid_service_status",
        metric_queries={
            "service_health": {
                "metric_name": "grid_service_health",
                "description": "电网数据同步服务健康状态",
                "unit": "boolean",
                "promql": _metric_query(service_name, "grid_service_health"),
                "threshold": 1.0,
                "comparator": "lt",
                "alert_message": "服务健康状态异常",
            },
            "station_online": {
                "metric_name": "grid_station_online",
                "description": "当前在线站点数量",
                "unit": "stations",
                "promql": _metric_query(service_name, "grid_station_online"),
            },
            "station_total": {
                "metric_name": "grid_station_total",
                "description": "应接入站点总数",
                "unit": "stations",
                "promql": _metric_query(service_name, "grid_station_total"),
            },
        },
        start_time=start_time,
        end_time=end_time,
        interval=interval,
    )


@mcp.tool()
@log_tool_call
def query_grid_data_sync_metrics(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m",
) -> dict[str, Any]:
    """查询电网遥测队列积压和数据同步失败率。

    当告警涉及消息积压、数据同步失败或下游接口异常时调用。默认查询最近一小时。
    """
    return _query_metric_group(
        service_name=service_name,
        group_name="grid_data_sync",
        metric_queries={
            "queue_depth": {
                "metric_name": "grid_telemetry_queue_depth",
                "description": "待处理遥测消息数量",
                "unit": "messages",
                "promql": _metric_query(service_name, "grid_telemetry_queue_depth"),
                "threshold": QUEUE_BACKLOG_THRESHOLD,
                "alert_message": "遥测消息队列超过 1000 条",
            },
            "sync_failure_rate": {
                "metric_name": "grid_data_sync_failure_rate",
                "description": "数据同步失败率",
                "unit": "percent",
                "promql": _metric_query(service_name, "grid_data_sync_failure_rate"),
                "threshold": SYNC_FAILURE_RATE_THRESHOLD,
                "alert_message": "数据同步失败率超过 20%",
            },
        },
        start_time=start_time,
        end_time=end_time,
        interval=interval,
    )


@mcp.tool()
@log_tool_call
def query_grid_telemetry_metrics(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m",
) -> dict[str, Any]:
    """查询电网遥测数据新鲜度、处理耗时和消息处理速率。

    当告警涉及数据延迟、数据长时间未更新或处理性能下降时调用。默认查询最近一小时。
    """
    return _query_metric_group(
        service_name=service_name,
        group_name="grid_telemetry",
        metric_queries={
            "data_freshness": {
                "metric_name": "grid_data_freshness_seconds",
                "description": "最新同步数据距当前时间的秒数",
                "unit": "seconds",
                "promql": _metric_query(service_name, "grid_data_freshness_seconds"),
                "threshold": DATA_FRESHNESS_THRESHOLD,
                "alert_message": "遥测数据超过 30 秒未更新",
            },
            "processing_latency": {
                "metric_name": "grid_telemetry_processing_latency_seconds",
                "description": "单批遥测数据处理耗时",
                "unit": "seconds",
                "promql": _metric_query(
                    service_name, "grid_telemetry_processing_latency_seconds"
                ),
                "threshold": PROCESSING_LATENCY_THRESHOLD,
                "alert_message": "遥测处理耗时超过 2 秒",
            },
            "message_rate": {
                "metric_name": "grid_telemetry_message_rate",
                "description": "每秒处理的遥测消息数量",
                "unit": "messages_per_second",
                "promql": _message_rate_query(service_name),
            },
        },
        start_time=start_time,
        end_time=end_time,
        interval=interval,
    )


_validate_configuration()
logger.info(
    "Monitor MCP configured: prometheus=%s service_label=%s",
    PROMETHEUS_BASE_URL,
    PROMETHEUS_SERVICE_LABEL,
)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8004, path="/mcp")
