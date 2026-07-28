"""腾讯云 CLS (Cloud Log Service) MCP Server

本地实现的 CLS 日志服务 MCP Server，提供日志查询、检索和分析功能。
"""

import logging
import functools
import json
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CLS_MCP_Server")

mcp = FastMCP("CLS")

GRID_SIMULATOR_BASE_URL = os.getenv(
    "GRID_SIMULATOR_BASE_URL", "http://127.0.0.1:9105"
).rstrip("/")
GRID_SIMULATOR_REQUEST_TIMEOUT = float(
    os.getenv("GRID_SIMULATOR_REQUEST_TIMEOUT", "10")
)


def query_grid_simulator_logs(params: Dict[str, Any]) -> Dict[str, Any]:
    """从电网模拟服务查询与当前故障场景一致的业务日志。"""
    url = f"{GRID_SIMULATOR_BASE_URL}/api/logs"
    try:
        with httpx.Client(
            timeout=GRID_SIMULATOR_REQUEST_TIMEOUT,
            trust_env=False,
        ) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"无法查询电网模拟服务日志: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("电网模拟服务返回的日志不是合法 JSON") from exc
    return body


def log_tool_call(func):
    """装饰器：记录工具调用的日志，包括方法名、参数和返回状态"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__

        # 记录调用信息
        logger.info(f"=" * 80)
        logger.info(f"调用方法: {method_name}")

        # 记录参数（排除self等）
        if kwargs:
            # 使用 json.dumps 格式化参数，处理可能的序列化错误
            try:
                params_str = json.dumps(kwargs, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                params_str = str(kwargs)
            logger.info(f"参数信息:\n{params_str}")
        else:
            logger.info("参数信息: 无")

        # 执行方法
        try:
            result = func(*args, **kwargs)

            # 记录返回状态
            logger.info(f"返回状态: SUCCESS")

            # 记录返回结果摘要（避免日志过长）
            if isinstance(result, dict):
                summary = {k: v if not isinstance(v, (list, dict)) else f"<{type(v).__name__} with {len(v)} items>"
                          for k, v in list(result.items())[:5]}
                logger.info(f"返回结果摘要: {json.dumps(summary, ensure_ascii=False)}")
            else:
                logger.info(f"返回结果: {result}")

            logger.info(f"=" * 80)
            return result

        except Exception as e:
            # 记录错误状态
            logger.error(f"返回状态: ERROR")
            logger.error(f"错误信息: {str(e)}")
            logger.error(f"=" * 80)
            raise

    return wrapper


def parse_time_or_default(time_str: Optional[str], default_offset_hours: int = 0) -> datetime:
    """解析时间字符串或返回默认时间。

    Args:
        time_str: 时间字符串（格式：YYYY-MM-DD HH:MM:SS）
        default_offset_hours: 默认时间偏移（小时）

    Returns:
        datetime: 解析后的时间对象
    """
    if time_str:
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.now() + timedelta(hours=default_offset_hours)


def generate_time_series(base_time: datetime, minutes_offset: int) -> str:
    """生成基于基准时间的时间字符串。

    Args:
        base_time: 基准时间
        minutes_offset: 分钟偏移量

    Returns:
        str: 格式化的时间字符串
    """
    result_time = base_time + timedelta(minutes=minutes_offset)
    return result_time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_log_time(value: int | str) -> int:
    """将毫秒时间戳、数字字符串或 RFC3339/本地时间转换为毫秒时间戳。"""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lower() in {
        "now",
        "current_time",
        "current_timestamp",
        "timestamp_placeholder",
        "当前时间",
    }:
        # ToolNode 会并行执行同一轮的工具调用，模型无法先取得
        # get_current_timestamp 的结果再填入 search_log。将常见的“当前时间”
        # 占位值在工具边界收敛为真实时间，避免演示诊断因占位符失败。
        return int(time.time() * 1000)
    if text.isdigit():
        return int(text)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError as exc:
            raise ValueError(
                "日志时间必须是毫秒时间戳、YYYY-MM-DD HH:MM:SS 或 RFC3339"
            ) from exc
    return int(parsed.timestamp() * 1000)


@mcp.tool()
@log_tool_call
def get_current_timestamp() -> int:
    """获取当前时间戳（以毫秒为单位）。
    
    此工具用于获取标准的毫秒时间戳，可用于：
    1. 作为 search_log 的 end_time 参数（查询到现在）
    2. 计算历史时间点作为 start_time 参数
    
    Returns:
        int: 当前时间戳（毫秒），例如: 1708012345000
    
    使用示例:
        # 获取当前时间
        current = get_current_timestamp()
        
        # 计算15分钟前的时间
        fifteen_min_ago = current - (15 * 60 * 1000)
        
        # 计算1小时前的时间
        one_hour_ago = current - (60 * 60 * 1000)
        
        # 用于搜索最近15分钟的日志
        search_log(
            topic_id="grid-topic-001",
            start_time=fifteen_min_ago,
            end_time=current
        )
    """
    return int(datetime.now().timestamp() * 1000)


@mcp.tool()
@log_tool_call
def get_region_code_by_name(region_name: str) -> Dict[str, Any]:
    """根据地区名称搜索对应的地区参数。

    Args:
        region_name: 地区名称（如：北京、上海、广州等）

    Returns:
        Dict: 包含地区代码和相关信息的字典
            - region_code: 地区代码
            - region_name: 地区名称
            - available: 是否可用
    """
    # 模拟地区映射表（实际应该从配置或数据库读取）
    region_mapping = {
        "演示地区": {"region_code": "demo-grid-region", "region_name": "演示地区", "available": True},
        "某地区": {"region_code": "demo-grid-region", "region_name": "某地区", "available": True},
        "北京": {"region_code": "ap-beijing", "region_name": "北京", "available": True},
        "上海": {"region_code": "ap-shanghai", "region_name": "上海", "available": True},
        "广州": {"region_code": "ap-guangzhou", "region_name": "广州", "available": True},
    }

    result = region_mapping.get(region_name)
    if result:
        return result
    else:
        return {
            "region_code": None,
            "region_name": region_name,
            "available": False,
            "error": f"未找到地区: {region_name}"
        }


@mcp.tool()
@log_tool_call
def get_topic_info_by_name(topic_name: str, region_code: Optional[str] = None) -> Dict[str, Any]:
    """根据主题名称搜索相关的主题信息。

    Args:
        topic_name: 主题名称
        region_code: 地区代码（可选）

    Returns:
        Dict: 包含主题信息的字典
            - topic_id: 主题ID
            - topic_name: 主题名称
            - region_code: 所属地区
            - create_time: 创建时间
            - log_count: 日志数量
    """
    mock_topics = [
        {
            "topic_id": "grid-topic-001",
            "topic_name": "电网数据采集与同步服务日志",
            "service_name": "grid-data-sync-service",
            "region_code": "demo-grid-region",
            "create_time": "2026-07-26 00:00:00",
            "log_count": 0,
            "description": "电网模拟服务产生的实时业务场景日志"
        }
    ]

    # 根据名称和地区筛选
    for topic in mock_topics:
        if topic["topic_name"] == topic_name:
            if region_code is None or topic["region_code"] == region_code:
                return topic

    return {
        "topic_id": None,
        "topic_name": topic_name,
        "region_code": region_code,
        "error": f"未找到主题: {topic_name}"
    }


@mcp.tool()
@log_tool_call
def search_topic_by_service_name(
    service_name: str,
    region_code: Optional[str] = None,
    fuzzy: bool = True
) -> Dict[str, Any]:
    """根据服务名称搜索相关的日志主题信息，支持模糊搜索。
    
    此工具用于根据服务名称查找对应的日志主题（topic），便于后续进行日志查询。
    
    Args:
        service_name: 服务名称（必填）
            示例: "grid-data-sync-service", "grid-data-sync"
            说明: 当 fuzzy=True 时，支持部分匹配
        
        region_code: 地区代码（可选）
            示例: "demo-grid-region"
            说明: 如果指定，只返回该地区的主题
        
        fuzzy: 是否启用模糊搜索（可选，默认 True）
            True: 部分匹配，例如 "grid-data-sync" 可以匹配 "grid-data-sync-service"
            False: 精确匹配，必须完全一致
    
    Returns:
        Dict: 搜索结果
            - total: 匹配到的主题数量
            - topics: 主题列表，每个主题包含:
                * topic_id: 主题ID（用于后续日志查询）
                * topic_name: 主题名称
                * service_name: 服务名称
                * region_code: 所属地区
                * create_time: 创建时间
                * log_count: 日志数量
                * description: 主题描述
            - query: 查询条件
    
    使用示例:
        # 示例1: 模糊搜索（推荐）
        search_topic_by_service_name(service_name="data-sync")
        # 可以匹配: "grid-data-sync-service"
        
        # 示例2: 精确搜索
        search_topic_by_service_name(
            service_name="grid-data-sync-service",
            fuzzy=False
        )
        
        # 示例3: 指定地区搜索
        search_topic_by_service_name(
            service_name="sync",
            region_code="demo-grid-region"
        )
        
        # 示例4: 查找后进行日志搜索的完整流程
        # 步骤1: 根据服务名查找 topic
        result = search_topic_by_service_name(service_name="grid-data-sync-service")
        
        # 步骤2: 获取 topic_id
        topic_id = result["topics"][0]["topic_id"]  # "grid-topic-001"
        
        # 步骤3: 使用 topic_id 查询日志
        current_ts = get_current_timestamp()
        start_ts = current_ts - (15 * 60 * 1000)
        search_log(
            topic_id=topic_id,
            start_time=start_ts,
            end_time=current_ts
        )
    """
    # Mock 主题数据（实际应该从配置或数据库读取）
    mock_topics = [
        {
            "topic_id": "grid-topic-001",
            "topic_name": "电网数据采集与同步服务日志",
            "service_name": "grid-data-sync-service",
            "region_code": "demo-grid-region",
            "create_time": "2026-07-26 00:00:00",
            "log_count": 0,
            "description": "电网模拟服务产生的实时业务场景日志"
        }
    ]
    
    matched_topics = []
    
    # 搜索逻辑
    for topic in mock_topics:
        # 地区筛选
        if region_code and topic["region_code"] != region_code:
            continue
        
        # 服务名称匹配
        topic_service_name = topic.get("service_name", "")
        
        if fuzzy:
            # 模糊匹配：服务名包含查询字符串，或查询字符串包含服务名
            if (service_name.lower() in topic_service_name.lower() or 
                topic_service_name.lower() in service_name.lower()):
                matched_topics.append(topic)
        else:
            # 精确匹配
            if topic_service_name == service_name:
                matched_topics.append(topic)
    
    return {
        "total": len(matched_topics),
        "topics": matched_topics,
        "query": {
            "service_name": service_name,
            "region_code": region_code,
            "fuzzy": fuzzy
        },
        "message": f"找到 {len(matched_topics)} 个匹配的日志主题" if matched_topics else f"未找到服务 '{service_name}' 的日志主题"
    }


@mcp.tool()
@log_tool_call
def search_log(
    topic_id: str,
    start_time: int | str,
    end_time: int | str,
    query: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """基于提供的查询参数搜索日志。

    Args:
        topic_id: 主题ID（必填）
            当前电网模拟服务使用 "grid-topic-001"
        
        start_time: 开始时间（必填），支持毫秒时间戳或 RFC3339 字符串
            获取方式: 
            1. 使用 get_current_timestamp() 工具获取当前时间戳
            2. 计算历史时间: current_timestamp - (分钟数 * 60 * 1000)
            示例: 
            - 当前时间: 1708012345000
            - 15分钟前: 1708012345000 - (15 * 60 * 1000) = 1708011445000
            - 1小时前: 1708012345000 - (60 * 60 * 1000) = 1708008745000
        
        end_time: 结束时间（必填），支持毫秒时间戳或 RFC3339 字符串
            通常使用 get_current_timestamp() 工具获取当前时间作为结束时间
            示例: 1708012345000
        
        query: 查询语句（可选，CLS 查询语法）
            示例: "level:ERROR" 或 "message:异常"
        
        limit: 返回结果数量限制（默认100，可选）

    Returns:
        Dict: 搜索结果
            - topic_id: 主题ID
            - start_time: 开始时间戳
            - end_time: 结束时间戳
            - query: 查询语句
            - limit: 结果限制
            - total: 实际返回的日志条数
            - logs: 日志列表，每条日志包含:
                * timestamp: 日志时间（格式: YYYY-MM-DD HH:MM:SS）
                * level: 日志级别
                * message: 日志内容
            - took_ms: 查询耗时（毫秒）
            - message: 查询状态消息
    
    使用示例:
        # 步骤1: 获取当前时间戳
        current_ts = get_current_timestamp()  # 返回: 1708012345000
        
        # 步骤2: 计算开始时间（15分钟前）
        start_ts = current_ts - (15 * 60 * 1000)  # 1708011445000
        
        # 步骤3: 搜索日志
        search_log(
            topic_id="grid-topic-001",
            start_time=start_ts,     # int类型: 1708011445000
            end_time=current_ts,     # int类型: 1708012345000
            limit=100
        )
    """
    raw_start_time = start_time
    raw_end_time = end_time
    try:
        start_time = normalize_log_time(start_time)
        end_time = normalize_log_time(end_time)
    except ValueError as exc:
        return {
            "success": False,
            "topic_id": topic_id,
            "start_time": raw_start_time,
            "end_time": raw_end_time,
            "total": 0,
            "logs": [],
            "error": str(exc),
            "message": "日志查询时间格式无效",
        }

    if topic_id != "grid-topic-001":
        return {
            "topic_id": topic_id,
            "start_time": start_time,
            "end_time": end_time,
            "query": query,
            "limit": limit,
            "total": 0,
            "logs": [],
            "took_ms": 0,
            "error": f"主题不存在: {topic_id}",
            "message": f"错误: 未找到主题 {topic_id}，请检查 topic_id 是否正确"
        }

    requested_end_time = end_time
    time_range_adjusted = False
    if start_time >= end_time:
        # 模型偶尔会把 activeAt 同时填入起止时间。演示查询应容错为“从告警前 5 分钟
        # 到当前时间”，避免因为参数退化而丢失本来存在的故障日志。
        end_time = max(int(time.time() * 1000), start_time + 60 * 1000)
        time_range_adjusted = True

    requested_levels: list[str] = []
    if query:
        upper_query = query.upper()
        for candidate in ("ERROR", "WARN", "INFO"):
            if f"LEVEL:{candidate}" in upper_query:
                requested_levels.append(candidate)

    # 告警通常在异常持续一个评估周期后触发，根因日志会早于 activeAt。自动向前扩展
    # 5 分钟，避免只从 activeAt 开始查询而漏掉最初的异常日志。
    effective_start_time = max(0, start_time - 5 * 60 * 1000)

    started_at = time.perf_counter()
    try:
        body = query_grid_simulator_logs(
            {
                "start_time": effective_start_time,
                "end_time": end_time,
                "level": requested_levels[0] if len(requested_levels) == 1 else None,
                "limit": max(1, min(limit, 500)),
            }
        )
        logs = body.get("logs", [])
        if len(requested_levels) > 1:
            logs = [
                item for item in logs
                if str(item.get("level", "")).upper() in requested_levels
            ]
        return {
            "success": True,
            "source": "grid-simulator",
            "topic_id": topic_id,
            "service_name": "grid-data-sync-service",
            "scenario": body.get("scenario", "unknown"),
            "start_time": start_time,
            "effective_start_time": effective_start_time,
            "end_time": end_time,
            "requested_end_time": requested_end_time,
            "time_range_adjusted": time_range_adjusted,
            "query": query,
            "limit": limit,
            "total": len(logs),
            "logs": logs,
            "took_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "message": f"已从电网模拟服务查询 {len(logs)} 条业务日志",
        }
    except RuntimeError as exc:
        return {
            "success": False,
            "source": "grid-simulator",
            "topic_id": topic_id,
            "start_time": start_time,
            "end_time": end_time,
            "query": query,
            "limit": limit,
            "total": 0,
            "logs": [],
            "took_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "error": str(exc),
            "message": "电网模拟服务日志查询失败",
        }



if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8003, path="/mcp")
