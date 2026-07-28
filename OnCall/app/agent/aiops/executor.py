"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现
"""

from datetime import datetime, timedelta, timezone
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.config import config
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS
from app.agent.mcp_client import get_mcp_client_with_retry
from .state import PlanExecuteState


GRID_SERVICE_TOOLS = {
    "query_grid_service_status",
    "query_grid_data_sync_metrics",
    "query_grid_telemetry_metrics",
}


def _prefetched_alert_context(input_text: str) -> dict[str, str]:
    """从权威预取告警 JSON 中提取工具调用所需的稳定参数。"""
    marker = "PREFETCHED_ACTIVE_ALERTS_JSON:"
    if marker not in input_text:
        return {}

    payload = input_text.split(marker, 1)[1].lstrip()
    try:
        parsed, _ = json.JSONDecoder().raw_decode(payload)
        alert = parsed.get("alerts", [])[0]
    except (json.JSONDecodeError, IndexError, AttributeError, TypeError):
        return {}

    return {
        "service_name": str(alert.get("service_name", "")),
        "active_at": str(alert.get("active_at", "")),
        "alert_name": str(alert.get("alert_name", "")),
    }


def _normalize_grid_tool_calls(
    tool_calls: list[dict[str, Any]],
    input_text: str,
    now: datetime | None = None,
) -> None:
    """用预取告警校正电网诊断工具参数，避免模型拼写和占位符漂移。"""
    context = _prefetched_alert_context(input_text)
    service_name = context.get("service_name") or "grid-data-sync-service"
    now = now or datetime.now(timezone.utc)
    end_time = now.isoformat().replace("+00:00", "Z")
    start_time = context.get("active_at") or (
        now - timedelta(hours=1)
    ).isoformat().replace("+00:00", "Z")

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        name = str(tool_call.get("name", ""))
        args = tool_call.get("args")
        if not isinstance(args, dict):
            continue

        if name == "list_active_alerts":
            # 活动告警已无过滤预取；如果模型仍调用，禁止错误过滤掉权威事实。
            args.clear()
        elif name in GRID_SERVICE_TOOLS:
            args.update(
                {
                    "service_name": service_name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "interval": "5s",
                }
            )
        elif name == "search_topic_by_service_name":
            args.update({"service_name": service_name, "fuzzy": True})
        elif name == "search_log":
            args.update(
                {
                    "topic_id": "grid-topic-001",
                    "start_time": start_time,
                    "end_time": end_time,
                    "limit": 100,
                }
            )


def _format_tool_evidence(tool_messages: list[Any]) -> str:
    """保留工具原始返回，防止二次摘要把成功结果误写为失败。"""
    evidence = []
    for message in tool_messages:
        name = getattr(message, "name", "tool") or "tool"
        content = getattr(message, "content", str(message))
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        evidence.append(f"[{name}]\n{content[:12000]}")
    return "\n\n".join(evidence)


async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤
    
    使用 LangGraph 的 ToolNode 自动处理工具调用
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])
    input_text = state.get("input", "")
    past_steps = state.get("past_steps", [])

    # 如果计划为空，不执行
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 取出第一个步骤
    task = plan[0]
    logger.info(f"当前任务: {task}")

    try:
        # 获取本地工具
        local_tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 合并所有工具
        all_tools = local_tools + mcp_tools

        # 创建 LLM（绑定工具）
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(all_tools)

        # 创建工具节点（自动执行工具调用）
        tool_node = ToolNode(all_tools)

        # 将原始任务中的预取告警和前序结果一并提供给 Executor，保证后续步骤能使用
        # 已确定的 service_name、告警时间和日志 topic_id，而不是重新猜测参数。
        history_context = "\n\n".join(
            f"步骤: {step}\n结果: {result[:2000]}"
            for step, result in past_steps[-3:]
        )
        messages = [
            SystemMessage(content="""你是一个能力强大的助手，负责执行具体的任务步骤。

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
3. 调用工具获取信息
4. 返回执行结果

注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要考虑其他任务"""),
            HumanMessage(
                content=(
                    f"原始任务与预取告警:\n{input_text[:6000]}\n\n"
                    f"已执行步骤结果:\n{history_context or '无'}\n\n"
                    f"本轮当前 UTC 时间: {datetime.now(timezone.utc).isoformat()}\n"
                    "必须使用预取告警中的精确 service_name 和 active_at；"
                    "不要在工具参数中使用 placeholder。\n\n"
                    f"请只执行当前任务:\n{task}"
                )
            )
        ]

        # 第一步：LLM 决定是否调用工具
        llm_response = await llm_with_tools.ainvoke(messages)
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")

            _normalize_grid_tool_calls(
                llm_response.tool_calls,
                input_text,
            )
            logger.info(f"校正后的工具调用: {llm_response.tool_calls}")

            # 使用 ToolNode 自动执行工具
            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})

            # 第三步：将工具结果返回给 LLM 生成最终答案
            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)
            summary = final_response.content if hasattr(final_response, 'content') else str(final_response)
            result = (
                "工具原始返回（权威证据）:\n"
                f"{_format_tool_evidence(tool_messages['messages'])}\n\n"
                "执行摘要:\n"
                f"{summary}"
            )
        else:
            # 没有工具调用，直接使用 LLM 的输出
            logger.info("LLM 未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        # 返回更新：移除已执行的步骤，添加执行历史
        return {
            "plan": plan[1:],  # 移除第一个步骤
            "past_steps": [(task, result)],  # 使用 operator.add 追加
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {str(e)}")],
        }
