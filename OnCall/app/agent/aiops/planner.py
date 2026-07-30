"""
Planner 节点：制定执行计划
基于 LangGraph 官方教程实现
"""

from textwrap import dedent
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
from .state import PlanExecuteState
from .utils import (
    format_tools_description,
    normalize_plan_steps,
    plan_covers_active_alert_diagnosis,
)


class Plan(BaseModel):
    """计划的输出格式"""
    steps: List[str] = Field(
        description="完成任务所需的不同步骤。这些步骤应该按顺序执行，每一步都建立在前一步的基础上。"
    )


def _prefetched_alert_total(input_text: str) -> int:
    marker = "PREFETCHED_ACTIVE_ALERTS_TOTAL="
    if marker not in input_text:
        return 0
    try:
        return int(input_text.split(marker, 1)[1].splitlines()[0].strip())
    except (ValueError, IndexError):
        return 0


def _standard_active_alert_plan() -> List[str]:
    """模型计划结构异常时使用的稳定电网告警诊断计划。"""
    return [
        "确认 PREFETCHED_ACTIVE_ALERTS_JSON 中的活动告警、受影响服务与触发时间",
        "根据告警类型使用 query_grid_service_status、query_grid_data_sync_metrics 或 query_grid_telemetry_metrics 查询同一服务在告警时间窗口内的业务指标",
        "使用 search_topic_by_service_name 获取受影响服务的日志主题",
        "使用 search_log 查询告警时间窗口内的 WARN/ERROR 业务日志",
        "使用 retrieve_knowledge 检索告警对应的知识库处置手册",
        "综合预取告警、业务指标、业务日志和处置手册生成诊断报告",
    ]


def _standard_no_alert_plan() -> List[str]:
    return [
        "确认 PREFETCHED_ACTIVE_ALERTS_JSON 中当前没有活动告警",
        "基于 Prometheus 当前状态生成无活动告警的诊断报告",
    ]


def _ensure_plan_coverage(input_text: str, plan_steps: List[str]) -> List[str]:
    """活动告警计划不完整时返回稳定的标准诊断计划。"""
    if (
        _prefetched_alert_total(input_text) > 0
        and not plan_covers_active_alert_diagnosis(plan_steps)
    ):
        return _standard_active_alert_plan()
    return plan_steps


# Planner 提示词
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个专家级别的规划者，你需要将复杂的任务分解为可执行的步骤。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定计划，实际的工具调用由 Executor 负责执行。

                {experience_context}

                对于给定的任务，请创建一个简单的、逐步的计划来完成它。计划应该：
                - 将任务分解为逻辑上独立的步骤
                - 每个步骤应该明确使用哪些工具(如果需要工具的话)来获取信息, 最好能同时提供工具执行所需要的参数
                - 步骤之间应该有清晰的依赖关系
                - 步骤描述要具体、可操作
                - **如果有相关经验文档，请参考其中的方法和步骤制定计划**
                - 查询监控告警时，优先使用 Monitor MCP 的 list_active_alerts 工具
                - 查询服务健康和站点在线情况时，使用 query_grid_service_status 工具
                - 查询队列积压和同步失败率时，使用 query_grid_data_sync_metrics 工具
                - 查询数据新鲜度和处理延迟时，使用 query_grid_telemetry_metrics 工具
                - 不要自行编写 PromQL；PromQL 由 Monitor MCP 内部管理
                - 先通过告警确定受影响服务，再使用同一 service_name 查询对应指标
                - 日志、指标和知识文档必须围绕同一告警及同一时间范围，禁止编造证据
                - 如果输入包含 PREFETCHED_ACTIVE_ALERTS_JSON，它是已经无过滤查询得到的权威告警数据，不要再规划 list_active_alerts 步骤
                - PREFETCHED_ACTIVE_ALERTS_TOTAL 大于 0 时，计划必须分别包含业务指标、日志和知识手册查询，不能直接生成报告
                - search_log 依赖日志 topic_id，因此 search_topic_by_service_name 和 search_log 必须拆成前后两个步骤
                - 每个步骤只放置同一依赖层级的工具调用，保证 Executor 能获得前序结果后再执行下一步

                示例输入："诊断当前电网数据采集与同步服务是否存在告警"
                示例输出（假设有对应工具）：
                步骤1: 使用 list_active_alerts 查询当前活动告警并确定受影响服务
                步骤2: 根据告警类型使用对应的 grid 业务指标工具查询异常趋势
                步骤3: 使用日志工具检查告警时间窗口内的业务错误日志
                步骤4: 使用 retrieve_knowledge 检索该告警对应的处置手册
                步骤5: 综合告警、业务指标、日志和处置手册生成诊断报告
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


async def planner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    规划节点：根据用户输入生成执行计划

    流程：
    1. 先查询内部文档，获取相关经验和最佳实践
    2. 基于经验文档和可用工具制定执行计划
    """
    logger.info("=== Planner：制定执行计划 ===")

    input_text = state.get("input", "")
    logger.info(f"用户输入: {input_text}")

    try:
        # 步骤1: 查询内部文档获取相关经验
        logger.info("查询内部文档，寻找相关经验...")
        experience_docs = ""
        try:
            # retrieve_knowledge 使用 response_format="content_and_artifact"
            # ainvoke() 只返回 content（字符串），不是元组
            context_str = await retrieve_knowledge.ainvoke({"query": input_text})
            if context_str and context_str.strip():
                experience_docs = context_str
                logger.info(f"找到相关经验文档，长度: {len(experience_docs)}")
            else:
                logger.info("未找到相关经验文档")
        except Exception as e:
            logger.warning(f"查询内部文档失败: {e}")

        # 步骤2: 获取可用工具列表
        # 获取本地工具
        local_tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()

        # 合并所有工具
        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 格式化工具描述
        tools_description = format_tools_description(all_tools)

        # 步骤3: 格式化经验文档上下文
        if experience_docs:
            experience_context = dedent(f"""
                ## 相关经验文档

                以下是从知识库中检索到的相关经验和最佳实践，请参考这些经验制定执行计划：

                {experience_docs}

                ---
            """).strip()
        else:
            experience_context = ""

        # 步骤4: 创建 LLM 并生成计划
        logger.info("Planner 模型: {}", config.aiops_planner_model)
        llm = ChatQwen(
            model=config.aiops_planner_model,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0,
            max_tokens=config.aiops_planner_max_output_tokens,
            enable_thinking=False,
        )

        planner_chain = planner_prompt | llm.with_structured_output(Plan)

        # 调用 LLM 生成计划
        plan_result = await planner_chain.ainvoke({
            "messages": [("user", input_text)],
            "tools_description": tools_description,
            "experience_context": experience_context
        })

        # 提取步骤列表
        if isinstance(plan_result, Plan):
            plan_steps = plan_result.steps
        else:
            # 如果返回的是字典，提取 steps 字段
            plan_steps = plan_result.get("steps", [])  # type: ignore

        plan_steps = normalize_plan_steps(plan_steps)

        validated_plan_steps = _ensure_plan_coverage(input_text, plan_steps)
        if validated_plan_steps != plan_steps:
            logger.warning(
                "模型计划未覆盖完整活动告警证据链，使用标准 6 步诊断计划"
            )
            plan_steps = validated_plan_steps

        logger.info(f"计划已生成，共 {len(plan_steps)} 个步骤")
        for i, step in enumerate(plan_steps, 1):
            logger.info(f"  步骤{i}: {step}")

        return {"plan": plan_steps}

    except Exception as e:
        logger.error(f"生成计划失败: {e}", exc_info=True)
        fallback_plan = (
            _standard_active_alert_plan()
            if _prefetched_alert_total(input_text) > 0
            else _standard_no_alert_plan()
        )
        return {"plan": fallback_plan}
