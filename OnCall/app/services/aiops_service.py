"""
通用 Plan-Execute-Replan 服务
基于 LangGraph 官方教程实现
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from loguru import logger

from app.agent.aiops import PlanExecuteState, executor, planner, replanner
from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.models.incident import PendingIncident
from app.services.incident_memory_service import incident_memory_service

# 节点名称常量
NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"


class AIOpsService:
    """通用 Plan-Execute-Replan 服务"""

    def __init__(self):
        """初始化服务"""
        self.checkpointer = MemorySaver()
        self._alert_status_tool = None
        self.graph = self._build_graph()
        logger.info("Plan-Execute-Replan Service 初始化完成")

    def _build_graph(self):
        """构建 Plan-Execute-Replan 工作流"""
        logger.info("构建工作流图...")

        # 创建状态图
        workflow = StateGraph(PlanExecuteState)

        # 添加节点
        workflow.add_node(NODE_PLANNER, planner)      # 制定计划
        workflow.add_node(NODE_EXECUTOR, executor)  # 执行步骤
        workflow.add_node(NODE_REPLANNER, replanner)  # 重新规划

        # 设置入口点
        workflow.set_entry_point(NODE_PLANNER)

        # 定义边
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)     # planner -> executor
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)   # executor -> replanner

        # replanner 的条件边
        def should_continue(state: PlanExecuteState) -> str:
            """判断是否继续执行"""
            # 如果已经生成了最终响应，结束
            if state.get("response"):
                logger.info("已生成最终响应，结束流程")
                return END

            # 如果还有计划步骤，继续执行
            plan = state.get("plan", [])
            if plan:
                logger.info(f"继续执行，剩余 {len(plan)} 个步骤")
                return NODE_EXECUTOR

            # 计划为空但没有响应，返回 replanner 生成响应
            logger.info("计划执行完毕，生成最终响应")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER,
            should_continue,
            {
                NODE_EXECUTOR: NODE_EXECUTOR,
                END: END
            }
        )

        # 编译工作流
        compiled_graph = workflow.compile(checkpointer=self.checkpointer)

        logger.info("工作流图构建完成")
        return compiled_graph

    async def execute(
        self,
        user_input: str,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行 Plan-Execute-Replan 流程

        Args:
            user_input: 用户的任务描述
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 流式事件
        """
        logger.info(f"[会话 {session_id}] 开始执行任务: {user_input}")

        try:
            # 初始化状态
            initial_state: PlanExecuteState = {
                "input": user_input,
                "plan": [],
                "past_steps": [],
                "response": ""
            }

            # 流式执行工作流
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            # ``stream_mode=updates`` 返回的是每个节点的局部更新，而不是完整状态。
            # 因此进度必须由服务层独立维护，不能直接用 node_output 中 past_steps
            # 的长度计算，否则每轮都会从 1 开始，最终报告也不会被计入进度。
            planned_total_steps = 0
            completed_steps = 0
            remaining_steps = 0
            final_total_steps = 0
            skipped_steps = 0

            async for event in self.graph.astream(
                input=initial_state,
                config=config_dict,
                stream_mode="updates"
            ):
                # 解析事件
                for node_name, node_output in event.items():
                    logger.info(f"节点 '{node_name}' 输出事件")

                    # 根据节点类型生成不同的事件
                    if node_name == NODE_PLANNER:
                        plan = node_output.get("plan", []) if node_output else []
                        planned_total_steps = len(plan)
                        remaining_steps = len(plan)
                        yield self._format_planner_event(
                            node_output,
                            total_steps=planned_total_steps,
                        )

                    elif node_name == NODE_EXECUTOR:
                        new_past_steps = (
                            node_output.get("past_steps", []) if node_output else []
                        )
                        completed_steps += len(new_past_steps)
                        if node_output and "plan" in node_output:
                            remaining_steps = len(node_output.get("plan", []))
                        planned_total_steps = max(
                            planned_total_steps,
                            completed_steps + remaining_steps,
                        )
                        yield self._format_executor_event(
                            node_output,
                            completed_steps=completed_steps,
                            total_steps=planned_total_steps,
                            remaining_steps=remaining_steps,
                        )

                    elif node_name == NODE_REPLANNER:
                        if node_output and "plan" in node_output:
                            remaining_steps = len(node_output.get("plan", []))
                            planned_total_steps = completed_steps + remaining_steps

                        if node_output and node_output.get("response"):
                            # 最终报告本身是流程中的最后一步。若 Replanner 提前结束，
                            # 将其视为对原计划的动态收敛，并明确记录被跳过的步骤。
                            skipped_steps = max(0, remaining_steps - 1)
                            if remaining_steps > 0 or completed_steps == 0:
                                completed_steps += 1
                            remaining_steps = 0
                            final_total_steps = completed_steps

                        yield self._format_replanner_event(
                            node_output,
                            completed_steps=completed_steps,
                            total_steps=final_total_steps or planned_total_steps,
                            remaining_steps=remaining_steps,
                            skipped_steps=skipped_steps,
                        )

            # 获取最终状态
            final_state = self.graph.get_state(config_dict)
            final_response = ""

            # 安全地获取响应（处理 values 可能为 None 的情况）
            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")

            if final_total_steps == 0:
                final_total_steps = max(planned_total_steps, completed_steps)

            # 发送完成事件
            yield {
                "type": "complete",
                "stage": "complete",
                "message": self._format_completion_message(
                    completed_steps,
                    final_total_steps,
                    skipped_steps,
                ),
                "response": final_response,
                "completed_steps": completed_steps,
                "total_steps": final_total_steps,
                "remaining_steps": 0,
                "skipped_steps": skipped_steps,
            }

            logger.info(f"[会话 {session_id}] 任务执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "stage": "error",
                "message": f"任务执行出错: {str(e)}"
            }

    async def get_alert_status(self) -> Dict[str, Any]:
        """返回供前端状态栏使用的轻量活动告警摘要。"""
        try:
            alert_tool = getattr(self, "_alert_status_tool", None)
            if alert_tool is None:
                # 状态栏只依赖 Monitor MCP；CLS 暂时不可用时不应影响告警提示。
                mcp_client = await get_mcp_client_with_retry(
                    servers={"monitor": config.mcp_servers["monitor"]},
                    force_new=True,
                )
                mcp_tools = await mcp_client.get_tools()
                alert_tool = next(
                    (
                        tool
                        for tool in mcp_tools
                        if getattr(tool, "name", "") == "list_active_alerts"
                    ),
                    None,
                )
                if alert_tool is None:
                    raise RuntimeError("Monitor MCP 未提供 list_active_alerts 工具")
                self._alert_status_tool = alert_tool

            raw_result = await alert_tool.ainvoke({})
            active_alerts = self._parse_mcp_json_result(raw_result)
            if not active_alerts.get("success", False):
                raise RuntimeError(
                    str(active_alerts.get("message") or "Prometheus 告警查询失败")
                )
            return self._summarize_alert_status(active_alerts)
        except Exception as exc:
            # MCP 服务重启后丢弃旧工具包装器，下次轮询时重新发现工具。
            self._alert_status_tool = None
            logger.warning(f"获取 AIOps 告警状态失败: {exc}")
            return {
                "success": False,
                "status": "unavailable",
                "total": 0,
                "firing": 0,
                "pending": 0,
                "alerts": [],
                "message": f"监控连接异常: {exc}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    @staticmethod
    def _summarize_alert_status(active_alerts: Dict[str, Any]) -> Dict[str, Any]:
        """将 Monitor MCP 结果收敛为稳定的前端状态模型。"""
        alerts = [
            alert
            for alert in active_alerts.get("alerts", [])
            if isinstance(alert, dict)
        ]
        firing = sum(
            1 for alert in alerts if str(alert.get("state", "")).lower() == "firing"
        )
        pending = sum(
            1 for alert in alerts if str(alert.get("state", "")).lower() == "pending"
        )
        total = len(alerts)

        if firing > 0 or (total > 0 and pending == 0):
            status = "firing"
        elif pending > 0:
            status = "pending"
        else:
            status = "healthy"

        summaries = [
            {
                "alert_name": str(alert.get("alert_name", "未知告警")),
                "severity": str(alert.get("severity", "")),
                "service_name": str(alert.get("service_name", "")),
                "state": str(alert.get("state", "")),
                "summary": str(alert.get("summary", "")),
            }
            for alert in alerts[:5]
        ]
        return {
            "success": True,
            "status": status,
            "total": total,
            "firing": firing,
            "pending": pending,
            "alerts": summaries,
            "message": str(active_alerts.get("message", "")),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def diagnose(
        self,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        AIOps 诊断接口（兼容旧接口）

        Args:
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 诊断过程的流式事件
        """
        # 告警发现是确定性步骤：先直接调用 Monitor MCP，避免让 LLM 猜测过滤条件或
        # 误读工具结果。后续 Planner 以这份原始数据作为诊断事实基线。
        yield {
            "type": "status",
            "stage": "fetching_alerts",
            "message": "正在从 Prometheus 获取电网业务活动告警...",
        }
        mcp_tools = []
        try:
            mcp_client = await get_mcp_client_with_retry()
            mcp_tools = await mcp_client.get_tools()
            alert_tool = next(
                (tool for tool in mcp_tools if getattr(tool, "name", "") == "list_active_alerts"),
                None,
            )
            if alert_tool is None:
                raise RuntimeError("Monitor MCP 未提供 list_active_alerts 工具")
            raw_alert_result = await alert_tool.ainvoke({})
            active_alerts = self._parse_mcp_json_result(raw_alert_result)
        except Exception as exc:
            logger.error(f"[会话 {session_id}] 预取活动告警失败: {exc}", exc_info=True)
            active_alerts = {
                "success": False,
                "source": "prometheus",
                "alerts": [],
                "total": 0,
                "message": f"活动告警预取失败: {exc}",
            }

        alert_total = int(active_alerts.get("total", 0) or 0)
        logger.info(f"[会话 {session_id}] 预取活动告警完成: total={alert_total}")

        current_incident_id = None
        similar_incidents: list[dict[str, Any]] = []
        if alert_total > 0:
            fingerprint = incident_memory_service.build_alert_fingerprint(active_alerts)
            current_incident_id = f"inc-{fingerprint[:32]}"
            similar_incidents = await incident_memory_service.search_for_alerts(
                active_alerts,
                current_incident_id=current_incident_id,
            )
            logger.info(
                f"[会话 {session_id}] 历史情景记忆检索完成: "
                f"similar={len(similar_incidents)}"
            )

        # 使用固定的 AIOps 任务描述
        from textwrap import dedent
        alerts_json = json.dumps(active_alerts, ensure_ascii=False, indent=2)
        historical_incidents_json = incident_memory_service.compact_for_prompt(
            similar_incidents
        )
        aiops_task = dedent(f"""诊断当前电网数据采集与同步服务是否存在告警。如果存在告警，请基于 Prometheus 电网业务指标、同一时间窗口的服务日志和知识库处置手册分析原因并生成诊断报告；如果没有告警，请如实说明当前服务状态。

                PREFETCHED_ACTIVE_ALERTS_TOTAL={alert_total}
                PREFETCHED_ACTIVE_ALERTS_JSON:
                {alerts_json}

                CONFIRMED_HISTORICAL_INCIDENTS_JSON:
                {historical_incidents_json}

                上述预取告警是本次诊断的权威事实基线，不要再次使用带 severity、state 或 alert_name 过滤条件的查询覆盖它。诊断报告输出格式要求：
                ```
                # 告警分析报告

                ---

                ## 📋 活跃告警清单

                | 告警名称 | 级别 | 目标服务 | 首次触发时间 | 最新触发时间 | 状态 |
                |---------|------|----------|-------------|-------------|------|
                | [告警1名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |
                | [告警2名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |

                ---

                ## 🔍 告警根因分析1 - [告警名称]

                ### 告警详情
                - **告警级别**: [级别]
                - **受影响服务**: [服务名]
                - **持续时间**: [X分钟]

                ### 症状描述
                [根据监控指标描述症状]

                ### 日志证据
                [引用查询到的关键日志]

                ### 根因结论
                [基于证据得出的根本原因]

                ---

                ## 🛠️ 处理方案执行1 - [告警名称]

                ### 已执行的排查步骤
                1. [步骤1]
                2. [步骤2]

                ### 处理建议
                [给出具体的处理建议]

                ### 预期效果
                [说明预期的效果]

                ---

                ## 🔍 告警根因分析2 - [告警名称]
                [如果有第2个告警，重复上述格式]

                ---

                ## 📊 结论

                ### 整体评估
                [总结所有告警的整体情况]

                ### 关键发现
                - [发现1]
                - [发现2]

                ### 后续建议
                1. [建议1]
                2. [建议2]

                ### 风险评估
                [评估当前风险等级和影响范围]
                ```

                **重要提醒**：
                - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
                - 所有内容必须基于工具查询的真实数据，严禁编造
                - 当前 Prometheus 指标和本次日志是权威证据；已确认历史事件只能作为辅助参考，不得用历史根因替代本次证据
                - 如果引用历史事件，必须明确标注为“历史相似事件参考”，并保留 incident_id
                - 如果处置手册检索结果包含 [来源N]，处理建议和知识性结论必须保留对应引用，并在报告末尾列出实际引用的文件、章节和片段ID
                - 如果某个步骤失败，在结论中如实说明，不要跳过""")

        candidate: PendingIncident | None = None
        async for event in self.execute(aiops_task, session_id):
            if event.get("type") == "error":
                logger.warning(
                    f"[会话 {session_id}] LLM 诊断不可用，切换确定性电网诊断: "
                    f"{event.get('message', '')}"
                )
                async for fallback_event in self._diagnose_grid_deterministically(
                    active_alerts=active_alerts,
                    mcp_tools=mcp_tools,
                    similar_incidents=similar_incidents,
                ):
                    fallback_report = str(fallback_event.get("report") or "")
                    if fallback_report and candidate is None:
                        candidate = incident_memory_service.create_candidate(
                            session_id=session_id,
                            active_alerts=active_alerts,
                            report=fallback_report,
                            diagnosis_mode="deterministic_fallback",
                            similar_incidents=similar_incidents,
                        )
                    fallback_event = self._with_incident_metadata(
                        fallback_event,
                        candidate,
                        alert_total,
                        len(similar_incidents),
                    )
                    yield fallback_event
                return

            report = str(event.get("report") or event.get("response") or "")
            if report and candidate is None:
                candidate = incident_memory_service.create_candidate(
                    session_id=session_id,
                    active_alerts=active_alerts,
                    report=report,
                    diagnosis_mode="llm",
                    similar_incidents=similar_incidents,
                )

            # 转换事件格式以兼容旧的 API
            if event.get("type") == "complete":
                # 将 response 包装为 diagnosis 格式
                complete_event = {
                    "type": "complete",
                    "stage": "diagnosis_complete",
                    "message": event.get("message", "诊断流程完成"),
                    "completed_steps": event.get("completed_steps", 0),
                    "total_steps": event.get("total_steps", 0),
                    "remaining_steps": event.get("remaining_steps", 0),
                    "skipped_steps": event.get("skipped_steps", 0),
                    "diagnosis": {
                        "status": "completed",
                        "report": event.get("response", "")
                    }
                }
                yield self._with_incident_metadata(
                    complete_event,
                    candidate,
                    alert_total,
                    len(similar_incidents),
                )
            else:
                yield self._with_incident_metadata(
                    event,
                    candidate,
                    alert_total,
                    len(similar_incidents),
                )

    @staticmethod
    def _with_incident_metadata(
        event: Dict[str, Any],
        candidate: PendingIncident | None,
        alert_total: int,
        similar_incident_count: int,
    ) -> Dict[str, Any]:
        """向报告/完成事件附加人工确认所需的稳定字段。"""
        enriched = dict(event)
        enriched["has_active_alerts"] = alert_total > 0
        enriched["similar_incident_count"] = similar_incident_count
        if candidate is not None:
            enriched["incident_id"] = candidate.incident_id
            enriched["can_confirm"] = candidate.status == "pending"
            enriched["incident_status"] = candidate.status
        else:
            enriched["incident_id"] = None
            enriched["can_confirm"] = False
            enriched["incident_status"] = None
        return enriched

    async def _diagnose_grid_deterministically(
        self,
        active_alerts: Dict[str, Any],
        mcp_tools: list[Any],
        similar_incidents: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """外部模型不可用时，仍基于实时证据完成可演示的电网诊断。"""
        yield {
            "type": "status",
            "stage": "deterministic_fallback",
            "message": "AI 模型暂不可用，已切换本地确定性诊断模式...",
        }

        if not mcp_tools:
            try:
                mcp_client = await get_mcp_client_with_retry()
                mcp_tools = await mcp_client.get_tools()
            except Exception as exc:
                logger.error(f"确定性诊断加载 MCP 工具失败: {exc}", exc_info=True)

        alerts = active_alerts.get("alerts", [])
        if not alerts:
            report = self._format_no_alert_report(active_alerts)
            yield {
                "type": "report",
                "stage": "final_report",
                "message": "诊断流程完成 (1/1)",
                "report": report,
                "mode": "deterministic_fallback",
                "completed_steps": 1,
                "total_steps": 1,
                "remaining_steps": 0,
            }
            yield {
                "type": "complete",
                "stage": "diagnosis_complete",
                "message": "诊断流程完成 (1/1)",
                "completed_steps": 1,
                "total_steps": 1,
                "remaining_steps": 0,
                "diagnosis": {
                    "status": "completed",
                    "mode": "deterministic_fallback",
                    "report": report,
                },
            }
            return

        evidence_records = []
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        total_steps = len(alerts) * 4
        completed_steps = 0

        for alert in alerts:
            alert_name = str(alert.get("alert_name", "unknown"))
            service_name = str(
                alert.get("service_name") or "grid-data-sync-service"
            )
            start_time = str(alert.get("active_at") or now)
            metric_tool_name = self._metric_tool_for_alert(alert_name)

            metrics = await self._call_mcp_tool_json(
                mcp_tools,
                metric_tool_name,
                {
                    "service_name": service_name,
                    "start_time": start_time,
                    "end_time": now,
                    "interval": "5s",
                },
            )
            completed_steps += 1
            yield self._fallback_step_event(
                completed_steps,
                total_steps,
                f"已查询 {alert_name} 对应的 Prometheus 电网业务指标",
            )

            topic = await self._call_mcp_tool_json(
                mcp_tools,
                "search_topic_by_service_name",
                {"service_name": service_name, "fuzzy": True},
            )
            completed_steps += 1
            yield self._fallback_step_event(
                completed_steps,
                total_steps,
                f"已定位 {service_name} 的业务日志主题",
            )

            topic_id = "grid-topic-001"
            topics = topic.get("topics", []) if isinstance(topic, dict) else []
            if topics:
                topic_id = str(topics[0].get("topic_id") or topic_id)
            logs = await self._call_mcp_tool_json(
                mcp_tools,
                "search_log",
                {
                    "topic_id": topic_id,
                    "start_time": start_time,
                    "end_time": now,
                    "query": "level:WARN OR level:ERROR",
                    "limit": 100,
                },
            )
            completed_steps += 1
            yield self._fallback_step_event(
                completed_steps,
                total_steps,
                f"已查询 {alert_name} 告警窗口内的 WARN/ERROR 业务日志",
            )

            knowledge = self._load_local_grid_runbook(alert_name)
            completed_steps += 1
            yield self._fallback_step_event(
                completed_steps,
                total_steps,
                f"已读取 {alert_name} 的本地处置手册",
            )

            evidence_records.append(
                {
                    "alert": alert,
                    "metrics": metrics,
                    "topic": topic,
                    "logs": logs,
                    "knowledge": knowledge,
                }
            )

        report = self._format_deterministic_grid_report(
            evidence_records,
            now,
            similar_incidents=similar_incidents or [],
        )
        yield {
            "type": "report",
            "stage": "final_report",
            "message": f"诊断流程完成 ({total_steps}/{total_steps})",
            "report": report,
            "mode": "deterministic_fallback",
            "completed_steps": total_steps,
            "total_steps": total_steps,
            "remaining_steps": 0,
        }
        yield {
            "type": "complete",
            "stage": "diagnosis_complete",
            "message": f"诊断流程完成 ({total_steps}/{total_steps})",
            "completed_steps": total_steps,
            "total_steps": total_steps,
            "remaining_steps": 0,
            "diagnosis": {
                "status": "completed",
                "mode": "deterministic_fallback",
                "report": report,
            },
        }

    @staticmethod
    def _fallback_step_event(
        completed_steps: int,
        total_steps: int,
        current_step: str,
    ) -> Dict[str, Any]:
        return {
            "type": "step_complete",
            "stage": "step_executed",
            "message": f"步骤执行完成 ({completed_steps}/{total_steps})",
            "current_step": current_step,
            "remaining_steps": total_steps - completed_steps,
            "mode": "deterministic_fallback",
        }

    async def _call_mcp_tool_json(
        self,
        mcp_tools: list[Any],
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        tool = next(
            (item for item in mcp_tools if getattr(item, "name", "") == tool_name),
            None,
        )
        if tool is None:
            return {
                "success": False,
                "tool": tool_name,
                "message": f"MCP 工具不存在: {tool_name}",
            }
        try:
            return self._parse_mcp_json_result(await tool.ainvoke(arguments))
        except Exception as exc:
            logger.error(f"确定性诊断调用 {tool_name} 失败: {exc}", exc_info=True)
            return {
                "success": False,
                "tool": tool_name,
                "message": str(exc),
            }

    @staticmethod
    def _metric_tool_for_alert(alert_name: str) -> str:
        if alert_name in {
            "GridTelemetryQueueBacklog",
            "GridDataSyncFailureRateHigh",
        }:
            return "query_grid_data_sync_metrics"
        if alert_name in {
            "GridDataFreshnessDelayHigh",
            "GridTelemetryProcessingLatencyHigh",
        }:
            return "query_grid_telemetry_metrics"
        return "query_grid_service_status"

    @staticmethod
    def _load_local_grid_runbook(alert_name: str) -> Dict[str, Any]:
        document_mapping = {
            "GridDataSyncServiceDown": "grid_service_down.md",
            "GridCommunicationInterrupted": "grid_communication_interrupted.md",
            "GridTelemetryQueueBacklog": "grid_queue_backlog.md",
            "GridDataSyncFailureRateHigh": "grid_sync_failure.md",
            "GridDataFreshnessDelayHigh": "grid_data_delay.md",
            "GridTelemetryProcessingLatencyHigh": "grid_data_delay.md",
        }
        filename = document_mapping.get(alert_name)
        if not filename:
            return {"success": False, "content": "未找到对应的本地处置手册"}

        path = Path(__file__).resolve().parents[2] / "aiops-docs" / filename
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            return {
                "success": False,
                "source": filename,
                "content": f"读取本地处置手册失败: {exc}",
            }
        return {"success": True, "source": filename, "content": content}

    @staticmethod
    def _markdown_section(content: str, heading: str) -> str:
        marker = f"## {heading}"
        if marker not in content:
            return ""
        section = content.split(marker, 1)[1]
        if "\n## " in section:
            section = section.split("\n## ", 1)[0]
        return section.strip()

    @classmethod
    def _format_deterministic_grid_report(
        cls,
        evidence_records: list[Dict[str, Any]],
        generated_at: str,
        similar_incidents: list[dict[str, Any]] | None = None,
    ) -> str:
        lines = [
            "# 告警分析报告",
            "",
            "> 诊断模式：本地确定性降级模式。外部 AI 模型不可用时，系统仍基于实时 Prometheus、业务日志和本地处置手册生成报告。",
            "",
            "## 活跃告警清单",
            "",
            "| 告警名称 | 级别 | 目标服务 | 首次触发时间 | 报告时间 | 状态 |",
            "|---|---|---|---|---|---|",
        ]
        for record in evidence_records:
            alert = record["alert"]
            lines.append(
                "| {name} | {severity} | {service} | {active_at} | {now} | 活跃 |".format(
                    name=alert.get("alert_name", "unknown"),
                    severity=alert.get("severity", "unknown"),
                    service=alert.get("service_name", "unknown"),
                    active_at=alert.get("active_at", "unknown"),
                    now=generated_at,
                )
            )

        for index, record in enumerate(evidence_records, 1):
            alert = record["alert"]
            alert_name = str(alert.get("alert_name", "unknown"))
            metrics = record["metrics"]
            logs = record["logs"]
            knowledge = record["knowledge"]
            lines.extend(
                [
                    "",
                    f"## 告警根因分析 {index} - {alert_name}",
                    "",
                    "### 告警详情",
                    "",
                    f"- 告警级别：{alert.get('severity', 'unknown')}",
                    f"- 受影响服务：{alert.get('service_name', 'unknown')}",
                    f"- 持续时间：{alert.get('duration', 'unknown')}",
                    f"- 告警描述：{alert.get('description', '')}",
                    "",
                    "### 指标证据",
                    "",
                ]
            )
            metric_items = metrics.get("metrics", {}) if isinstance(metrics, dict) else {}
            if metric_items:
                for metric in metric_items.values():
                    stats = metric.get("statistics", {})
                    lines.append(
                        "- {description} (`{name}`)：当前值 {current} {unit}，最大值 {maximum} {unit}".format(
                            description=metric.get("description", "业务指标"),
                            name=metric.get("metric_name", "unknown"),
                            current=stats.get("current", "无数据"),
                            maximum=stats.get("max", "无数据"),
                            unit=metric.get("unit", ""),
                        )
                    )
            else:
                lines.append(f"- 指标查询未获得有效数据：{metrics.get('message', 'unknown')}")

            lines.extend(["", "### 日志证据", ""])
            log_items = logs.get("logs", []) if isinstance(logs, dict) else []
            if log_items:
                for item in log_items[:3]:
                    lines.append(
                        f"- `{item.get('timestamp', 'unknown')}` "
                        f"`{item.get('level', 'unknown')}` {item.get('message', '')}"
                    )
            else:
                lines.append(f"- 未查询到业务日志：{logs.get('message', 'unknown')}")

            common_causes = cls._markdown_section(
                str(knowledge.get("content", "")), "常见原因"
            )
            recommendations = cls._markdown_section(
                str(knowledge.get("content", "")), "处置建议"
            )
            root_log = log_items[0].get("message", "") if log_items else "无关键日志"
            anomalies = metrics.get("anomalies", []) if isinstance(metrics, dict) else []
            lines.extend(
                [
                    "",
                    "### 根因结论",
                    "",
                    f"实时日志显示“{root_log}”；Prometheus 同时检测到 {len(anomalies)} 项阈值异常。结合指标与处置手册，当前故障与数据处理消费能力低于采集压力相符。",
                    "",
                    "### 处置手册中的常见原因",
                    "",
                    common_causes or "本地处置手册未提供常见原因。",
                    "",
                    f"## 处理建议 {index} - {alert_name}",
                    "",
                    recommendations or "本地处置手册未提供处置建议。",
                    "",
                    f"- 处置手册来源：`{knowledge.get('source', 'unknown')}`",
                ]
            )

        if similar_incidents:
            lines.extend(
                [
                    "",
                    "## 历史相似事件参考",
                    "",
                    "> 以下内容来自人工确认过的历史事件，仅用于辅助比对；本次指标和日志仍是根因判断的权威证据。",
                    "",
                ]
            )
            for item in similar_incidents[: config.incident_rrf_final_k]:
                lines.append(
                    "- `{incident_id}`：{alert_name} / {service_name}（RRF {score:.6f}）".format(
                        incident_id=item.get("incident_id", "unknown"),
                        alert_name=item.get("alert_name", "unknown"),
                        service_name=item.get("service_name", "unknown"),
                        score=float(item.get("rrf_score", 0.0)),
                    )
                )

        lines.extend(
            [
                "",
                "## 结论",
                "",
                f"当前共有 {len(evidence_records)} 条活动告警。报告中的告警、指标和日志均来自本次实时查询；建议按本地处置手册处理，并在恢复后确认 Prometheus 告警清零。",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _format_no_alert_report(active_alerts: Dict[str, Any]) -> str:
        return (
            "# 告警分析报告\n\n"
            "> 诊断模式：本地确定性降级模式。\n\n"
            "## 当前状态\n\n"
            f"Prometheus 当前活动告警数为 {active_alerts.get('total', 0)}，"
            "未发现需要诊断的电网业务告警。"
        )

    @staticmethod
    def _parse_mcp_json_result(result: Any) -> Dict[str, Any]:
        """解析 MCP adapter 返回的 TextContent 列表或 JSON 字符串。"""
        if isinstance(result, list) and result:
            block = result[0]
            if isinstance(block, dict):
                result = block.get("text", "")
            else:
                result = getattr(block, "text", str(block))
        if isinstance(result, str):
            parsed = json.loads(result)
            if not isinstance(parsed, dict):
                raise ValueError("MCP 告警结果不是 JSON 对象")
            return parsed
        if isinstance(result, dict):
            return result
        raise TypeError(f"无法解析 MCP 告警结果: {type(result)!r}")

    def _format_planner_event(
        self,
        state: Dict | None,
        total_steps: int | None = None,
    ) -> Dict:
        """格式化 Planner 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "planner",
                "message": "规划节点执行中"
            }

        plan = state.get("plan", [])
        total = len(plan) if total_steps is None else total_steps

        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"执行计划已制定，共 {total} 个步骤",
            "plan": plan,
            "completed_steps": 0,
            "total_steps": total,
            "remaining_steps": total,
        }

    @staticmethod
    def _format_completion_message(
        completed_steps: int,
        total_steps: int,
        skipped_steps: int = 0,
    ) -> str:
        if not total_steps:
            return "诊断流程完成"
        message = f"诊断流程完成 ({completed_steps}/{total_steps})"
        if skipped_steps:
            message += f"，重规划后省略 {skipped_steps} 个不再需要的步骤"
        return message

    def _format_executor_event(
        self,
        state: Dict | None,
        completed_steps: int | None = None,
        total_steps: int | None = None,
        remaining_steps: int | None = None,
    ) -> Dict:
        """格式化 Executor 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "executor",
                "message": "执行节点运行中"
            }

        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])

        if past_steps:
            last_step, _ = past_steps[-1]
            completed = len(past_steps) if completed_steps is None else completed_steps
            remaining = len(plan) if remaining_steps is None else remaining_steps
            total = completed + remaining if total_steps is None else total_steps
            return {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"步骤执行完成 ({completed}/{total})",
                "current_step": last_step,
                "completed_steps": completed,
                "total_steps": total,
                "remaining_steps": remaining,
            }
        else:
            return {
                "type": "status",
                "stage": "executor",
                "message": "开始执行步骤"
            }

    def _format_replanner_event(
        self,
        state: Dict | None,
        completed_steps: int = 0,
        total_steps: int = 0,
        remaining_steps: int | None = None,
        skipped_steps: int = 0,
    ) -> Dict:
        """格式化 Replanner 节点事件"""
        if not state:
            progress = (
                f" ({completed_steps}/{total_steps})"
                if total_steps
                else ""
            )
            return {
                "type": "status",
                "stage": "replanner",
                "message": f"评估完成，正在执行下一步骤{progress}...",
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "remaining_steps": (
                    remaining_steps
                    if remaining_steps is not None
                    else max(0, total_steps - completed_steps)
                ),
            }

        response = state.get("response", "")
        plan = state.get("plan", [])
        remaining = len(plan) if remaining_steps is None else remaining_steps

        if response:
            # 已生成最终响应
            return {
                "type": "report",
                "stage": "final_report",
                "message": self._format_completion_message(
                    completed_steps,
                    total_steps,
                    skipped_steps,
                ),
                "report": response,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "remaining_steps": 0,
                "skipped_steps": skipped_steps,
            }
        else:
            # 重新规划
            return {
                "type": "status",
                "stage": "replanner",
                "message": f"评估完成，{'继续执行剩余步骤' if remaining else '准备生成最终响应'}",
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "remaining_steps": remaining,
            }


# 全局单例
aiops_service = AIOpsService()
