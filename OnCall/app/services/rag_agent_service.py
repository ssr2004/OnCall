"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_qwq import ChatQwen
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict

from app.agent.mcp_client import (
    format_exception_chain,
    get_mcp_client_with_retry,
    load_mcp_tools_safe,
    suggest_mcp_transport,
)
from app.config import config
from app.services.conversation_context_service import (
    CONVERSATION_SUMMARY_PROMPT,
    OldToolOutputPruningMiddleware,
    SafeSummarizationMiddleware,
    build_conversation_context_budget,
    count_qwen_tokens_conservatively,
    estimate_fixed_prompt_tokens,
)
from app.services.conversation_transcript_service import (
    ConversationTranscriptStore,
    default_conversation_data_dir,
)
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS

# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """
    修剪消息历史，只保留最近的几条消息以适应上下文窗口

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 6 条消息（3 轮对话）
    - 当消息少于等于 7 条时，不做修剪

    Args:
        state: Agent 状态

    Returns:
        包含修剪后消息的字典，如果无需修剪则返回 None
    """
    messages = state["messages"]

    # 如果消息数量较少，无需修剪
    if len(messages) <= 7:
        return None

    # 提取第一条系统消息
    first_msg = messages[0]

    # 保留最近的 6 条消息（确保包含完整的对话轮次）
    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    # 构建新的消息列表
    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史: {len(messages)} -> {len(new_messages)} 条")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.chat_model
        self.summary_model_name = config.chat_summary_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()


        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0.7,
            streaming=streaming,
            max_tokens=config.chat_model_max_output_tokens,
            enable_thinking=False,
        )
        self.summary_model = ChatQwen(
            model=self.summary_model_name,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0,
            streaming=False,
            max_tokens=config.chat_summary_max_output_tokens,
            enable_thinking=False,
        )

        # 定义基础工具（与 AIOps Planner/Executor 使用同一套默认本地工具）
        self.tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 创建内存检查点（用于会话管理）
        self.checkpointer = MemorySaver()

        # 完整会话使用本地追加式 Transcript 持久化；MemorySaver 仅保存当前
        # 进程内发给模型的工作上下文。
        self.transcript_store = ConversationTranscriptStore(
            default_conversation_data_dir(config.chat_session_data_dir)
        )
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._restored_sessions: set[str] = set()
        logger.info(
            "对话 Transcript 目录: {}",
            self.transcript_store.root_dir,
        )

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(
            "RAG Agent 服务初始化完成 (ChatQwen), model={}, "
            "summary_model={}, streaming={}",
            self.model_name,
            self.summary_model_name,
            streaming,
        )

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        for name, server in config.mcp_servers.items():
            hint = suggest_mcp_transport(
                str(server.get("url", "")),
                str(server.get("transport", "")),
            )
            if hint:
                logger.warning(f"MCP 配置 [{name}]: {hint}")

        mcp_client = await get_mcp_client_with_retry()
        mcp_tools, mcp_err = await load_mcp_tools_safe(mcp_client)
        if mcp_err:
            logger.warning(
                f"MCP 工具加载失败，将仅使用本地工具继续运行:\n{mcp_err}"
            )
            self.mcp_tools = []
        else:
            self.mcp_tools = mcp_tools
            logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        all_tools = self.tools + self.mcp_tools
        fixed_prompt_tokens = estimate_fixed_prompt_tokens(
            self.system_prompt,
            all_tools,
        )
        self.context_budget = build_conversation_context_budget(
            context_window_tokens=config.chat_model_context_window_tokens,
            max_input_tokens=config.chat_model_max_input_tokens,
            max_output_tokens=config.chat_model_max_output_tokens,
            operating_input_tokens=config.chat_context_operating_input_tokens,
            fixed_prompt_tokens=fixed_prompt_tokens,
        )
        logger.info(
            "对话上下文预算: model={}, hard_input={}, operating_input={}, "
            "fixed_prompt={}, message_trigger={}, keep_recent={}",
            self.model_name,
            self.context_budget.hard_input_tokens,
            self.context_budget.effective_input_tokens,
            self.context_budget.fixed_prompt_tokens,
            self.context_budget.message_trigger_tokens,
            self.context_budget.keep_recent_tokens,
        )

        self.agent = create_agent(
            self.model,
            tools=all_tools,
            system_prompt=self.system_prompt,
            middleware=[
                OldToolOutputPruningMiddleware(
                    trigger_tokens=self.context_budget.message_trigger_tokens,
                    keep_recent_tokens=self.context_budget.keep_recent_tokens,
                    max_tool_output_chars=config.chat_tool_output_max_chars,
                    token_counter=count_qwen_tokens_conservatively,
                ),
                SafeSummarizationMiddleware(
                    model=self.summary_model,
                    trigger=(
                        "tokens",
                        self.context_budget.message_trigger_tokens,
                    ),
                    keep=("tokens", self.context_budget.keep_recent_tokens),
                    token_counter=count_qwen_tokens_conservatively,
                    summary_prompt=CONVERSATION_SUMMARY_PROMPT,
                    trim_tokens_to_summarize=None,
                ),
            ],
            checkpointer=self.checkpointer,
        )

        self._agent_initialized = True


        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    @staticmethod
    def _session_config(session_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": session_id}}

    def _lock_for_session(self, session_id: str) -> asyncio.Lock:
        if not hasattr(self, "_session_locks"):
            self._session_locks = {}
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    async def _current_messages(self, session_id: str) -> list[BaseMessage]:
        current_state = await self.agent.aget_state(self._session_config(session_id))
        if not current_state or not current_state.values:
            return []
        return list(current_state.values.get("messages", []))

    async def _restore_session_unlocked(self, session_id: str) -> None:
        """首次访问时从最新快照和 JSONL 增量恢复 MemorySaver。"""
        if not hasattr(self, "_restored_sessions"):
            self._restored_sessions = set()
        if session_id in self._restored_sessions:
            return

        current_messages = await self._current_messages(session_id)
        if current_messages:
            self._restored_sessions.add(session_id)
            return

        transcript_store = getattr(self, "transcript_store", None)
        if transcript_store is None:
            self._restored_sessions.add(session_id)
            return

        recovery_messages = transcript_store.load_recovery_messages(session_id)
        if recovery_messages:
            await self.agent.aupdate_state(
                self._session_config(session_id),
                {"messages": recovery_messages},
            )
            logger.info(
                "[会话 {}] 已从本地 Transcript 恢复 {} 条工作上下文消息",
                session_id,
                len(recovery_messages),
            )
        self._restored_sessions.add(session_id)

    @staticmethod
    def _new_state_messages(
        before: Sequence[BaseMessage],
        after: Sequence[BaseMessage],
    ) -> list[BaseMessage]:
        before_ids = {message.id for message in before if message.id is not None}
        return [
            message
            for message in after
            if message.id is None or message.id not in before_ids
        ]

    def _persist_state_messages_unlocked(
        self,
        session_id: str,
        before: Sequence[BaseMessage],
        after: Sequence[BaseMessage],
    ) -> None:
        transcript_store = getattr(self, "transcript_store", None)
        if transcript_store is None:
            return

        new_messages = self._new_state_messages(before, after)
        transcript_sequence = transcript_store.append_messages(
            session_id,
            new_messages,
        )
        transcript_store.write_compaction_snapshot_if_changed(
            session_id,
            after,
            transcript_sequence,
        )

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户
            5. 如果会话中包含 AIOps 历史诊断报告：用户询问“上面、刚才、这次告警”时，优先依据该报告回答；用户询问“现在、当前、是否恢复”时，重新查询实时监控数据，并明确区分历史诊断与当前状态
            6. retrieve_knowledge 返回的资料带有 [来源N] 编号；凡是依据知识库资料得出的结论，都必须在相关句末保留来源编号，并在回答末尾列出实际引用的文件和章节

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明
            - 不得引用没有出现在工具结果中的来源，不得为实时监控数据伪造知识库引用

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    async def record_aiops_report(self, session_id: str, report: str) -> bool:
        """将 AIOps 最终报告交接到普通对话的同一会话上下文。

        AIOps 工作流和普通对话使用不同的 LangGraph 状态结构，因此不能直接
        共享 checkpointer。这里把最终报告转换为一轮隐藏触发消息和助手回复，
        使后续追问能够引用诊断快照，同时不把执行过程写入聊天上下文。

        Returns:
            bool: 本次是否新增了报告；相同报告已存在时返回 False。
        """
        normalized_report = report.strip()
        if not session_id or not normalized_report:
            return False

        await self._initialize_agent()
        async with self._lock_for_session(session_id):
            await self._restore_session_unlocked(session_id)
            config_dict = self._session_config(session_id)
            current_messages = await self._current_messages(session_id)
            transcript_store = getattr(self, "transcript_store", None)
            if transcript_store and transcript_store.contains_aiops_report(
                session_id,
                normalized_report,
            ):
                logger.info(f"[会话 {session_id}] AIOps 报告已存在，跳过重复交接")
                return False
            for message in current_messages:
                metadata = getattr(message, "additional_kwargs", {}) or {}
                if (
                    isinstance(message, AIMessage)
                    and metadata.get("source") == "aiops_report"
                    and message.content == normalized_report
                ):
                    logger.info(f"[会话 {session_id}] AIOps 报告已存在，跳过重复交接")
                    return False

            context_instruction = (
                "请记住以下刚刚完成的 AIOps 诊断报告。它是诊断时刻的历史快照："
                "后续询问‘上面、刚才、这次告警’时应依据报告回答；"
                "询问‘现在、当前、是否恢复’时应重新查询实时监控数据，"
                "不要用当前无告警否定历史上已经发生的告警。"
            )
            handoff_messages = [
                HumanMessage(
                    content=context_instruction,
                    additional_kwargs={
                        "source": "aiops_context_handoff",
                        "hidden_from_history": True,
                    },
                ),
                AIMessage(
                    content=normalized_report,
                    additional_kwargs={
                        "source": "aiops_report",
                        "historical_snapshot": True,
                    },
                ),
            ]
            await self.agent.aupdate_state(
                config_dict,
                {"messages": handoff_messages},
            )
            updated_messages = await self._current_messages(session_id)
            self._persist_state_messages_unlocked(
                session_id,
                current_messages,
                updated_messages,
            )
        logger.info(f"[会话 {session_id}] AIOps 最终报告已交接到普通对话上下文")
        return True

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            async with self._lock_for_session(session_id):
                await self._restore_session_unlocked(session_id)
                previous_messages = await self._current_messages(session_id)
                result = await self.agent.ainvoke(
                    input={"messages": [HumanMessage(content=question)]},
                    config=self._session_config(session_id),
                )

                messages_result = list(result.get("messages", []))
                self._persist_state_messages_unlocked(
                    session_id,
                    previous_messages,
                    messages_result,
                )
                if messages_result:
                    last_message = messages_result[-1]
                    answer = (
                        last_message.content
                        if hasattr(last_message, "content")
                        else str(last_message)
                    )

                    if (
                        hasattr(last_message, "tool_calls")
                        and last_message.tool_calls
                    ):
                        tool_names = [
                            tc.get("name", "unknown")
                            for tc in last_message.tool_calls
                        ]
                        logger.info(
                            f"[会话 {session_id}] Agent 调用了工具: {tool_names}"
                        )

                    logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                    return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(
                f"[会话 {session_id}] RAG Agent 查询失败（非流式）: "
                f"{format_exception_chain(e)}"
            )
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            async with self._lock_for_session(session_id):
                await self._restore_session_unlocked(session_id)
                previous_messages = await self._current_messages(session_id)
                async for token, metadata in self.agent.astream(
                    input={"messages": [HumanMessage(content=question)]},
                    config=self._session_config(session_id),
                    stream_mode="messages",
                ):
                    node_name = (
                        metadata.get("langgraph_node", "unknown")
                        if isinstance(metadata, dict)
                        else "unknown"
                    )
                    message_type = type(token).__name__

                    if message_type in ("AIMessage", "AIMessageChunk"):
                        content_blocks = getattr(token, "content_blocks", None)

                        if content_blocks and isinstance(content_blocks, list):
                            for block in content_blocks:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    text_content = block.get("text", "")
                                    if text_content:
                                        yield {
                                            "type": "content",
                                            "data": text_content,
                                            "node": node_name,
                                        }

                updated_messages = await self._current_messages(session_id)
                self._persist_state_messages_unlocked(
                    session_id,
                    previous_messages,
                    updated_messages,
                )

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            detail = format_exception_chain(e)
            logger.error(
                f"[会话 {session_id}] RAG Agent 查询失败（流式）: {detail}"
            )
            yield {"type": "error", "data": detail}

    def get_session_history(self, session_id: str) -> list:
        """
        获取完整会话历史。优先读取 JSONL Transcript，兼容旧内存会话。

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            transcript_store = getattr(self, "transcript_store", None)
            if (
                transcript_store is not None
                and transcript_store.transcript_path(session_id).exists()
            ):
                history = transcript_store.get_history(session_id)
                logger.info(
                    f"获取本地 Transcript 会话历史: {session_id}, "
                    f"消息数量: {len(history)}"
                )
                return history

            # 使用 checkpointer 的 get 方法获取最新的检查点
            config = {"configurable": {"thread_id": session_id}}

            # 新版 MemorySaver.get() 直接返回 checkpoint 字典；旧版或测试替身
            # 可能返回带 checkpoint 属性的对象或元组，因此需要兼容处理。
            checkpoint_result = self.checkpointer.get(config)

            if not checkpoint_result:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []

            if isinstance(checkpoint_result, dict):
                checkpoint_data = checkpoint_result
            elif hasattr(checkpoint_result, 'checkpoint'):
                checkpoint_data = checkpoint_result.checkpoint  # type: ignore
            else:
                checkpoint_data = checkpoint_result[0] if checkpoint_result else {}

            # 从检查点中提取消息
            messages = checkpoint_data.get("channel_values", {}).get("messages", [])

            # 转换为前端需要的格式
            history = []
            for msg in messages:
                # 跳过系统消息
                if isinstance(msg, SystemMessage):
                    continue

                metadata = getattr(msg, "additional_kwargs", {}) or {}
                if metadata.get("hidden_from_history"):
                    continue

                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)

                # 提取时间戳（如果有的话）
                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    async def clear_session(self, session_id: str) -> bool:
        """
        逻辑清空会话历史。Transcript 追加 clear 事件，不物理删除原始记录。

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        try:
            async with self._lock_for_session(session_id):
                transcript_store = getattr(self, "transcript_store", None)
                if transcript_store is not None:
                    transcript_store.append_clear(session_id)
                self.checkpointer.delete_thread(session_id)
                if hasattr(self, "_restored_sessions"):
                    self._restored_sessions.discard(session_id)

            logger.info(f"已清除会话历史: {session_id}")
            return True

        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)
