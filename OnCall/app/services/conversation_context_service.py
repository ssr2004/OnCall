"""对话 Agent 的 Token 预算、旧工具输出裁剪和上下文压缩。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime

TokenCounter = Callable[[Iterable[Any]], int]


def count_qwen_tokens_conservatively(messages: Iterable[Any]) -> int:
    """保守估算中英文混合 Qwen 消息 Token，避免按英文 chars/4 低估中文。"""
    return count_tokens_approximately(messages, chars_per_token=2.0)


def estimate_fixed_prompt_tokens(
    system_prompt: str,
    tools: Sequence[Any],
) -> int:
    """估算每次模型调用固定携带的 System Prompt 与工具 Schema 开销。"""
    return count_tokens_approximately(
        [SystemMessage(content=system_prompt)],
        chars_per_token=2.0,
        tools=list(tools),
    )


@dataclass(frozen=True)
class ConversationContextBudget:
    """由模型真实规格和运行时固定开销推导出的对话工作窗口预算。"""

    hard_input_tokens: int
    effective_input_tokens: int
    fixed_prompt_tokens: int
    message_trigger_tokens: int
    keep_recent_tokens: int


def build_conversation_context_budget(
    *,
    context_window_tokens: int,
    max_input_tokens: int,
    max_output_tokens: int,
    operating_input_tokens: int,
    fixed_prompt_tokens: int,
) -> ConversationContextBudget:
    """计算可供消息历史使用的预算，不再依赖固定 18K/24K 阈值。"""
    if min(context_window_tokens, max_input_tokens, max_output_tokens) <= 0:
        raise ValueError("模型上下文、最大输入和最大输出必须为正整数")
    if max_output_tokens >= context_window_tokens:
        raise ValueError("模型最大输出必须小于上下文窗口")

    hard_input_tokens = min(
        max_input_tokens,
        context_window_tokens - max_output_tokens,
    )
    effective_input_tokens = hard_input_tokens
    if operating_input_tokens > 0:
        effective_input_tokens = min(
            effective_input_tokens,
            operating_input_tokens,
        )

    message_trigger_tokens = effective_input_tokens - fixed_prompt_tokens
    if message_trigger_tokens <= 0:
        raise ValueError("System Prompt 与工具 Schema 已超过模型输入预算")

    # 保留约两个最大回答长度的最近完整对话，兼顾连续追问和压缩收益。
    keep_recent_tokens = min(
        max_output_tokens * 2,
        message_trigger_tokens,
    )
    return ConversationContextBudget(
        hard_input_tokens=hard_input_tokens,
        effective_input_tokens=effective_input_tokens,
        fixed_prompt_tokens=fixed_prompt_tokens,
        message_trigger_tokens=message_trigger_tokens,
        keep_recent_tokens=keep_recent_tokens,
    )

CONVERSATION_SUMMARY_PROMPT = """
<role>
你是智能运维对话的上下文压缩器。
</role>

<objective>
将即将退出模型工作窗口的历史消息压缩为可继续对话的可靠交接摘要。
摘要会替代原始消息，因此必须保留后续回答所需的事实，不能补充或猜测信息。
</objective>

<requirements>
请严格使用以下章节；没有内容时写“无”：

## 会话目标
用户当前希望解决的问题，以及本次会话的主要目标。

## 已确认事实
用户明确提供或确认的事实、精确服务名、告警名、时间、数值和状态。

## AIOps 诊断上下文
保留历史诊断报告中的告警、根因、关键证据、处置建议、风险等级、事件标识和引用来源。
明确区分“诊断时刻的历史状态”和“当前实时状态”，不得用当前状态覆盖历史故障。

## 用户反馈与约束
保留用户对诊断的确认、否定、纠正、偏好和尚未改变的要求。

## 已完成事项
已经执行的查询、工具调用和得到的有效结论。工具原始输出只保留关键证据与来源标识。

## 未解决问题与下一步
仍需回答的问题、待验证假设和建议的下一步动作。

## 可追溯引用
保留所有仍被结论引用的知识来源编号、事件 ID、证据 ID 或报告标识。
</requirements>

<messages>
{messages}
</messages>

只输出压缩摘要，不要添加开场白或解释。
""".strip()


class OldToolOutputPruningMiddleware(AgentMiddleware):
    """在整体压缩前先缩短较早且过长的工具输出。"""

    def __init__(
        self,
        *,
        trigger_tokens: int,
        keep_recent_messages: int | None = None,
        keep_recent_tokens: int | None = None,
        max_tool_output_chars: int,
        token_counter: TokenCounter = count_qwen_tokens_conservatively,
    ) -> None:
        super().__init__()
        if keep_recent_messages is None and keep_recent_tokens is None:
            raise ValueError("必须配置最近消息数量或最近消息 Token 预算")
        self.trigger_tokens = trigger_tokens
        self.keep_recent_messages = keep_recent_messages
        self.keep_recent_tokens = keep_recent_tokens
        self.max_tool_output_chars = max_tool_output_chars
        self.token_counter = token_counter

    def _recent_cutoff(self, messages: Sequence[BaseMessage]) -> int:
        if self.keep_recent_tokens is None:
            return max(0, len(messages) - int(self.keep_recent_messages or 0))

        cutoff = len(messages)
        recent_tokens = 0
        for index in range(len(messages) - 1, -1, -1):
            message_tokens = self.token_counter([messages[index]])
            if cutoff < len(messages) and (
                recent_tokens + message_tokens > self.keep_recent_tokens
            ):
                break
            recent_tokens += message_tokens
            cutoff = index

        # Token 边界落在一轮对话中间时，回退到该轮 HumanMessage。
        while cutoff > 0 and not isinstance(messages[cutoff], HumanMessage):
            cutoff -= 1
        return cutoff

    def _prune(self, state: dict[str, Any]) -> dict[str, Any] | None:
        messages: Sequence[BaseMessage] = state.get("messages", [])
        if self.token_counter(messages) < self.trigger_tokens:
            return None

        cutoff = self._recent_cutoff(messages)
        replacements: list[ToolMessage] = []
        for message in messages[:cutoff]:
            if not isinstance(message, ToolMessage):
                continue
            content = message.content
            rendered = content if isinstance(content, str) else str(content)
            if len(rendered) <= self.max_tool_output_chars:
                continue
            metadata = dict(message.additional_kwargs or {})
            metadata["context_pruned"] = True
            replacement = message.model_copy(
                update={
                    "content": (
                        "[较早的工具原始输出已从模型工作上下文裁剪；"
                        "完整内容仍保存在本地会话 Transcript 中。]"
                    ),
                    "additional_kwargs": metadata,
                }
            )
            replacements.append(replacement)

        if not replacements:
            return None
        return {"messages": replacements}

    def before_model(
        self,
        state: dict[str, Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        del runtime
        return self._prune(state)

    async def abefore_model(
        self,
        state: dict[str, Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        del runtime
        return self._prune(state)


class SafeSummarizationMiddleware(SummarizationMiddleware):
    """摘要模型异常时保持原上下文，避免错误摘要替换完整历史。"""

    @staticmethod
    def _find_safe_cutoff_point(
        messages: list[BaseMessage],
        cutoff_index: int,
    ) -> int:
        """在框架工具调用安全边界基础上，再保持完整用户对话轮次。"""
        safe_cutoff = SummarizationMiddleware._find_safe_cutoff_point(
            messages,
            cutoff_index,
        )
        while safe_cutoff > 0 and not isinstance(
            messages[safe_cutoff],
            HumanMessage,
        ):
            safe_cutoff -= 1
        return safe_cutoff

    @staticmethod
    def _summary_failed(result: dict[str, Any] | None) -> bool:
        if not result:
            return False
        for message in result.get("messages", []):
            content = getattr(message, "content", "")
            if isinstance(content, str) and "Error generating summary:" in content:
                return True
        return False

    def before_model(
        self,
        state: dict[str, Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        result = super().before_model(state, runtime)
        return None if self._summary_failed(result) else result

    async def abefore_model(
        self,
        state: dict[str, Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        result = await super().abefore_model(state, runtime)
        return None if self._summary_failed(result) else result
