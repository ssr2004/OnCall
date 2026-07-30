"""RAG 查询改写服务。"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config


class RewrittenQuery(BaseModel):
    """同时服务 Dense 和 BM25 的结构化检索查询。"""

    semantic_query: str = Field(description="语义完整、可独立理解的检索问题")
    keywords: list[str] = Field(
        default_factory=list,
        description="用于 BM25 的告警名、服务名、技术术语和关键短语",
    )

    @property
    def keyword_query(self) -> str:
        return " ".join(item.strip() for item in self.keywords if item.strip())


class RagQueryRewriteService:
    """使用百炼 Qwen 将自然问题改写为双路检索查询。"""

    def __init__(self) -> None:
        self._chain = None

    def _get_chain(self):
        if self._chain is None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """你是电网运维知识库的查询改写器。请将输入改写成适合检索的结构化查询。

要求：
1. semantic_query 必须补全代词和上下文，形成可以独立理解的自然语言问题。
2. keywords 保留告警名称、服务名、错误码、指标名、日志术语和核心中文短语。
3. 不回答问题，不推断根因，不编造输入中不存在的实体。
4. keywords 去重，最多 12 个。""",
                    ),
                    ("user", "原始查询：\n{query}"),
                ]
            )
            model = ChatQwen(
                model=config.rag_query_rewrite_model,
                api_key=config.dashscope_api_key,
                base_url=config.dashscope_api_base,
                temperature=0,
                max_tokens=config.rag_query_rewrite_max_output_tokens,
                enable_thinking=False,
            )
            logger.info("RAG 查询改写模型: {}", config.rag_query_rewrite_model)
            self._chain = prompt | model.with_structured_output(RewrittenQuery)
        return self._chain

    @staticmethod
    def _fallback(query: str) -> RewrittenQuery:
        technical = re.findall(
            r"Grid[A-Za-z0-9_]+|[A-Za-z][A-Za-z0-9_.:-]{2,}|[\u4e00-\u9fff]{2,12}",
            query,
        )
        keywords: list[str] = []
        seen = set()
        for item in technical:
            normalized = item.strip()
            lowered = normalized.lower()
            if not normalized or lowered in seen:
                continue
            seen.add(lowered)
            keywords.append(normalized)
            if len(keywords) >= 12:
                break
        return RewrittenQuery(
            semantic_query=query,
            keywords=keywords or [query[:500]],
        )

    async def rewrite(self, query: str) -> RewrittenQuery:
        normalized = " ".join(str(query or "").split()).strip()
        if not normalized:
            return RewrittenQuery(semantic_query="", keywords=[])
        # AIOps 原始任务可能携带大段 JSON 和格式模板；改写器只需要前部事实上下文。
        model_input = normalized[:8000]
        try:
            result: Any = await self._get_chain().ainvoke({"query": model_input})
            if not isinstance(result, RewrittenQuery):
                result = RewrittenQuery.model_validate(result)
            semantic_query = " ".join(result.semantic_query.split()).strip()
            keywords = list(dict.fromkeys(item.strip() for item in result.keywords if item.strip()))
            if not semantic_query:
                raise ValueError("查询改写结果缺少 semantic_query")
            return RewrittenQuery(
                semantic_query=semantic_query,
                keywords=keywords[:12] or [semantic_query[:500]],
            )
        except Exception as exc:
            logger.warning(f"RAG 查询改写失败，使用确定性查询降级: {exc}")
            return self._fallback(normalized)


rag_query_rewrite_service = RagQueryRewriteService()
