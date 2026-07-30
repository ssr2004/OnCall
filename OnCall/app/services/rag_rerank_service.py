"""百炼 Reranker 精排服务。"""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from typing import Any

from dashscope import TextReRank
from loguru import logger

from app.config import config


class RagRerankService:
    """对 RRF 候选进行模型精排，并在不可用时稳定降级。"""

    @staticmethod
    def _content(candidate: dict[str, Any]) -> str:
        content = str(candidate.get("content") or "").strip()
        if content:
            return content[:12000]
        records = candidate.get("records") or []
        return "\n".join(str(item.get("content") or "") for item in records)[:12000]

    @staticmethod
    def _result_value(result: Any, name: str, default: Any) -> Any:
        if isinstance(result, dict):
            return result.get(name, default)
        return getattr(result, name, default)

    @staticmethod
    def _call(query: str, documents: list[str], top_n: int):
        return TextReRank.call(
            model=config.rag_rerank_model,
            query=query,
            documents=documents,
            return_documents=False,
            top_n=top_n,
            api_key=config.dashscope_api_key,
        )

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        final_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        final_k = min(final_k or config.rag_rerank_final_k, len(candidates))
        fallback = []
        for rank, candidate in enumerate(candidates[:final_k], 1):
            item = dict(candidate)
            item.update(
                {
                    "rerank_rank": rank,
                    "rerank_score": None,
                    "rerank_fallback": True,
                }
            )
            fallback.append(item)

        if not config.rag_rerank_enabled:
            return fallback

        documents = [self._content(candidate) for candidate in candidates]
        if not any(documents):
            return fallback
        try:
            response = await asyncio.to_thread(
                self._call,
                query,
                documents,
                final_k,
            )
            if response.status_code != HTTPStatus.OK:
                raise RuntimeError(
                    f"{getattr(response, 'code', '')}: {getattr(response, 'message', '')}"
                )
            results = getattr(getattr(response, "output", None), "results", None) or []
            ranked: list[dict[str, Any]] = []
            for rank, result in enumerate(results, 1):
                index = int(self._result_value(result, "index", -1))
                if index < 0 or index >= len(candidates):
                    continue
                score = float(self._result_value(result, "relevance_score", 0.0))
                item = dict(candidates[index])
                item.update(
                    {
                        "rerank_rank": rank,
                        "rerank_score": score,
                        "rerank_fallback": False,
                    }
                )
                ranked.append(item)
            if not ranked:
                raise RuntimeError("百炼精排未返回有效候选")
            logger.info(
                f"百炼精排完成: candidates={len(candidates)}, final={len(ranked)}"
            )
            return ranked[:final_k]
        except Exception as exc:
            logger.warning(f"百炼精排不可用，降级为 RRF Top {final_k}: {exc}")
            return fallback


rag_rerank_service = RagRerankService()
