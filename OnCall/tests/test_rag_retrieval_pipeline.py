"""查询改写、百炼精排降级和来源格式测试。"""

from http import HTTPStatus
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.services.rag_query_rewrite_service import RagQueryRewriteService, RewrittenQuery
from app.services.rag_rerank_service import RagRerankService
from app.tools.knowledge_tool import format_docs


@pytest.mark.asyncio
async def test_query_rewrite_falls_back_without_blocking(monkeypatch):
    service = RagQueryRewriteService()
    chain = SimpleNamespace(
        ainvoke=lambda _input: None,
    )

    async def fail(_input):
        raise RuntimeError("model unavailable")

    chain.ainvoke = fail
    monkeypatch.setattr(service, "_get_chain", lambda: chain)

    result = await service.rewrite(
        "GridDataSyncFailureRateHigh 数据同步失败应该怎么处理"
    )

    assert isinstance(result, RewrittenQuery)
    assert result.semantic_query.startswith("GridDataSyncFailureRateHigh")
    assert "GridDataSyncFailureRateHigh" in result.keywords


@pytest.mark.asyncio
async def test_bailian_reranker_selects_model_ranked_top3(monkeypatch):
    service = RagRerankService()
    response = SimpleNamespace(
        status_code=HTTPStatus.OK,
        output=SimpleNamespace(
            results=[
                {"index": 3, "relevance_score": 0.99},
                {"index": 1, "relevance_score": 0.88},
                {"index": 0, "relevance_score": 0.77},
            ]
        ),
    )
    monkeypatch.setattr(service, "_call", lambda *_args: response)
    candidates = [
        {"chunk_id": f"chunk-{index}", "content": f"document {index}"}
        for index in range(5)
    ]

    ranked = await service.rerank("query", candidates, final_k=3)

    assert [item["chunk_id"] for item in ranked] == [
        "chunk-3",
        "chunk-1",
        "chunk-0",
    ]
    assert ranked[0]["rerank_score"] == 0.99
    assert ranked[0]["rerank_fallback"] is False


@pytest.mark.asyncio
async def test_reranker_failure_falls_back_to_rrf_order(monkeypatch):
    service = RagRerankService()

    def fail(*_args):
        raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(service, "_call", fail)
    candidates = [
        {"chunk_id": f"chunk-{index}", "content": f"document {index}"}
        for index in range(5)
    ]

    ranked = await service.rerank("query", candidates, final_k=3)

    assert [item["chunk_id"] for item in ranked] == ["chunk-0", "chunk-1", "chunk-2"]
    assert all(item["rerank_fallback"] is True for item in ranked)


def test_formatted_context_contains_verifiable_sources():
    context = format_docs(
        [
            Document(
                page_content="检查下游同步接口。",
                metadata={
                    "source_id": "来源1",
                    "_file_name": "grid_sync_failure.md",
                    "h1": "数据同步失败",
                    "h2": "处理建议",
                    "chunk_id": "chunk-abc",
                    "rerank_score": 0.91234567,
                },
            )
        ]
    )

    assert "[来源1]" in context
    assert "grid_sync_failure.md" in context
    assert "数据同步失败 > 处理建议" in context
    assert "chunk-abc" in context
    assert "0.912346" in context
