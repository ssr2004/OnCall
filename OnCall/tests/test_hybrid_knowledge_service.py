"""混合知识库索引、双路召回与引用元数据测试。"""

import pytest
from langchain_core.documents import Document

from app.services.hybrid_knowledge_service import HybridKnowledgeService
from app.services.rag_query_rewrite_service import RewrittenQuery


class FakeEmbeddings:
    def __init__(self):
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[0.1] * 1024 for _ in texts]

    def embed_query(self, _text):
        return [0.2] * 1024


class FakeHit:
    def __init__(self, chunk_id, content):
        self.entity = {
            "chunk_id": chunk_id,
            "source": "D:/uploads/grid.md",
            "file_name": "grid.md",
            "content_hash": "hash",
            "content": content,
            "metadata": {"h1": "处置手册"},
        }


class FakeCollection:
    def __init__(self):
        self.rows = []
        self.search_calls = []
        self.delete_calls = []

    def upsert(self, rows):
        self.rows = list(rows)

    def flush(self):
        return None

    def delete(self, expr):
        self.delete_calls.append(expr)

    def query(self, **_kwargs):
        return [{"chunk_id": "exists"}]

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        if kwargs["anns_field"] == "dense_vector":
            return [[FakeHit("chunk-a", "通信中断"), FakeHit("chunk-b", "队列积压")]]
        return [[FakeHit("chunk-b", "队列积压"), FakeHit("chunk-c", "同步失败")]]


def test_document_hash_skips_unchanged_embedding(monkeypatch):
    service = HybridKnowledgeService()
    embeddings = FakeEmbeddings()
    documents = [Document(page_content="相同内容", metadata={"h1": "测试"})]
    content_hash = service.calculate_content_hash("相同内容")
    monkeypatch.setattr(
        service,
        "_existing_chunks",
        lambda _source: [{"chunk_id": "old", "content_hash": content_hash}],
    )
    monkeypatch.setattr(service, "_embedding_service", lambda: embeddings)

    result = service.index_documents(documents, "uploads/grid.md", "相同内容")

    assert result["status"] == "unchanged"
    assert result["chunk_count"] == 1
    assert embeddings.document_calls == 0


def test_new_document_uses_stable_ids_and_cleans_only_stale_hash(monkeypatch):
    service = HybridKnowledgeService()
    collection = FakeCollection()
    embeddings = FakeEmbeddings()
    service._collection = collection
    monkeypatch.setattr(service, "_existing_chunks", lambda _source: [])
    monkeypatch.setattr(service, "_embedding_service", lambda: embeddings)
    documents = [
        Document(page_content="第一段", metadata={"h1": "同步失败"}),
        Document(page_content="第二段", metadata={"h2": "处理建议"}),
    ]

    first = service.index_documents(documents, "uploads/grid.md", "完整文档")
    first_ids = [row["chunk_id"] for row in collection.rows]
    second = service.index_documents(documents, "uploads/grid.md", "完整文档")
    second_ids = [row["chunk_id"] for row in collection.rows]

    assert first["status"] == "indexed"
    assert second["status"] == "indexed"
    assert first_ids == second_ids
    assert len(set(first_ids)) == 2
    assert collection.delete_calls
    assert "content_hash !=" in collection.delete_calls[-1]


def test_dense_and_bm25_results_are_rrf_fused(monkeypatch):
    service = HybridKnowledgeService()
    collection = FakeCollection()
    service._collection = collection
    monkeypatch.setattr(service, "_has_documents", lambda: True)
    monkeypatch.setattr(service, "_embedding_service", lambda: FakeEmbeddings())

    results = service._hybrid_search_sync(
        RewrittenQuery(
            semantic_query="电网同步故障",
            keywords=["GridDataSyncFailureRateHigh", "同步失败"],
        )
    )

    assert [item["chunk_id"] for item in results] == ["chunk-b", "chunk-a", "chunk-c"]
    assert results[0]["dense_rank"] == 2
    assert results[0]["bm25_rank"] == 1
    assert collection.search_calls[0]["limit"] == 20
    assert collection.search_calls[1]["limit"] == 20
    assert collection.search_calls[1]["data"] == ["GridDataSyncFailureRateHigh 同步失败"]


@pytest.mark.asyncio
async def test_search_trace_reuses_one_rewrite_and_exposes_ablation_stages(monkeypatch):
    service = HybridKnowledgeService()
    collection = FakeCollection()
    service._collection = collection
    monkeypatch.setattr(service, "_has_documents", lambda: True)
    monkeypatch.setattr(service, "_embedding_service", lambda: FakeEmbeddings())

    rewrite_calls = 0

    async def rewrite(_query):
        nonlocal rewrite_calls
        rewrite_calls += 1
        return RewrittenQuery(semantic_query="电网同步故障", keywords=["同步失败"])

    async def rerank(_query, candidates, final_k):
        return [dict(item, rerank_rank=index) for index, item in enumerate(candidates[:final_k], 1)]

    monkeypatch.setattr(
        "app.services.hybrid_knowledge_service.rag_query_rewrite_service.rewrite",
        rewrite,
    )
    monkeypatch.setattr(
        "app.services.hybrid_knowledge_service.rag_rerank_service.rerank",
        rerank,
    )

    trace = await service.search_with_trace("同步怎么了")

    assert rewrite_calls == 1
    assert [item["chunk_id"] for item in trace.dense_results] == ["chunk-a", "chunk-b"]
    assert [item["chunk_id"] for item in trace.bm25_results] == ["chunk-b", "chunk-c"]
    assert [item["chunk_id"] for item in trace.rrf_results] == [
        "chunk-b",
        "chunk-a",
        "chunk-c",
    ]
    assert trace.results_for("production", 2)[0]["rerank_rank"] == 1
    assert {"rewrite", "embedding", "dense", "bm25", "rrf", "rerank", "total"} <= set(
        trace.timings_ms
    )


def test_ranked_items_keep_source_and_scores_in_documents():
    documents = HybridKnowledgeService.to_documents(
        [
            {
                "chunk_id": "chunk-1",
                "source": "D:/uploads/grid.md",
                "file_name": "grid.md",
                "content": "检查下游接口",
                "metadata": {"h1": "同步失败", "h2": "处理建议"},
                "rrf_score": 0.03,
                "rerank_score": 0.95,
                "rerank_rank": 1,
            }
        ]
    )

    assert documents[0].metadata["source_id"] == "来源1"
    assert documents[0].metadata["chunk_id"] == "chunk-1"
    assert documents[0].metadata["rerank_score"] == 0.95
