"""Milvus 处置手册混合检索与幂等索引服务。"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.documents import Document
from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    connections,
    utility,
)

from app.config import config
from app.services.rag_query_rewrite_service import RewrittenQuery, rag_query_rewrite_service
from app.services.rag_rerank_service import rag_rerank_service


@dataclass(slots=True)
class HybridSearchTrace:
    """一次生产检索各阶段的可评测快照。"""

    query: str
    rewritten: RewrittenQuery
    dense_results: list[dict[str, Any]] = field(default_factory=list)
    bm25_results: list[dict[str, Any]] = field(default_factory=list)
    rrf_results: list[dict[str, Any]] = field(default_factory=list)
    reranked_results: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def results_for(self, variant: str, final_k: int | None = None) -> list[dict[str, Any]]:
        """返回指定消融阶段的排名结果。"""
        mapping = {
            "dense": self.dense_results,
            "bm25": self.bm25_results,
            "rrf": self.rrf_results,
            "production": self.reranked_results,
        }
        if variant not in mapping:
            raise ValueError(f"未知检索版本: {variant}")
        results = mapping[variant]
        return results if final_k is None else results[:final_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "rewritten": self.rewritten.model_dump(),
            "dense_results": self.dense_results,
            "bm25_results": self.bm25_results,
            "rrf_results": self.rrf_results,
            "reranked_results": self.reranked_results,
            "timings_ms": self.timings_ms,
            "error": self.error,
        }


class HybridKnowledgeService:
    """管理 `biz_hybrid` Collection 和两路召回、RRF、模型精排。"""

    COLLECTION_NAME = "biz_hybrid"
    VECTOR_DIM = 1024
    CONTENT_MAX_LENGTH = 60000
    OUTPUT_FIELDS = [
        "chunk_id",
        "source",
        "file_name",
        "content_hash",
        "content",
        "metadata",
    ]

    def __init__(self) -> None:
        self._collection: Collection | None = None
        self._collection_lock = threading.RLock()

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _truncate_utf8(text: str, max_bytes: int) -> str:
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    @staticmethod
    def _embedding_service():
        from app.services.vector_embedding_service import vector_embedding_service

        return vector_embedding_service

    def _get_collection(self) -> Collection:
        if self._collection is not None:
            return self._collection
        with self._collection_lock:
            if self._collection is not None:
                return self._collection
            if not connections.has_connection("default"):
                connections.connect(
                    alias="default",
                    host=config.milvus_host,
                    port=str(config.milvus_port),
                    timeout=config.milvus_timeout / 1000,
                )
            if not utility.has_collection(self.COLLECTION_NAME):
                self._create_collection()
            else:
                self._collection = Collection(self.COLLECTION_NAME)
            self._collection.load()
            return self._collection

    def _create_collection(self) -> None:
        fields = [
            FieldSchema("chunk_id", DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema("source", DataType.VARCHAR, max_length=4096),
            FieldSchema("file_name", DataType.VARCHAR, max_length=512),
            FieldSchema("content_hash", DataType.VARCHAR, max_length=64),
            FieldSchema(
                "content",
                DataType.VARCHAR,
                max_length=self.CONTENT_MAX_LENGTH,
                enable_analyzer=True,
                analyzer_params={"tokenizer": "jieba", "filter": ["lowercase"]},
            ),
            FieldSchema("metadata", DataType.JSON),
            FieldSchema("dense_vector", DataType.FLOAT_VECTOR, dim=self.VECTOR_DIM),
            FieldSchema("sparse_vector", DataType.SPARSE_FLOAT_VECTOR),
        ]
        bm25 = Function(
            name="knowledge_content_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
        )
        self._collection = Collection(
            name=self.COLLECTION_NAME,
            schema=CollectionSchema(
                fields=fields,
                functions=[bm25],
                description="Hybrid business knowledge base with dense and BM25 retrieval",
                enable_dynamic_field=False,
            ),
            num_shards=2,
        )
        self._collection.create_index(
            field_name="dense_vector",
            index_params={
                "index_type": "AUTOINDEX",
                "metric_type": "COSINE",
                "params": {},
            },
        )
        self._collection.create_index(
            field_name="sparse_vector",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
            },
        )
        logger.info(f"已创建混合知识库 Collection: {self.COLLECTION_NAME}")

    @staticmethod
    def calculate_content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _chunk_id(source: str, content_hash: str, index: int) -> str:
        value = f"{source}|{content_hash}|{index}"
        return f"kb-{hashlib.sha256(value.encode('utf-8')).hexdigest()}"

    def _existing_chunks(self, source: str) -> list[dict[str, Any]]:
        escaped = self._escape(source)
        return self._get_collection().query(
            expr=f'source == "{escaped}"',
            output_fields=["chunk_id", "content_hash"],
            limit=16384,
        )

    def index_documents(
        self,
        documents: list[Document],
        source: str,
        original_content: str,
    ) -> dict[str, Any]:
        """按内容哈希幂等写入文档；相同内容不会重复 Embedding。"""
        normalized_source = Path(source).resolve().as_posix()
        content_hash = self.calculate_content_hash(original_content)
        existing = self._existing_chunks(normalized_source)
        if (
            existing
            and len(existing) == len(documents)
            and all(item.get("content_hash") == content_hash for item in existing)
        ):
            logger.info(f"文档内容未变化，跳过重新向量化: {normalized_source}")
            return {
                "status": "unchanged",
                "source": normalized_source,
                "content_hash": content_hash,
                "chunk_count": len(existing),
            }

        contents = [
            self._truncate_utf8(document.page_content, self.CONTENT_MAX_LENGTH - 256)
            for document in documents
        ]
        if not contents:
            return {
                "status": "empty",
                "source": normalized_source,
                "content_hash": content_hash,
                "chunk_count": 0,
            }
        vectors = self._embedding_service().embed_documents(contents)
        if len(vectors) != len(contents) or any(
            len(vector) != self.VECTOR_DIM for vector in vectors
        ):
            raise RuntimeError("知识库 Embedding 返回的向量数量或维度不正确")

        file_name = Path(normalized_source).name
        rows = []
        for index, (document, content, vector) in enumerate(
            zip(documents, contents, vectors, strict=True)
        ):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "_source": normalized_source,
                    "_file_name": file_name,
                    "_content_hash": content_hash,
                    "_chunk_index": index,
                    "_chunk_count": len(documents),
                }
            )
            rows.append(
                {
                    "chunk_id": self._chunk_id(normalized_source, content_hash, index),
                    "source": normalized_source,
                    "file_name": file_name,
                    "content_hash": content_hash,
                    "content": content,
                    "metadata": metadata,
                    "dense_vector": vector,
                }
            )

        collection = self._get_collection()
        collection.upsert(rows)
        collection.flush()

        # 新版本已成功落库后才清理同一来源的旧哈希，避免更新失败造成知识空窗。
        escaped_source = self._escape(normalized_source)
        escaped_hash = self._escape(content_hash)
        collection.delete(
            expr=(f'source == "{escaped_source}" and content_hash != "{escaped_hash}"')
        )
        collection.flush()
        status = "updated" if existing else "indexed"
        logger.info(f"混合知识库文档{status}: {normalized_source}, chunks={len(rows)}")
        return {
            "status": status,
            "source": normalized_source,
            "content_hash": content_hash,
            "chunk_count": len(rows),
        }

    @staticmethod
    def _entity_value(entity: Any, name: str, default: Any = None) -> Any:
        if isinstance(entity, dict):
            return entity.get(name, default)
        getter = getattr(entity, "get", None)
        if getter:
            try:
                value = getter(name)
            except TypeError:
                value = getter(name, default)
            return default if value is None else value
        return getattr(entity, name, default)

    @classmethod
    def _hit_to_dict(cls, hit: Any) -> dict[str, Any]:
        entity = getattr(hit, "entity", None)
        if entity is None and isinstance(hit, dict):
            entity = hit.get("entity", hit)
        return {field: cls._entity_value(entity, field) for field in cls.OUTPUT_FIELDS}

    def _has_documents(self) -> bool:
        rows = self._get_collection().query(
            expr='chunk_id != ""',
            output_fields=["chunk_id"],
            limit=1,
        )
        return bool(rows)

    def _hybrid_search_sync(
        self,
        rewritten: RewrittenQuery,
    ) -> list[dict[str, Any]]:
        """兼容旧调用：返回 RRF 候选列表。"""
        _, _, rrf_results, _ = self._hybrid_recall_sync(rewritten)
        return rrf_results

    @staticmethod
    def _hit_score(hit: Any) -> float | None:
        for name in ("score", "distance"):
            value = getattr(hit, name, None)
            if value is None and isinstance(hit, dict):
                value = hit.get(name)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    def _ranked_channel_results(
        self,
        hits: list[Any],
        channel: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for rank, hit in enumerate(hits, 1):
            item = self._hit_to_dict(hit)
            if not item.get("chunk_id"):
                continue
            item.update(
                {
                    "dense_rank": rank if channel == "dense" else None,
                    "bm25_rank": rank if channel == "bm25" else None,
                    f"{channel}_score": self._hit_score(hit),
                }
            )
            results.append(item)
        return results

    def _hybrid_recall_sync(
        self,
        rewritten: RewrittenQuery,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, float],
    ]:
        """执行一次双路召回并返回各阶段排名及耗时。"""
        collection = self._get_collection()
        if not self._has_documents():
            return [], [], [], {"embedding": 0.0, "dense": 0.0, "bm25": 0.0, "rrf": 0.0}

        started = perf_counter()
        query_vector = self._embedding_service().embed_query(rewritten.semantic_query)
        embedding_ms = (perf_counter() - started) * 1000
        if len(query_vector) != self.VECTOR_DIM:
            raise RuntimeError("知识库查询向量维度不正确")

        started = perf_counter()
        dense_hits = collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=config.rag_dense_recall_k,
            output_fields=self.OUTPUT_FIELDS,
        )[0]
        dense_ms = (perf_counter() - started) * 1000

        started = perf_counter()
        sparse_hits = collection.search(
            data=[rewritten.keyword_query or rewritten.semantic_query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25", "params": {}},
            limit=config.rag_bm25_recall_k,
            output_fields=self.OUTPUT_FIELDS,
        )[0]
        bm25_ms = (perf_counter() - started) * 1000

        dense_results = self._ranked_channel_results(dense_hits, "dense")
        bm25_results = self._ranked_channel_results(sparse_hits, "bm25")

        started = perf_counter()
        merged: dict[str, dict[str, Any]] = {}
        for channel, results in (("dense", dense_results), ("bm25", bm25_results)):
            for rank, item in enumerate(results, 1):
                chunk_id = str(item.get("chunk_id") or "")
                if not chunk_id:
                    continue
                record = merged.setdefault(
                    chunk_id,
                    {
                        **item,
                        "rrf_score": 0.0,
                        "dense_rank": None,
                        "bm25_rank": None,
                    },
                )
                record["rrf_score"] += 1.0 / (config.rag_rrf_rank_constant + rank)
                record[f"{channel}_rank"] = rank
                score_name = f"{channel}_score"
                if item.get(score_name) is not None:
                    record[score_name] = item[score_name]
        rrf_results = sorted(
            merged.values(),
            key=lambda item: (-item["rrf_score"], item["chunk_id"]),
        )[: config.rag_rrf_candidate_k]
        rrf_ms = (perf_counter() - started) * 1000
        return (
            dense_results,
            bm25_results,
            rrf_results,
            {
                "embedding": embedding_ms,
                "dense": dense_ms,
                "bm25": bm25_ms,
                "rrf": rrf_ms,
            },
        )

    async def search_with_trace(self, query: str) -> HybridSearchTrace:
        """执行真实生产检索，同时保留可用于评测和消融的阶段 Trace。"""
        total_started = perf_counter()
        started = perf_counter()
        rewritten = await rag_query_rewrite_service.rewrite(query)
        rewrite_ms = (perf_counter() - started) * 1000
        trace = HybridSearchTrace(
            query=query,
            rewritten=rewritten,
            timings_ms={"rewrite": rewrite_ms},
        )
        if not rewritten.semantic_query:
            trace.timings_ms["total"] = (perf_counter() - total_started) * 1000
            return trace
        try:
            dense, bm25, rrf, recall_timings = await asyncio.to_thread(
                self._hybrid_recall_sync,
                rewritten,
            )
            trace.dense_results = dense
            trace.bm25_results = bm25
            trace.rrf_results = rrf
            trace.timings_ms.update(recall_timings)

            started = perf_counter()
            trace.reranked_results = await rag_rerank_service.rerank(
                rewritten.semantic_query,
                rrf,
                final_k=config.rag_rerank_final_k,
            )
            trace.timings_ms["rerank"] = (perf_counter() - started) * 1000
        except Exception as exc:
            trace.error = str(exc)
            logger.error(f"混合知识库召回失败: {exc}", exc_info=True)
        trace.timings_ms["total"] = (perf_counter() - total_started) * 1000
        return trace

    async def search(self, query: str) -> tuple[RewrittenQuery, list[dict[str, Any]]]:
        trace = await self.search_with_trace(query)
        return trace.rewritten, trace.reranked_results

    @staticmethod
    def to_documents(items: list[dict[str, Any]]) -> list[Document]:
        documents = []
        for index, item in enumerate(items, 1):
            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "source_id": f"来源{index}",
                    "chunk_id": item.get("chunk_id"),
                    "file_name": item.get("file_name"),
                    "source": item.get("source"),
                    "dense_rank": item.get("dense_rank"),
                    "bm25_rank": item.get("bm25_rank"),
                    "rrf_score": item.get("rrf_score"),
                    "rerank_rank": item.get("rerank_rank"),
                    "rerank_score": item.get("rerank_score"),
                    "rerank_fallback": item.get("rerank_fallback", False),
                }
            )
            documents.append(
                Document(
                    page_content=str(item.get("content") or ""),
                    metadata=metadata,
                )
            )
        return documents


hybrid_knowledge_service = HybridKnowledgeService()
