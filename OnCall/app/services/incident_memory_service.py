"""基于 Milvus 的故障情景记忆服务。

候选事件只保存在当前进程中；只有人工确认后的事件才会写入 Milvus。
检索时分别执行 Dense 语义召回和 BM25 关键词召回，再按 incident_id
去重并使用 RRF 融合，避免历史案例覆盖本次实时监控证据。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

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
from app.models.incident import PendingIncident
from app.services.rag_query_rewrite_service import rag_query_rewrite_service
from app.services.rag_rerank_service import rag_rerank_service


class IncidentMemoryError(RuntimeError):
    """情景记忆操作失败。"""


class IncidentNotFoundError(IncidentMemoryError):
    """候选事件不存在或已经过期。"""


class IncidentDecisionConflictError(IncidentMemoryError):
    """候选事件的人工决策与当前状态冲突。"""


class IncidentMemoryService:
    """管理候选事件、Milvus 持久化和混合检索。"""

    COLLECTION_NAME = "incident_memory"
    VECTOR_DIM = 1024
    CONTENT_MAX_LENGTH = 60000
    OUTPUT_FIELDS = [
        "memory_id",
        "incident_id",
        "alert_fingerprint",
        "memory_type",
        "service_name",
        "alert_name",
        "severity",
        "status",
        "verification_status",
        "started_at",
        "confirmed_at",
        "content",
        "metadata",
    ]

    def __init__(self) -> None:
        self._collection: Collection | None = None
        self._collection_lock = threading.RLock()
        self._pending_lock = threading.RLock()
        self._pending: OrderedDict[str, PendingIncident] = OrderedDict()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _parse_timestamp_ms(value: Any) -> int:
        if isinstance(value, (int, float)):
            number = int(value)
            return number if number > 10_000_000_000 else number * 1000
        text = str(value or "").strip()
        if not text:
            return 0
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1000)
        except ValueError:
            return 0

    @staticmethod
    def _safe_text(value: Any, default: str = "unknown") -> str:
        text = str(value or "").strip()
        return text or default

    @classmethod
    def _alert_identity(cls, alert: dict[str, Any]) -> str:
        labels = alert.get("labels") if isinstance(alert.get("labels"), dict) else {}
        alert_name = cls._safe_text(
            alert.get("alert_name") or alert.get("alertname") or labels.get("alertname")
        )
        service_name = cls._safe_text(
            alert.get("service_name") or labels.get("service") or labels.get("job")
        )
        instance = cls._safe_text(alert.get("instance") or labels.get("instance"), "")
        started_at = cls._safe_text(
            alert.get("active_at") or alert.get("started_at") or alert.get("startsAt"),
            "",
        )
        return "|".join((alert_name, service_name, instance, started_at))

    @classmethod
    def build_alert_fingerprint(cls, active_alerts: dict[str, Any]) -> str:
        alerts = [
            alert
            for alert in active_alerts.get("alerts", [])
            if isinstance(alert, dict)
        ]
        identities = sorted(cls._alert_identity(alert) for alert in alerts)
        payload = "\n".join(identities)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cleanup_pending_locked(self) -> None:
        cutoff = self._now_ms() - config.incident_pending_ttl_seconds * 1000
        expired = [
            incident_id
            for incident_id, incident in self._pending.items()
            if incident.created_at < cutoff
        ]
        for incident_id in expired:
            self._pending.pop(incident_id, None)
        while len(self._pending) > config.incident_pending_max_items:
            self._pending.popitem(last=False)

    def create_candidate(
        self,
        session_id: str,
        active_alerts: dict[str, Any],
        report: str,
        diagnosis_mode: str = "llm",
        similar_incidents: list[dict[str, Any]] | None = None,
    ) -> PendingIncident | None:
        """创建待人工确认事件；无活动告警时不创建。"""
        alerts = [
            alert
            for alert in active_alerts.get("alerts", [])
            if isinstance(alert, dict)
        ]
        if not alerts or not report.strip():
            return None

        fingerprint = self.build_alert_fingerprint(active_alerts)
        incident_id = f"inc-{fingerprint[:32]}"
        service_names = sorted(
            {
                self._safe_text(alert.get("service_name"), "grid-data-sync-service")
                for alert in alerts
            }
        )
        alert_names = sorted(
            {self._safe_text(alert.get("alert_name") or alert.get("alertname")) for alert in alerts}
        )
        severities = sorted(
            {self._safe_text(alert.get("severity")) for alert in alerts}
        )
        started_values = [
            self._parse_timestamp_ms(
                alert.get("active_at") or alert.get("started_at") or alert.get("startsAt")
            )
            for alert in alerts
        ]
        started_values = [value for value in started_values if value > 0]
        candidate = PendingIncident(
            incident_id=incident_id,
            session_id=session_id,
            alert_fingerprint=fingerprint,
            active_alerts=active_alerts,
            service_name=", ".join(service_names),
            alert_name=", ".join(alert_names),
            severity=", ".join(severities),
            started_at=min(started_values) if started_values else self._now_ms(),
            report=report.strip(),
            diagnosis_mode=diagnosis_mode,
            similar_incidents=similar_incidents or [],
            status="pending",
            created_at=self._now_ms(),
        )
        with self._pending_lock:
            self._cleanup_pending_locked()
            # 重新运行 AIOps 会生成一份新的待确认候选，因此被拒绝的报告不会锁死。
            self._pending[incident_id] = candidate
            self._pending.move_to_end(incident_id)
            self._cleanup_pending_locked()
        logger.info(f"已创建待确认故障事件: {incident_id}")
        return candidate

    def get_candidate(self, incident_id: str) -> PendingIncident:
        with self._pending_lock:
            self._cleanup_pending_locked()
            candidate = self._pending.get(incident_id)
            if candidate is None:
                raise IncidentNotFoundError("候选故障事件不存在或已过期，请重新运行 AIOps")
            return candidate.model_copy(deep=True)

    async def confirm_incident(self, incident_id: str) -> dict[str, Any]:
        """幂等确认候选事件，并将其摘要及报告写入 Milvus。"""
        with self._pending_lock:
            self._cleanup_pending_locked()
            candidate = self._pending.get(incident_id)
            if candidate is not None and candidate.status == "rejected":
                raise IncidentDecisionConflictError(
                    "该报告已标记为诊断不准确；如需确认，请重新运行 AIOps"
                )
            if candidate is not None and candidate.status == "confirmed":
                return {
                    "incident_id": incident_id,
                    "status": "confirmed",
                    "persisted": True,
                    "message": "该诊断已经确认，无需重复写入",
                }

        if candidate is None:
            already_persisted = await asyncio.to_thread(self._is_persisted, incident_id)
            if already_persisted:
                return {
                    "incident_id": incident_id,
                    "status": "confirmed",
                    "persisted": True,
                    "message": "该诊断已经确认，无需重复写入",
                }
            raise IncidentNotFoundError("候选故障事件不存在或已过期，请重新运行 AIOps")

        confirmed_at = self._now_ms()
        await asyncio.to_thread(self._persist_candidate, candidate, confirmed_at)
        with self._pending_lock:
            current = self._pending.get(incident_id)
            if current is not None:
                current.status = "confirmed"
                current.confirmed_at = confirmed_at
        return {
            "incident_id": incident_id,
            "status": "confirmed",
            "persisted": True,
            "message": "诊断已确认，并已写入长期情景记忆",
        }

    async def reject_incident(self, incident_id: str) -> dict[str, Any]:
        """拒绝候选事件，不写入 Milvus。"""
        with self._pending_lock:
            self._cleanup_pending_locked()
            candidate = self._pending.get(incident_id)
            if candidate is None:
                raise IncidentNotFoundError("候选故障事件不存在或已过期，请重新运行 AIOps")
            if candidate.status == "confirmed":
                raise IncidentDecisionConflictError("该诊断已确认并写入情景记忆，不能再标记为不准确")
            if candidate.status == "rejected":
                return {
                    "incident_id": incident_id,
                    "status": "rejected",
                    "persisted": False,
                    "message": "该诊断已标记为不准确，不会写入情景记忆",
                }
            candidate.status = "rejected"
        logger.info(f"故障事件被人工拒绝，不写入情景记忆: {incident_id}")
        return {
            "incident_id": incident_id,
            "status": "rejected",
            "persisted": False,
            "message": "已标记为诊断不准确，本次报告不会写入情景记忆",
        }

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
            FieldSchema("memory_id", DataType.VARCHAR, max_length=160, is_primary=True),
            FieldSchema("incident_id", DataType.VARCHAR, max_length=80),
            FieldSchema("alert_fingerprint", DataType.VARCHAR, max_length=80),
            FieldSchema("memory_type", DataType.VARCHAR, max_length=40),
            FieldSchema("service_name", DataType.VARCHAR, max_length=500),
            FieldSchema("alert_name", DataType.VARCHAR, max_length=1000),
            FieldSchema("severity", DataType.VARCHAR, max_length=100),
            FieldSchema("status", DataType.VARCHAR, max_length=40),
            FieldSchema("verification_status", DataType.VARCHAR, max_length=40),
            FieldSchema("started_at", DataType.INT64),
            FieldSchema("confirmed_at", DataType.INT64),
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
            name="incident_content_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
        )
        schema = CollectionSchema(
            fields=fields,
            functions=[bm25],
            description="Confirmed AIOps incident episodic memory",
            enable_dynamic_field=False,
        )
        self._collection = Collection(
            name=self.COLLECTION_NAME,
            schema=schema,
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
        logger.info(f"已创建故障情景记忆 Collection: {self.COLLECTION_NAME}")

    @staticmethod
    def _truncate_utf8(text: str, max_bytes: int) -> str:
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    @staticmethod
    def _embedding_service():
        # 延迟导入，避免仅使用候选事件接口时提前初始化外部 Embedding 客户端。
        from app.services.vector_embedding_service import vector_embedding_service

        return vector_embedding_service

    def _build_summary(self, candidate: PendingIncident) -> str:
        alerts = candidate.active_alerts.get("alerts", [])
        descriptions = [
            self._safe_text(alert.get("description") or alert.get("summary"), "")
            for alert in alerts
            if isinstance(alert, dict)
        ]
        report_excerpt = self._truncate_utf8(candidate.report, 12000)
        return (
            f"故障事件 {candidate.incident_id}\n"
            f"告警名称：{candidate.alert_name}\n"
            f"受影响服务：{candidate.service_name}\n"
            f"告警级别：{candidate.severity}\n"
            f"告警描述：{'；'.join(item for item in descriptions if item)}\n"
            f"诊断模式：{candidate.diagnosis_mode}\n"
            f"已确认诊断报告摘要：\n{report_excerpt}"
        )

    def _persist_candidate(self, candidate: PendingIncident, confirmed_at: int) -> None:
        collection = self._get_collection()
        summary = self._build_summary(candidate)
        report = self._truncate_utf8(candidate.report, self.CONTENT_MAX_LENGTH - 256)
        contents = [summary, report]
        vectors = self._embedding_service().embed_documents(contents)
        if len(vectors) != 2 or any(len(vector) != self.VECTOR_DIM for vector in vectors):
            raise IncidentMemoryError("Embedding 返回的向量数量或维度不正确")

        common = {
            "incident_id": candidate.incident_id,
            "alert_fingerprint": candidate.alert_fingerprint,
            "service_name": candidate.service_name,
            "alert_name": candidate.alert_name,
            "severity": candidate.severity,
            "status": "closed",
            "verification_status": "confirmed",
            "started_at": candidate.started_at,
            "confirmed_at": confirmed_at,
        }
        metadata = {
            "session_id": candidate.session_id,
            "diagnosis_mode": candidate.diagnosis_mode,
            "active_alerts": candidate.active_alerts,
            "similar_incident_ids": [
                item.get("incident_id") for item in candidate.similar_incidents[:3]
            ],
        }
        rows = [
            {
                "memory_id": f"{candidate.incident_id}:incident_summary",
                **common,
                "memory_type": "incident_summary",
                "content": self._truncate_utf8(summary, self.CONTENT_MAX_LENGTH - 256),
                "metadata": metadata,
                "dense_vector": vectors[0],
            },
            {
                "memory_id": f"{candidate.incident_id}:aiops_report",
                **common,
                "memory_type": "aiops_report",
                "content": report,
                "metadata": metadata,
                "dense_vector": vectors[1],
            },
        ]
        collection.upsert(rows)
        collection.flush()
        logger.info(f"故障事件已持久化到 Milvus: {candidate.incident_id}")

    @staticmethod
    def _escape_milvus_string(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _base_filter(self, current_incident_id: str | None = None) -> str:
        expression = (
            'memory_type == "incident_summary" and status == "closed" '
            'and verification_status == "confirmed"'
        )
        if current_incident_id:
            escaped = self._escape_milvus_string(current_incident_id)
            expression += f' and incident_id != "{escaped}"'
        return expression

    def _is_persisted(self, incident_id: str) -> bool:
        collection = self._get_collection()
        escaped = self._escape_milvus_string(incident_id)
        rows = collection.query(
            expr=(
                f'incident_id == "{escaped}" and memory_type == "incident_summary" '
                'and verification_status == "confirmed"'
            ),
            output_fields=["memory_id"],
            limit=1,
        )
        return bool(rows)

    def _has_searchable_memories(self, current_incident_id: str | None) -> bool:
        rows = self._get_collection().query(
            expr=self._base_filter(current_incident_id),
            output_fields=["memory_id"],
            limit=1,
        )
        return bool(rows)

    @staticmethod
    def _hit_to_dict(hit: Any) -> dict[str, Any]:
        entity = getattr(hit, "entity", None)
        if entity is None and isinstance(hit, dict):
            entity = hit.get("entity", hit)
        def value(name: str, default: Any = None) -> Any:
            if entity is None:
                return default
            if isinstance(entity, dict):
                return entity.get(name, default)
            getter = getattr(entity, "get", None)
            if getter:
                try:
                    result = getter(name)
                except TypeError:
                    result = getter(name, default)
                return default if result is None else result
            return getattr(entity, name, default)

        return {
            field: value(field)
            for field in IncidentMemoryService.OUTPUT_FIELDS
        }

    def _search_sync(
        self,
        query: str,
        current_incident_id: str | None,
        candidate_k: int,
        keyword_query: str | None = None,
    ) -> list[dict[str, Any]]:
        collection = self._get_collection()
        if not self._has_searchable_memories(current_incident_id):
            return []
        query_vector = self._embedding_service().embed_query(query)
        if len(query_vector) != self.VECTOR_DIM:
            raise IncidentMemoryError("Embedding 返回的查询向量维度不正确")

        expression = self._base_filter(current_incident_id)
        dense_hits = collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=config.incident_dense_recall_k,
            expr=expression,
            output_fields=self.OUTPUT_FIELDS,
        )[0]
        sparse_hits = collection.search(
            data=[keyword_query or query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25", "params": {}},
            limit=config.incident_bm25_recall_k,
            expr=expression,
            output_fields=self.OUTPUT_FIELDS,
        )[0]

        merged: dict[str, dict[str, Any]] = {}
        rank_constant = config.incident_rrf_rank_constant
        for channel, hits in (("dense", dense_hits), ("bm25", sparse_hits)):
            for rank, hit in enumerate(hits, 1):
                item = self._hit_to_dict(hit)
                incident_id = str(item.get("incident_id") or "")
                if not incident_id:
                    continue
                record = merged.setdefault(
                    incident_id,
                    {
                        **item,
                        "rrf_score": 0.0,
                        "dense_rank": None,
                        "bm25_rank": None,
                    },
                )
                record["rrf_score"] += 1.0 / (rank_constant + rank)
                record[f"{channel}_rank"] = rank

        ranked = sorted(
            merged.values(),
            key=lambda item: (-item["rrf_score"], item["incident_id"]),
        )[:candidate_k]
        if not ranked:
            return []

        ids = ", ".join(
            f'"{self._escape_milvus_string(str(item["incident_id"]))}"'
            for item in ranked
        )
        records = collection.query(
            expr=f"incident_id in [{ids}]",
            output_fields=self.OUTPUT_FIELDS,
            limit=max(2, len(ranked) * 8),
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(str(record.get("incident_id", "")), []).append(record)
        for item in ranked:
            item["records"] = grouped.get(str(item["incident_id"]), [])
        return ranked

    async def search(
        self,
        query: str,
        current_incident_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """执行查询改写、双路 Top20、RRF Top10 和百炼精排 Top3。"""
        normalized = query.strip()
        if not normalized:
            return []
        final_k = min(limit or config.incident_rrf_final_k, config.incident_rrf_final_k)
        try:
            rewritten = await rag_query_rewrite_service.rewrite(normalized)
            candidates = await asyncio.to_thread(
                self._search_sync,
                rewritten.semantic_query,
                current_incident_id,
                config.incident_rrf_candidate_k,
                rewritten.keyword_query,
            )
            return await rag_rerank_service.rerank(
                rewritten.semantic_query,
                candidates,
                final_k=final_k,
            )
        except Exception as exc:
            # 历史记忆是辅助证据；不可用时绝不能阻断当前实时诊断。
            logger.warning(f"故障情景记忆检索不可用，按无历史案例继续: {exc}")
            return []

    @classmethod
    def build_search_query(cls, active_alerts: dict[str, Any]) -> str:
        lines = ["电网业务故障历史案例"]
        for alert in active_alerts.get("alerts", []):
            if not isinstance(alert, dict):
                continue
            lines.extend(
                [
                    f"告警名称 {cls._safe_text(alert.get('alert_name') or alert.get('alertname'))}",
                    f"服务 {cls._safe_text(alert.get('service_name'), 'grid-data-sync-service')}",
                    f"级别 {cls._safe_text(alert.get('severity'))}",
                    f"描述 {cls._safe_text(alert.get('description') or alert.get('summary'), '')}",
                ]
            )
        return "\n".join(lines)

    async def search_for_alerts(
        self,
        active_alerts: dict[str, Any],
        current_incident_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not active_alerts.get("alerts"):
            return []
        return await self.search(
            self.build_search_query(active_alerts),
            current_incident_id=current_incident_id,
        )

    @staticmethod
    def compact_for_prompt(items: list[dict[str, Any]]) -> str:
        compact = []
        for item in items[: config.incident_rrf_final_k]:
            compact.append(
                {
                    "incident_id": item.get("incident_id"),
                    "alert_name": item.get("alert_name"),
                    "service_name": item.get("service_name"),
                    "severity": item.get("severity"),
                    "started_at": item.get("started_at"),
                    "rrf_score": round(float(item.get("rrf_score", 0.0)), 6),
                    "rerank_score": item.get("rerank_score"),
                    "confirmed_summary": str(item.get("content") or "")[:4000],
                }
            )
        return json.dumps(compact, ensure_ascii=False, indent=2)


incident_memory_service = IncidentMemoryService()
