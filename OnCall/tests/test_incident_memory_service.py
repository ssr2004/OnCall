"""故障情景记忆候选状态、幂等写入和 RRF 融合测试。"""

from typing import Any

import pytest

from app.services.incident_memory_service import (
    IncidentDecisionConflictError,
    IncidentMemoryService,
)


def active_alerts() -> dict[str, Any]:
    return {
        "success": True,
        "total": 1,
        "alerts": [
            {
                "alert_name": "GridCommunicationInterrupted",
                "service_name": "grid-data-sync-service",
                "instance": "grid-simulator:9105",
                "severity": "critical",
                "active_at": "2026-07-28T08:00:00Z",
                "description": "站点通信中断",
            }
        ],
    }


@pytest.mark.asyncio
async def test_confirm_is_idempotent_and_persists_only_once(monkeypatch):
    service = IncidentMemoryService()
    candidate = service.create_candidate(
        "session-1",
        active_alerts(),
        "# 告警分析报告\n\n通信链路中断",
    )
    persisted = []
    monkeypatch.setattr(
        service,
        "_persist_candidate",
        lambda item, confirmed_at: persisted.append((item.incident_id, confirmed_at)),
    )

    first = await service.confirm_incident(candidate.incident_id)
    second = await service.confirm_incident(candidate.incident_id)

    assert first["status"] == "confirmed"
    assert first["persisted"] is True
    assert second["status"] == "confirmed"
    assert len(persisted) == 1


@pytest.mark.asyncio
async def test_rejected_candidate_cannot_be_confirmed_until_aiops_reruns(monkeypatch):
    service = IncidentMemoryService()
    candidate = service.create_candidate(
        "session-1",
        active_alerts(),
        "# 告警分析报告\n\n第一次诊断",
    )

    rejected = await service.reject_incident(candidate.incident_id)
    assert rejected["status"] == "rejected"
    with pytest.raises(IncidentDecisionConflictError):
        await service.confirm_incident(candidate.incident_id)

    rerun = service.create_candidate(
        "session-1",
        active_alerts(),
        "# 告警分析报告\n\n重新诊断",
    )
    writes = []
    monkeypatch.setattr(
        service,
        "_persist_candidate",
        lambda item, confirmed_at: writes.append(item.report),
    )
    confirmed = await service.confirm_incident(rerun.incident_id)
    assert confirmed["status"] == "confirmed"
    assert writes == ["# 告警分析报告\n\n重新诊断"]


class FakeHit:
    def __init__(self, incident_id: str, alert_name: str):
        self.entity = {
            "memory_id": f"{incident_id}:incident_summary",
            "incident_id": incident_id,
            "alert_name": alert_name,
            "service_name": "grid-data-sync-service",
            "memory_type": "incident_summary",
            "status": "closed",
            "verification_status": "confirmed",
            "content": f"{alert_name} 历史报告",
        }


class FakeCollection:
    def __init__(self):
        self.search_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        if kwargs["anns_field"] == "dense_vector":
            return [[FakeHit("inc-a", "通信中断"), FakeHit("inc-b", "队列积压")]]
        return [[FakeHit("inc-b", "队列积压"), FakeHit("inc-c", "同步失败")]]

    def query(self, **_kwargs):
        return [
            {
                "memory_id": f"{incident_id}:incident_summary",
                "incident_id": incident_id,
                "memory_type": "incident_summary",
                "content": f"{incident_id} 完整证据",
            }
            for incident_id in ("inc-a", "inc-b", "inc-c")
        ]


class FakeEmbeddings:
    def embed_query(self, _query):
        return [0.1] * 1024


def test_dense_and_bm25_candidates_are_fused_by_incident_id(monkeypatch):
    service = IncidentMemoryService()
    collection = FakeCollection()
    service._collection = collection
    monkeypatch.setattr(service, "_has_searchable_memories", lambda _current: True)
    monkeypatch.setattr(service, "_embedding_service", lambda: FakeEmbeddings())

    results = service._search_sync("通信中断", "inc-current", 3)

    assert [item["incident_id"] for item in results] == ["inc-b", "inc-a", "inc-c"]
    assert results[0]["dense_rank"] == 2
    assert results[0]["bm25_rank"] == 1
    assert len(collection.search_calls) == 2
    assert collection.search_calls[0]["limit"] == 20
    assert collection.search_calls[1]["limit"] == 20
    assert collection.search_calls[1]["data"] == ["通信中断"]
    assert 'incident_id != "inc-current"' in collection.search_calls[0]["expr"]
    assert all(item["records"] for item in results)
