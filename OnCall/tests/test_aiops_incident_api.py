"""AIOps 情景记忆人工反馈接口测试。"""

from unittest.mock import AsyncMock

import pytest

from app.api import aiops as aiops_api
from app.models.incident import IncidentSearchRequest


@pytest.mark.asyncio
async def test_confirm_incident_api_returns_decision(monkeypatch):
    confirm = AsyncMock(
        return_value={
            "incident_id": "inc-123",
            "status": "confirmed",
            "persisted": True,
            "message": "诊断已确认，并已写入长期情景记忆",
        }
    )
    monkeypatch.setattr(aiops_api.incident_memory_service, "confirm_incident", confirm)

    response = await aiops_api.confirm_incident("inc-123")

    confirm.assert_awaited_once_with("inc-123")
    assert response.status == "confirmed"
    assert response.persisted is True


@pytest.mark.asyncio
async def test_search_incident_api_exposes_retrieval_parameters(monkeypatch):
    search = AsyncMock(return_value=[{"incident_id": "inc-old", "rrf_score": 0.03}])
    monkeypatch.setattr(aiops_api.incident_memory_service, "search", search)

    response = await aiops_api.search_incidents(
        IncidentSearchRequest(
            query="电网站点通信中断",
            current_incident_id="inc-current",
        )
    )

    search.assert_awaited_once_with(
        "电网站点通信中断",
        current_incident_id="inc-current",
        limit=None,
    )
    assert response["total"] == 1
    assert response["retrieval"] == {
        "dense_k": 20,
        "bm25_k": 20,
        "final_k": 3,
        "rrf_rank_constant": 60,
    }
