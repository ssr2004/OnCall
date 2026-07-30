"""故障情景记忆的数据模型。"""

from typing import Any, Literal

from pydantic import BaseModel, Field

IncidentDecision = Literal["pending", "confirmed", "rejected"]


class PendingIncident(BaseModel):
    """AIOps 报告生成后、人工确认前的候选故障事件。"""

    incident_id: str
    session_id: str
    alert_fingerprint: str
    active_alerts: dict[str, Any]
    service_name: str
    alert_name: str
    severity: str
    started_at: int
    report: str
    diagnosis_mode: str = "llm"
    similar_incidents: list[dict[str, Any]] = Field(default_factory=list)
    status: IncidentDecision = "pending"
    created_at: int
    confirmed_at: int | None = None


class IncidentDecisionResponse(BaseModel):
    """确认或拒绝候选故障事件后的统一响应。"""

    success: bool = True
    incident_id: str
    status: IncidentDecision
    persisted: bool = False
    message: str


class IncidentSearchRequest(BaseModel):
    """故障情景记忆混合检索请求。"""

    query: str = Field(min_length=1, max_length=8000)
    current_incident_id: str | None = None
    limit: int | None = Field(default=None, ge=1, le=20)
