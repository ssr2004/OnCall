"""
AIOps 智能运维接口
"""

import json

from fastapi import APIRouter, HTTPException
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.config import config
from app.models.aiops import AIOpsRequest
from app.models.incident import (
    IncidentDecisionResponse,
    IncidentSearchRequest,
)
from app.services.aiops_service import aiops_service
from app.services.incident_memory_service import (
    IncidentDecisionConflictError,
    IncidentMemoryError,
    IncidentNotFoundError,
    incident_memory_service,
)
from app.services.rag_agent_service import rag_agent_service

router = APIRouter()


@router.get("/aiops/alert-status")
async def get_alert_status():
    """返回当前 Prometheus 活动告警摘要，不调用大模型。"""
    return await aiops_service.get_alert_status()


@router.get("/aiops/incidents/{incident_id}")
async def get_incident(incident_id: str):
    """查询当前进程中的候选事件状态。"""
    try:
        return incident_memory_service.get_candidate(incident_id).model_dump()
    except IncidentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/aiops/incidents/{incident_id}/confirm",
    response_model=IncidentDecisionResponse,
)
async def confirm_incident(incident_id: str):
    """人工确认诊断，并将事件写入 Milvus 长期情景记忆。"""
    try:
        result = await incident_memory_service.confirm_incident(incident_id)
        return IncidentDecisionResponse(**result)
    except IncidentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IncidentDecisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except IncidentMemoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"确认故障事件失败: {exc}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"情景记忆写入失败: {exc}") from exc


@router.post(
    "/aiops/incidents/{incident_id}/reject",
    response_model=IncidentDecisionResponse,
)
async def reject_incident(incident_id: str):
    """人工拒绝诊断，候选报告不会写入 Milvus。"""
    try:
        result = await incident_memory_service.reject_incident(incident_id)
        return IncidentDecisionResponse(**result)
    except IncidentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IncidentDecisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/aiops/incidents/search")
async def search_incidents(request: IncidentSearchRequest):
    """使用 Dense + BM25 + RRF 检索已确认的历史故障事件。"""
    items = await incident_memory_service.search(
        request.query,
        current_incident_id=request.current_incident_id,
        limit=request.limit,
    )
    return {
        "success": True,
        "total": len(items),
        "items": items,
        "retrieval": {
            "dense_k": config.incident_dense_recall_k,
            "bm25_k": config.incident_bm25_recall_k,
            "final_k": min(
                request.limit or config.incident_rrf_final_k,
                config.incident_rrf_final_k,
            ),
            "rrf_rank_constant": config.incident_rrf_rank_constant,
        },
    }


@router.post("/aiops")
async def diagnose_stream(request: AIOpsRequest):
    """
    AIOps 故障诊断接口（流式 SSE）

    **功能说明：**
    - 自动获取当前系统的活动告警
    - 使用 Plan-Execute-Replan 模式进行智能诊断
    - 流式返回诊断过程和结果

    **SSE 事件类型：**

    1. `status` - 状态更新
       ```json
       {
         "type": "status",
         "stage": "fetching_alerts",
         "message": "正在获取系统告警信息..."
       }
       ```

    2. `plan` - 诊断计划制定完成
       ```json
       {
         "type": "plan",
         "stage": "plan_created",
         "message": "诊断计划已制定，共 6 个步骤",
         "target_alert": {...},
         "plan": ["步骤1: ...", "步骤2: ..."]
       }
       ```

    3. `step_complete` - 步骤执行完成
       ```json
       {
         "type": "step_complete",
         "stage": "step_executed",
         "message": "步骤执行完成 (2/6)",
         "current_step": "查询系统日志",
         "result_preview": "...",
         "remaining_steps": 4
       }
       ```

    4. `report` - 最终诊断报告
       ```json
       {
         "type": "report",
         "stage": "final_report",
         "message": "最终诊断报告已生成",
         "report": "# 故障诊断报告\\n...",
         "evidence": {...}
       }
       ```

    5. `complete` - 诊断完成
       ```json
       {
         "type": "complete",
         "stage": "diagnosis_complete",
         "message": "诊断流程完成",
         "diagnosis": {...}
       }
       ```

    6. `error` - 错误信息
       ```json
       {
         "type": "error",
         "stage": "error",
         "message": "诊断过程发生错误: ..."
       }
       ```

    **使用示例：**
    ```bash
    curl -X POST "http://localhost:9900/api/aiops" \\
      -H "Content-Type: application/json" \\
      -d '{"session_id": "session-123"}' \\
      --no-buffer
    ```

    **前端使用示例：**
    ```javascript
    const eventSource = new EventSource('/api/aiops');

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'plan') {
        console.log('诊断计划:', data.plan);
      } else if (data.type === 'step_complete') {
        console.log('步骤完成:', data.current_step);
      } else if (data.type === 'report') {
        console.log('最终报告:', data.report);
      } else if (data.type === 'complete') {
        console.log('诊断完成');
        eventSource.close();
      }
    };
    ```

    Args:
        request: AIOps 诊断请求

    Returns:
        SSE 事件流
    """
    session_id = request.session_id or "default"
    logger.info(f"[会话 {session_id}] 收到 AIOps 诊断请求（流式）")

    async def event_generator():
        context_recorded = False
        try:
            async for event in aiops_service.diagnose(session_id=session_id):
                report = event.get("report") or event.get("response") or ""
                diagnosis = event.get("diagnosis")
                if not report and isinstance(diagnosis, dict):
                    report = diagnosis.get("report", "")

                if report and not context_recorded:
                    try:
                        await rag_agent_service.record_aiops_report(session_id, report)
                        context_recorded = True
                    except Exception as context_error:
                        # 上下文交接失败不应吞掉已经生成的诊断报告；如果后续还有
                        # complete 事件，会再尝试一次。
                        logger.error(
                            f"[会话 {session_id}] AIOps 报告上下文交接失败: "
                            f"{context_error}",
                            exc_info=True,
                        )

                # 发送事件
                yield {
                    "event": "message",
                    "data": json.dumps(event, ensure_ascii=False)
                }

                # 如果是完成或错误事件，结束流
                if event.get("type") in ["complete", "error"]:
                    break

            logger.info(f"[会话 {session_id}] AIOps 诊断流式响应完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] AIOps 诊断流式响应异常: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "stage": "exception",
                    "message": f"诊断异常: {str(e)}"
                }, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())
