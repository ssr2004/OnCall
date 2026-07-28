"""电网业务模拟服务的核心行为测试。"""

from fastapi.testclient import TestClient

from grid_simulator.service import SCENARIOS, app, state


def setup_function() -> None:
    state.set_scenario("normal")


def test_normal_scenario_exposes_business_metrics() -> None:
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "grid_service_health" in response.text
    assert "grid_telemetry_queue_depth" in response.text
    assert 'service="grid-data-sync-service"' in response.text


def test_control_page_exposes_scenario_actions() -> None:
    with TestClient(app) as client:
        response = client.get("/control")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "电网数据采集与同步" in response.text
    assert "queue_backlog" in response.text
    assert "/api/recover" in response.text


def test_queue_backlog_scenario_is_deterministic() -> None:
    snapshot = state.set_scenario("queue_backlog")

    assert snapshot["scenario"] == "queue_backlog"
    assert snapshot["metrics"]["grid_telemetry_queue_depth"] > 1000
    logs = state.query_logs(None, None, "WARN", None, 10)
    assert any(item["scenario"] == "queue_backlog" for item in logs)


def test_service_down_makes_prometheus_scrape_fail_and_can_recover() -> None:
    with TestClient(app) as client:
        down_response = client.post("/api/scenario/service_down")
        metrics_response = client.get("/metrics")
        health_response = client.get("/health")
        recover_response = client.post("/api/recover")
        recovered_metrics = client.get("/metrics")

    assert down_response.status_code == 200
    assert metrics_response.status_code == 503
    assert health_response.status_code == 503
    assert recover_response.status_code == 200
    assert recovered_metrics.status_code == 200


def test_unknown_scenario_returns_available_choices() -> None:
    with TestClient(app) as client:
        response = client.post("/api/scenario/not-a-scenario")

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert set(detail["available_scenarios"]) == set(SCENARIOS)


def test_fault_profiles_cross_their_prometheus_thresholds() -> None:
    communication = state.set_scenario("communication_interruption")["metrics"]
    assert communication["grid_station_online"] < communication["grid_station_total"]

    queue = state.set_scenario("queue_backlog")["metrics"]
    assert queue["grid_telemetry_queue_depth"] > 1000

    sync = state.set_scenario("sync_failure")["metrics"]
    assert sync["grid_data_sync_failure_rate"] > 20

    delay = state.set_scenario("data_delay")["metrics"]
    assert delay["grid_data_freshness_seconds"] > 30
    assert delay["grid_telemetry_processing_latency_seconds"] > 2
