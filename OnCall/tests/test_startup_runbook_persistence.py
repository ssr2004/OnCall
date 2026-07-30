"""启动脚本不得重复上传已经持久化的知识文档。"""

from pathlib import Path


def test_startup_does_not_reupload_runbooks():
    script = Path("start-windows.bat").read_text(encoding="utf-8")

    assert "aiops-docs\\*.md" not in script
    assert 'curl -s -X POST http://localhost:9900/api/upload' not in script
    assert "[8/8] Starting FastAPI server" in script
    assert "Knowledge documents are persisted in Milvus" in script


def test_startup_waits_for_project_owned_infrastructure():
    script = Path("start-windows.bat").read_text(encoding="utf-8")
    compose = Path("vector-database.yml").read_text(encoding="utf-8")
    health_check = Path("scripts/wait-docker-infrastructure.ps1").read_text(
        encoding="utf-8"
    )

    assert 'set "COMPOSE_PROJECT_NAME=oncall"' in script
    assert "docker network create milvus" in script
    assert "docker compose -f vector-database.yml up -d" in script
    assert "MILVUS_STACK_EXISTS" not in script
    assert "wait-docker-infrastructure.ps1" in script
    assert "wait-http-endpoint.ps1" in script
    assert "The API may still be starting" not in script

    for volume in (
        "oncall_milvus_etcd",
        "oncall_milvus_minio",
        "oncall_milvus_data",
    ):
        assert volume in compose
        assert volume in health_check
    assert "external: true" in compose
    assert compose.count("condition: service_healthy") == 3
    assert "Assert-ComposeOwnership" in health_check
    assert "Wait-ContainerHealthy" in health_check
