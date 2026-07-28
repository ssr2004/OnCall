# MCP Servers

为对话 Agent 和 AIOps 诊断 Agent 提供日志与监控工具。主应用通过
`MultiServerMCPClient` 动态发现工具，Agent 不直接依赖具体监控平台 API。

## 服务列表

### CLS Server (`cls_server.py`)

日志查询服务，默认监听 `http://127.0.0.1:8003/mcp`。

当前工具：

- `get_current_timestamp`
- `get_region_code_by_name`
- `get_topic_info_by_name`
- `search_topic_by_service_name`
- `search_log`

CLS Server 当前从电网模拟服务读取场景日志，保证日志与告警、指标来自同一故障状态。
后续可在服务内部替换为甲方日志平台适配器。

### Monitor Server (`monitor_server.py`)

Prometheus 监控适配服务，默认监听 `http://127.0.0.1:8004/mcp`。

当前工具：

- `list_active_alerts`：查询 Prometheus 当前 firing/pending 告警
- `query_grid_service_status`：查询服务健康状态和站点在线情况
- `query_grid_data_sync_metrics`：查询遥测队列积压和同步失败率
- `query_grid_telemetry_metrics`：查询数据新鲜度、处理耗时和消息速率

Monitor MCP 内部调用以下 Prometheus HTTP API：

- `GET /api/v1/alerts`
- `GET /api/v1/query_range`

Agent 只需要使用业务工具，不需要生成 PromQL，也不需要知道 Prometheus 地址、
底层指标名和认证配置。

## Prometheus 配置

项目通过 `vector-database.yml` 启动 Prometheus，默认地址为：

```text
http://127.0.0.1:9090
```

配置文件：

```text
monitoring/prometheus.yml
monitoring/alert-rules.yml
```

本地演示环境默认抓取电网模拟服务：

```text
host.docker.internal:9105
```

模拟服务由项目启动脚本运行在 9105 端口，并为指标提供以下业务标签：

```text
system=grid-oncall-demo
region=demo-grid-region
service=grid-data-sync-service
station=demo-control-center
```

Monitor MCP 默认使用 `service` 标签定位查询目标。

## 环境变量

Monitor MCP 会读取项目根目录的 `.env`：

```dotenv
PROMETHEUS_BASE_URL=http://127.0.0.1:9090
PROMETHEUS_REQUEST_TIMEOUT=10
PROMETHEUS_SERVICE_LABEL=service
GRID_SIMULATOR_BASE_URL=http://127.0.0.1:9105
GRID_SIMULATOR_REQUEST_TIMEOUT=10
```

生产环境如果使用不同的服务标签，例如 `app`，可设置：

```dotenv
PROMETHEUS_SERVICE_LABEL=app
```

## 启动方式

### Docker 基础设施

```bash
docker compose -f vector-database.yml up -d
```

启动后检查：

```text
Prometheus: http://localhost:9090
Prometheus Targets: http://localhost:9090/targets
Prometheus Alerts: http://localhost:9090/alerts
```

### MCP 服务

Linux/macOS：

```bash
make start-cls
make start-monitor
make start-grid-simulator
make status-mcp
```

Windows：

```bat
.venv\Scripts\python.exe mcp_servers\cls_server.py
.venv\Scripts\python.exe mcp_servers\monitor_server.py
.venv\Scripts\python.exe -m uvicorn grid_simulator.service:app --host 0.0.0.0 --port 9105
```

也可以使用项目的 `start-windows.bat` 启动整套服务。

## 工具示例

查询活动告警：

```python
list_active_alerts(
    severity="warning",
    service_name="grid-data-sync-service",
)
```

查询服务健康和站点在线情况：

```python
query_grid_service_status(
    service_name="grid-data-sync-service",
    interval="1m",
)
```

查询指定时间范围的数据同步指标：

```python
query_grid_data_sync_metrics(
    service_name="grid-data-sync-service",
    start_time="2026-07-26 10:00:00",
    end_time="2026-07-26 11:00:00",
    interval="5m",
)
```

成功结果会包含：

- `source=prometheus`
- 查询时间范围
- 原始目标标签
- 时间序列
- 当前值、平均值、最大值、最小值和 P95
- 阈值判断
- 实际 PromQL 查询元数据

Prometheus 不可用时工具返回结构化错误，不会静默切换为随机数据。

## 生产环境接入

如果甲方已有 Prometheus：

1. 将 `PROMETHEUS_BASE_URL` 指向内部 Prometheus。
2. 根据标签规范设置 `PROMETHEUS_SERVICE_LABEL`。
3. 在 Monitor MCP 内将甲方指标映射为现有电网业务工具的返回结构。
4. 在 Monitor MCP 内维护经过验证的 PromQL，不让 Agent 自行生成查询。

如果甲方使用其他内部监控平台，可以保留 MCP 工具名称和返回结构，只替换 Monitor MCP
内部适配器，避免修改对话 Agent 和 AIOps 工作流。

## 数据边界

Prometheus 适合采集服务器、应用、中间件和服务运行指标。SCADA、EMS、DMS 等电网
业务遥测数据应通过甲方授权的只读接口、数据副本或专用适配器接入，不应让 Agent
直接连接现场控制设备。
