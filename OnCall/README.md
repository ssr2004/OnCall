# 电网智能 OnCall 诊断系统

面向电网横向项目演示的智能监控诊断原型。项目通过可控的电网数据采集与同步模拟服务
产生业务指标和日志，由 Prometheus 触发告警，再由 LangGraph Agent 通过 MCP 查询告警、
指标、日志和 Milvus 知识库，最终生成基于证据的诊断报告。

模拟服务只用于演示数据采集与同步链路，不包含任何真实电网控制能力。

## 架构

```text
Grid Service Simulator (:9105)
  ├─ /metrics ───────────────> Prometheus (:9090)
  └─ /api/logs ──────────────> CLS MCP (:8003)
                                   │
Prometheus ──> Monitor MCP (:8004) ├─> LangGraph AIOps Agent
Milvus 故障知识库 ──────────────────┘          │
                                              └─> Web / SSE (:9900)
```

演示环境和真实环境通过 MCP 适配层解耦。后续接入甲方环境时，可以替换 Prometheus
采集目标、日志数据源和指标映射，前端、AIOps 工作流和工具返回结构无需重构。

## 主要模块

- `grid_simulator/service.py`：电网数据采集与同步模拟服务。
- `monitoring/prometheus.yml`：Prometheus 抓取配置。
- `monitoring/alert-rules.yml`：电网业务告警规则。
- `mcp_servers/monitor_server.py`：Prometheus 业务指标适配器。
- `mcp_servers/cls_server.py`：模拟业务日志适配器。
- `app/agent/aiops/`：Planner、Executor、Replanner 工作流。
- `aiops-docs/`：告警处置知识库。

## 一键启动

Windows：

```bat
start-windows.bat
```

Linux/macOS：

```bash
make up
make start
make upload
```

服务地址：

- Web 与 API：<http://localhost:9900>
- 电网模拟服务：<http://localhost:9105>
- 电网模拟控制台：<http://localhost:9105/control>
- 模拟服务状态：<http://localhost:9105/api/status>
- Prometheus：<http://localhost:9090>
- Prometheus Targets：<http://localhost:9090/targets>
- Prometheus Alerts：<http://localhost:9090/alerts>

## 故障场景

演示时推荐直接打开电网模拟控制台：<http://localhost:9105/control>。页面可以查看
实时业务指标、手动注入五类异常、观察告警生效倒计时并一键恢复正常状态，无需执行命令。

以下接口保留用于自动化测试和脚本调用。

正常状态：

```bash
curl -X POST http://localhost:9105/api/recover
```

服务不可用：

```bash
curl -X POST http://localhost:9105/api/scenario/service_down
```

站点通信中断：

```bash
curl -X POST http://localhost:9105/api/scenario/communication_interruption
```

遥测消息积压：

```bash
curl -X POST http://localhost:9105/api/scenario/queue_backlog
```

数据同步失败：

```bash
curl -X POST http://localhost:9105/api/scenario/sync_failure
```

遥测数据延迟：

```bash
curl -X POST http://localhost:9105/api/scenario/data_delay
```

告警规则的持续时间为 15 秒。切换场景后等待约 20 秒，可在 Prometheus Alerts 页面看到
告警进入 firing，然后点击现有 Web 页面的 `AI Ops` 按钮生成诊断报告。

## 推荐演示流程

1. 恢复到 `normal`，展示 Prometheus Target 正常且无告警。
2. 注入 `queue_backlog` 或 `sync_failure` 场景。
3. 展示 Prometheus 业务指标和 firing 告警。
4. 点击现有页面的 `AI Ops` 按钮。
5. 展示 Agent 查询活动告警、指标、日志和知识库并生成报告。
6. 调用 `/api/recover`，展示指标恢复及 Prometheus 告警消失。

如果外部大模型临时不可用，`/api/aiops` 会自动切换到本地确定性诊断模式，
继续通过 MCP 查询实时告警、Prometheus 业务指标和模拟业务日志，并读取本地告警
处置手册生成报告。该降级路径用于保证甲方演示不依赖外部模型账户或网络状态；
外部模型恢复后仍优先使用正常的 Planner-Executor-Replanner 工作流。

当前版本暂不实现告警认领、处置状态流转、Alertmanager 通知和事件关闭留痕。

## 真实环境接入边界

- 通过甲方授权的 Prometheus、日志平台或只读数据接口接入。
- 在 Monitor/CLS MCP 内完成指标与日志格式适配。
- 不让 Agent 直接连接 SCADA、EMS、DMS 等现场控制系统。
- 不向真实设备下发控制指令。
