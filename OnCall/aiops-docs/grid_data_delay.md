# 电网遥测数据更新延迟告警处理方案

## 告警定义

- 告警名：`GridDataFreshnessDelayHigh`
- 条件：`grid_data_freshness_seconds` 连续 15 秒超过 30 秒
- 关联告警：`GridTelemetryProcessingLatencyHigh`
- 条件：`grid_telemetry_processing_latency_seconds` 连续 15 秒超过 2 秒
- 演示场景：`data_delay`

## 诊断步骤

1. 使用 `list_active_alerts` 确认数据新鲜度和处理延迟告警。
2. 使用 `query_grid_telemetry_metrics` 查看新鲜度、处理耗时和消息速率趋势。
3. 使用 `query_grid_data_sync_metrics` 判断是否同时存在队列积压或同步失败。
4. 使用 `search_log` 查询“超过新鲜度窗口”和“处理耗时升高”日志。

## 根因判断原则

- 数据新鲜度与处理耗时同时升高，优先排查处理链路变慢。
- 新鲜度升高且在线站点减少，优先排查通信中断。
- 新鲜度升高且队列持续增长，优先排查消费能力不足。

## 处置建议

1. 定位延迟发生在采集、处理还是同步阶段。
2. 检查批处理、数据校验和下游接口耗时。
3. 恢复后确认数据新鲜度低于 30 秒、处理耗时低于 2 秒。

