# 电网遥测消息积压告警处理方案

## 告警定义

- 告警名：`GridTelemetryQueueBacklog`
- 级别：警告
- 条件：`grid_telemetry_queue_depth` 连续 15 秒超过 1000
- 演示场景：`queue_backlog`

## 诊断步骤

1. 使用 `list_active_alerts` 确认告警和服务名。
2. 使用 `query_grid_data_sync_metrics` 查看队列深度和同步失败率趋势。
3. 使用 `query_grid_telemetry_metrics` 查看处理耗时和消息处理速率。
4. 使用 `search_log` 查询“消费速度低于采集速度”等 WARN/ERROR 日志。

## 常见原因

- 遥测数据采集速度超过处理速度。
- 下游同步接口变慢导致消费阻塞。
- 批处理任务或数据校验耗时增加。

## 处置建议

1. 检查消息处理速率和下游同步耗时。
2. 临时增加消费能力或降低非关键数据处理压力。
3. 恢复后确认队列深度持续下降并低于阈值。

