# 电网站点通信中断告警处理方案

## 告警定义

- 告警名：`GridCommunicationInterrupted`
- 级别：严重
- 条件：在线站点数量连续 15 秒低于应接入站点总数
- 演示场景：`communication_interruption`

## 诊断步骤

1. 使用 `list_active_alerts` 获取受影响服务、站点和告警时间。
2. 使用 `query_grid_service_status` 对比 `grid_station_online` 与 `grid_station_total`。
3. 使用 `query_grid_telemetry_metrics` 检查数据新鲜度是否同步恶化。
4. 使用 `search_log` 查询同一时间窗口内的通信中断日志。

## 根因判断原则

- 服务本身健康但在线站点减少，优先判断采集通道或站端通信异常。
- 如果 Prometheus Target 同时 down，应优先处理服务不可用，不能仅归因于站点通信。
- 没有日志或指标证据时，应明确说明证据不足。

## 处置建议

1. 确认受影响站点范围及通信链路状态。
2. 检查采集通道连接、网络质量和站端数据上送状态。
3. 通信恢复后验证在线站点数量等于站点总数。

