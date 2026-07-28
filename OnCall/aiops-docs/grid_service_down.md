# 电网数据同步服务不可用告警处理方案

## 告警定义

- 告警名：`GridDataSyncServiceDown`
- 级别：严重
- 条件：Prometheus 连续 15 秒无法抓取 `grid-data-sync-service` 指标端点
- 演示场景：`service_down`

## 诊断步骤

1. 使用 `list_active_alerts` 确认告警状态、服务名、首次触发时间和实例。
2. 使用 `query_grid_service_status` 查询告警前后的服务健康和站点在线趋势。
3. 使用 `search_topic_by_service_name(service_name="grid-data-sync-service")` 获取日志主题。
4. 使用 `search_log` 查询告警时间窗口内的 ERROR 日志。
5. 结合指标缺失时间与“健康检查失败”日志确认服务不可用，而不是编造 CPU、内存原因。

## 常见原因

- 数据采集与同步进程退出或未启动。
- 指标端点不可访问。
- 服务监听端口或网络链路异常。
- 配置加载失败导致服务无法完成初始化。

## 处置建议

1. 检查服务进程、健康接口和指标接口。
2. 检查最近配置或版本变更。
3. 恢复服务后确认 Prometheus Target 重新变为 up。
4. 持续观察至少一个告警评估周期，确认告警消失。

## 演示恢复

```bash
curl -X POST http://localhost:9105/api/recover
```

