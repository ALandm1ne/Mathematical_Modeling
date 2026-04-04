# UAV 路径规划实现总结（2026-04 更新）

## 当前状态

当前系统已切换为**仅外置路径驱动**，不再提供默认条带扫描路径。

- 运行入口：`main.py` 仅接受外置路径来源（配置文件/API）
- 推荐 API：
  - `UAVFleetBuilder.from_custom_json(cfg, json_filepath)`
  - `UAVFleetBuilder.from_path_specs(cfg, path_specs)`
- 已移除旧接口（调用即报错）：
  - `UAVFleetBuilder.from_default_config()`
  - `UAVFleetBuilder.from_strip_scan()`
  - `UAVPathGenerator.generate_strip_scan_paths()`

## 配置收敛（config.py）

路径开关统一集中在 `UAVFleetModeConfig`：

- `custom_paths_json`
- `strict_path_validation=True`
- `require_external_paths=True`
- `missing_path_action`（`warn_and_exit` / `raise`）
- `path_source_conflict_action`（`warn_and_exit` / `raise`）
- `resolve_relative_to_script_dir=True`

`RuntimeSwitchesConfig` 中路径来源相关字段：
- `api_demo_enable`
- `api_demo_json_path`

说明：不再暴露“路径模式切换”配置，UAV 数量由路径条目数决定。

## main.py 路径来源策略

`main.py` 通过 `_resolve_external_path_source()` 统一执行：

1. 收集路径来源
   - `cfg.uav_fleet_mode.custom_paths_json`
   - `cfg.runtime.api_demo_json_path`（仅当 `api_demo_enable=True`）
2. 冲突检测
   - 两个来源同时存在：按 `path_source_conflict_action` 执行（默认告警并退出）
3. 路径解析
   - 相对路径按 `script_dir` 解析（可关闭）
   - 文件不存在：按 `missing_path_action` 执行（默认告警并退出）

若路径未成功解析，仿真直接退出，不进入主循环。

## 强校验定义（strict_path_validation）

当前强校验覆盖（新 schema）：

- 字段/类型/维度合法性
- `start_time_h >= 0`
- 段列表非空
- 段类型必须是 `line/arc`
- `line` 段必须有 `end_point_u`
- `arc` 段必须有 `arc.start_point_u/radius_u/is_clockwise/rotation_angle_deg`
- 圆弧参数中 `radius_u > 0`
- 圆弧参数中 `rotation_angle_deg >= 0`（`0` 允许，视为退化圆弧）
- 段连续性：前一段终点需与 `arc.start_point_u` 一致（容差内）
- 若首段为圆弧，其 `start_point_u` 必须等于 UAV 起点

说明：
- 旧 JSON 字段 `arc_turn.end_point_u/center_u` 已不兼容，检测到直接报错并给迁移提示。
- 圆弧终点不再由 JSON 给定，而由“起点+半径+方向+旋转角度”运动学计算得到。

## 数据与接口

- 单机控制：`UAVController` 仅处理自定义路径分段（typed `line/arc`）
- 机群构建：`UAVFleetController` 仅接受外置路径输入
- 轨迹导出：沿用现有 `DataLogger` 能力（CSV/Parquet）

## 验证结果

建议使用 `uv run`：

```bash
uv run python -m py_compile config.py main.py core/uav_controller.py visualizer.py pictures/3.py test_custom_paths.py
uv run python test_custom_paths.py
```

## 迁移建议

对外调用统一迁移到以下两类：

1. 文件驱动：`from_custom_json()`
2. 内存驱动：`from_path_specs()`

避免任何对默认条带扫描入口的依赖。

---

更新时间：2026-04-04
