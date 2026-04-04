# Monte Carlo 搜索覆盖仿真系统

## 项目概述

本项目实现了基于 GPU 的蒙特卡洛粒子流仿真，并通过多 UAV 协同路径对目标进行搜索覆盖。

## UAV 路径规划（当前版本）

当前版本仅支持**外置路径驱动**：
- 文件驱动：`UAVFleetBuilder.from_custom_json(cfg, json_filepath)`
- 内存驱动：`UAVFleetBuilder.from_path_specs(cfg, path_specs)`

`from_default_config()` 与 `from_strip_scan()` 已移除，不再可用。  
旧 JSON 中的 `arc_turn.end_point_u/center_u` 已不兼容，检测到会直接报错并提示迁移。

## 配置开关（统一在 `config.py`）

核心开关位于 `UAVFleetModeConfig`：

- `custom_paths_json`: 外置路径文件路径
- `strict_path_validation`: 默认 `True`，启用几何与字段强校验
- `require_external_paths`: 默认 `True`，必须提供外置路径
- `missing_path_action`: `"warn_and_exit"` / `"raise"`
- `path_source_conflict_action`: `"warn_and_exit"` / `"raise"`
- `resolve_relative_to_script_dir`: 默认 `True`

`RuntimeSwitchesConfig` 中与路径来源相关字段：
- `api_demo_enable`
- `api_demo_json_path`

说明：`api_demo_*` 仅作为路径来源，不改变路径模式语义。

## 路径来源规则（main）

`main.py` 统一按以下规则解析路径：

1. 收集路径来源：
   - `cfg.uav_fleet_mode.custom_paths_json`
   - `cfg.runtime.api_demo_json_path`（仅当 `api_demo_enable=True`）
2. 若两个来源同时存在：按 `path_source_conflict_action` 执行（默认告警并退出）
3. 若无来源：按 `missing_path_action` 执行（默认告警并退出）
4. 自动解析相对路径到 `script_dir`（可通过配置关闭）

## API 使用示例

### 从 JSON 文件加载（推荐）

```python
from config import build_default_config
from core.uav_controller import UAVFleetBuilder

cfg = build_default_config(".", require_cuda_override=False)
fleet = UAVFleetBuilder.from_custom_json(cfg, "config_templates/uav_paths.json")
```

### 编程式构造路径

```python
from config import build_default_config
from core.uav_controller import UAVFleetBuilder, UAVPathSpec, ArcTurnSpec

cfg = build_default_config(".", require_cuda_override=False)
path_specs = [
    UAVPathSpec(
        uav_id=0,
        start_time_h=0.0,
        start_pos_u=(20000, 0),
        segments=[
            {
                "segment_type": "line",
                "end_point_u": (100000, 100000),
            },
            {
                "segment_type": "arc",
                "arc": ArcTurnSpec(
                    start_point_u=(100000, 100000),
                    radius_u=15000,
                    is_clockwise=True,
                    rotation_angle_deg=90.0,
                )
            }
        ],
    )
]

fleet = UAVFleetBuilder.from_path_specs(cfg, path_specs)
```

## 测试

```bash
uv run python test_custom_paths.py
```
