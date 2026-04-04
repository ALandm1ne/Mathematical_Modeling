# Monte Carlo 搜索覆盖仿真系统

## 项目概述

本项目实现了一个基于 GPU 加速的蒙特卡洛粒子流仿真系统，配合多 UAV 协同搜索覆盖策略，用于建模和优化无人机群的搜索和跟踪效率。

---

## UAV 路径规划

### 概述

系统支持两种 UAV 路径规划模式：
1. **自动条带扫描模式** (`auto_strip_scan`) — 默认模式，自动生成条带状扫描路径
2. **自定义路径模式** (`custom_paths`) — 从 JSON 配置文件加载自定义路径

每架 UAV 支持：
- **独立出发时间** — 每个 UAV 可在不同时刻启动，通过 `start_time_h` 指定（单位：小时）
- **任意起始位置** — 通过 `start_pos_u` 指定 (x, y) 坐标
- **分段直线路径** — 路径由多个直线段组成
- **圆弧转弯** — 每个路径段可附加圆弧转弯参数，所有参数可由外部脚本修改

### 使用默认条带扫描模式

```python
from config import build_default_config
from core.uav_controller import UAVFleetBuilder

cfg = build_default_config(".")
fleet = UAVFleetBuilder.from_default_config(cfg)
```

### 加载自定义路径

#### 方式 1：从 JSON 文件加载

```python
from config import build_default_config
from core.uav_controller import UAVFleetBuilder

cfg = build_default_config(".")
fleet = UAVFleetBuilder.from_custom_json(cfg, "config_templates/multi_uav_paths.json")
```

#### 方式 2：编程式构造路径

```python
from core.uav_controller import UAVFleetBuilder, UAVPathSpec, ArcTurnSpec

path_specs = [
    UAVPathSpec(
        uav_id=0,
        start_time_h=0.0,
        start_pos_u=(20000, 0),
        segments=[
            {
                "end_point_u": (100000, 100000),
                "arc_turn": ArcTurnSpec(
                    radius_u=15000,
                    start_point_u=(100000, 100000),
                    end_point_u=(150000, 100000),
                    is_clockwise=True,
                ),
            },
            {
                "end_point_u": (200000, 200000),
                "arc_turn": None,
            }
        ]
    )
]

fleet = UAVFleetBuilder.from_path_specs(cfg, path_specs)
```

### JSON 配置文件格式

#### 标准格式示例

```json
[
  {
    "uav_id": 0,
    "start_time_h": 0.0,
    "start_pos_u": [20000, 0],
    "segments": [
      {
        "end_point_u": [100000, 100000],
        "arc_turn": {
          "radius_u": 15000,
          "start_point_u": [100000, 100000],
          "end_point_u": [150000, 100000],
          "center_u": null,
          "is_clockwise": true
        }
      },
      {
        "end_point_u": [200000, 200000],
        "arc_turn": null
      }
    ],
    "auto_gen_type": "custom"
  }
]
```

#### 字段说明

- **uav_id** (int): UAV 编号（从 0 开始）
- **start_time_h** (float): 绝对出发时间，单位小时。为 0 立即启动
- **start_pos_u** (list[int, int]): 起始位置 (x, y)，单位为仿真内部单位 (u)
- **segments** (list): 路径段列表
  - **end_point_u** (list[int, int]): 该段的目标点
  - **arc_turn** (dict or null): 转弯参数（null 表示无转弯）
    - **radius_u** (float): 转弯半径
    - **start_point_u** (list): 圆弧起点
    - **end_point_u** (list): 圆弧终点
    - **center_u** (list or null): 圆心坐标（null 表示由半径自动计算）
    - **is_clockwise** (bool): 转弯方向（true=顺时针，false=逆时针）
- **auto_gen_type** (string or null): 来源标记（"strip_scan"、"custom" 或 null）

### 坐标系统

- **单位 u**：仿真内部使用的整数坐标单位
- **单位 km**：实际地理坐标：`x_km = x_u / 1000`（默认 scale = 1000）

### 配置 JSON 路径加载模式

在 main 脚本中设置配置：

```python
# 方式 1：在运行代码中配置
cfg.uav_fleet_mode.mode = "custom_paths"
cfg.uav_fleet_mode.custom_paths_json = "path/to/my_config.json"

# 方式 2：直接使用 Builder（推荐）
fleet = UAVFleetBuilder.from_custom_json(cfg, "path/to/my_config.json")
```

### 默认运行方式

`main.py` 默认使用 `auto_strip_scan` 模式，不会自动读取 `config_templates/multi_uav_paths.json`。
如果需要切换到演示路径，请在 [config.py](config.py) 中显式设置：

```python
cfg.runtime.api_demo_enable = True
cfg.runtime.api_demo_json_path = "config_templates/multi_uav_paths.json"
cfg.runtime.api_demo_steps = 4000
cfg.runtime.api_demo_realtime_visualization = True
```

同时，如果要在自定义路径下启用几何强校验，也需要显式开启：

```python
cfg.uav_fleet_mode.strict_path_validation = True
```

### API 文档

面向 agent 的最快读取入口：
- 机器可读清单：`api/uav_api_manifest.json`
- 代码类型契约：`core/uav_controller.py` 中的 `ArcTurnSpec`、`SegmentSpec`、`UAVPathSpec`

#### UAVFleetBuilder 工厂类

```python
class UAVFleetBuilder:
    @staticmethod
    def from_default_config(cfg) -> UAVFleetController:
        """使用 cfg.uav_fleet_mode.mode 配置初始化机群"""

    @staticmethod
    def from_strip_scan(cfg, override_params=None) -> UAVFleetController:
        """显式使用条带扫描模式"""

    @staticmethod
    def from_custom_json(cfg, json_filepath: str) -> UAVFleetController:
        """从 JSON 文件加载自定义路径"""

    @staticmethod
    def from_path_specs(cfg, path_specs: list[UAVPathSpec]) -> UAVFleetController:
        """从路径规范列表直接构造"""
```

#### 数据模型

```python
@dataclass
class ArcTurnSpec:
    """圆弧转弯参数"""
    radius_u: float
    start_point_u: tuple[float, float]
    end_point_u: tuple[float, float]
    center_u: Optional[tuple[float, float]] = None
    is_clockwise: bool = True

@dataclass
class UAVPathSpec:
    """完整路径规划规范"""
    uav_id: int
    start_time_h: float
    start_pos_u: tuple[int, int]
    segments: list[dict]  # [{"end_point_u": ..., "arc_turn": ...}, ...]
    auto_gen_type: Optional[str] = None
```

### 测试

运行测试脚本验证各种路径规划模式：

```bash
python test_custom_paths.py
```

输出会显示：
1. 条带扫描模式初始化
2. JSON 文件加载验证
3. 编程式路径构造验证

---

## 向后兼容性

- 现有仅使用条带扫描模式的代码无需修改
- `main.py` 自动使用 `UAVFleetBuilder.from_default_config()` 保持默认行为
- 条带扫描逻辑由工厂自动透明生成

---

## 扩展示例

### 场景 1：不同启动时间的多 UAV 协同

```json
[
  {"uav_id": 0, "start_time_h": 0.0, "start_pos_u": [10000, 0], "segments": [...]},
  {"uav_id": 1, "start_time_h": 2.0, "start_pos_u": [50000, 0], "segments": [...]},
  {"uav_id": 2, "start_time_h": 4.0, "start_pos_u": [90000, 0], "segments": [...]}
]
```

### 场景 2：复杂路径（多个转弯）

```json
[
  {
    "uav_id": 0,
    "start_time_h": 0.0,
    "start_pos_u": [20000, 0],
    "segments": [
      {"end_point_u": [100000, 100000], "arc_turn": {...}},
      {"end_point_u": [200000, 150000], "arc_turn": {...}},
      {"end_point_u": [300000, 250000], "arc_turn": null}
    ]
  }
]
```

---

## 注意事项

1. **坐标系**：所有坐标使用统一的内部单位 (u)，确保与仿真系统一致
2. **时间精度**：`start_time_h` 应与仿真时间步长 `cfg.simulation.dt_h` 配合考虑
3. **路径验证**：暂不在加载时验证路径的几何有效性，运行时若路径不合理会影响仿真效果
4. **圆弧参数**：圆弧启点、终点、半径和方向应保持几何一致
