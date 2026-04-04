"""UAV 控制器：负责条带扫描状态机与转向几何。

Agent-Oriented API Notes:
- This module exposes stable external APIs via UAVFleetBuilder.
- Data contracts are represented by ArcTurnSpec, SegmentSpec, UAVPathSpec.
- All coordinates are in integer units (u), not km.
"""

import json
import math
from dataclasses import dataclass
from typing import Optional, TypedDict

import numpy as np

__all__ = [
    "ArcTurnSpec",
    "SegmentSpec",
    "UAVPathSpec",
    "UAVPathGenerator",
    "UAVController",
    "UAVFleetController",
    "UAVFleetBuilder",
]


class SegmentSpec(TypedDict):
    """Single path segment contract for agents.

    Keys:
        end_point_u: tuple[int, int]
        arc_turn: ArcTurnSpec | None
    """

    end_point_u: tuple[int, int]
    arc_turn: "ArcTurnSpec | None"


@dataclass
class ArcTurnSpec:
    """Arc-turn contract used by custom-path mode.

    External callers can fully control turn geometry with this object.
    """
    radius_u: float                           # 转弯半径（单位 u）
    start_point_u: tuple[float, float]        # 圆弧起点
    end_point_u: tuple[float, float]          # 圆弧终点
    center_u: Optional[tuple[float, float]] = None  # 圆心（可自动计算或外部指定）
    is_clockwise: bool = True                 # 转弯方向（顺时针/逆时针）


@dataclass
class UAVPathSpec:
    """Full per-UAV path contract used by UAVFleetBuilder.from_path_specs."""
    uav_id: int                               # UAV 编号
    start_time_h: float                       # 绝对出发时间（小时）
    start_pos_u: tuple[int, int]              # 起始位置 (x, y)，单位 u
    segments: list[SegmentSpec]               # 路径段列表
    auto_gen_type: Optional[str] = None       # 可选标记，源自自动生成（如 "strip_scan"）


class UAVPathGenerator:
    """Path-spec factory for agent-driven orchestration.

    Public methods in this class are pure constructors/parsers and do not
    mutate simulation runtime state.
    """

    @staticmethod
    def generate_strip_scan_paths(cfg) -> list[UAVPathSpec]:
        """Generate default strip-scan paths from cfg.

        Args:
            cfg: Application config with fleet and derived fields.

        Returns:
            list[UAVPathSpec]: One path spec per UAV.
        """
        d = cfg.derived
        path_specs: list[UAVPathSpec] = []

        # 计算起始间距（与现有逻辑一致）
        spacing_u = max(1, int(round(cfg.fleet.start_spacing_scan_diameters * 2 * d.uav_scan_radius_u)))
        base_x = d.uav_scan_radius_u
        y0 = 0

        for i in range(cfg.fleet.uav_count):
            x = base_x + i * spacing_u
            x_clamped = max(0, min(d.area_width_u, x))

            # 条带扫描模式：单个段，目标点为区域右上角，无圆弧转弯（由 UAVController.update() 自动管理）
            spec = UAVPathSpec(
                uav_id=i,
                start_time_h=0.0,
                start_pos_u=(int(x_clamped), int(y0)),
                segments=[
                    {
                        "end_point_u": (d.area_width_u, d.area_height_u),
                        "arc_turn": None,  # 条带扫描不指定圆弧，由 update() 自动管理
                    }
                ],
                auto_gen_type="strip_scan",
            )
            path_specs.append(spec)

        return path_specs

    @staticmethod
    def load_custom_paths_from_json(filepath: str) -> list[UAVPathSpec]:
        """Load custom path specs from JSON file.

        Args:
            filepath: Path to UTF-8 JSON array of UAVPathSpec-like objects.

        Returns:
            list[UAVPathSpec]

        Raises:
            FileNotFoundError: If filepath is invalid.
            KeyError/TypeError: If required keys or value types are invalid.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        path_specs: list[UAVPathSpec] = []
        for item in data:
            # 转换 arc_turn 字段为 ArcTurnSpec 对象
            segments = []
            for seg in item.get("segments", []):
                arc_turn_data = seg.get("arc_turn")
                arc_turn = None
                if arc_turn_data is not None:
                    arc_turn = ArcTurnSpec(
                        radius_u=arc_turn_data["radius_u"],
                        start_point_u=tuple(arc_turn_data["start_point_u"]),
                        end_point_u=tuple(arc_turn_data["end_point_u"]),
                        center_u=tuple(arc_turn_data["center_u"]) if arc_turn_data.get("center_u") else None,
                        is_clockwise=arc_turn_data.get("is_clockwise", True),
                    )
                segments.append(
                    {
                        "end_point_u": tuple(seg["end_point_u"]),
                        "arc_turn": arc_turn,
                    }
                )

            spec = UAVPathSpec(
                uav_id=item["uav_id"],
                start_time_h=item.get("start_time_h", 0.0),
                start_pos_u=tuple(item["start_pos_u"]),
                segments=segments,
                auto_gen_type=item.get("auto_gen_type"),
            )
            path_specs.append(spec)

        return path_specs

    @staticmethod
    def validate_path_specs(path_specs: list[UAVPathSpec]) -> None:
        """严格校验路径规范的字段和圆弧几何一致性。"""
        if not path_specs:
            raise ValueError("path_specs must not be empty")

        for index, spec in enumerate(path_specs):
            if not isinstance(spec.uav_id, int):
                raise TypeError(f"path_specs[{index}].uav_id must be int")
            if spec.start_time_h < 0:
                raise ValueError(f"path_specs[{index}].start_time_h must be >= 0")
            if len(spec.start_pos_u) != 2:
                raise ValueError(f"path_specs[{index}].start_pos_u must contain 2 values")
            if not spec.segments:
                raise ValueError(f"path_specs[{index}].segments must not be empty")

            for seg_index, segment in enumerate(spec.segments):
                if "end_point_u" not in segment:
                    raise KeyError(f"path_specs[{index}].segments[{seg_index}] missing end_point_u")
                if len(segment["end_point_u"]) != 2:
                    raise ValueError(
                        f"path_specs[{index}].segments[{seg_index}].end_point_u must contain 2 values"
                    )

                arc_turn = segment.get("arc_turn")
                if arc_turn is None:
                    continue
                if arc_turn.radius_u <= 0:
                    raise ValueError(
                        f"path_specs[{index}].segments[{seg_index}].arc_turn.radius_u must be > 0"
                    )
                if len(arc_turn.start_point_u) != 2 or len(arc_turn.end_point_u) != 2:
                    raise ValueError(
                        f"path_specs[{index}].segments[{seg_index}].arc_turn points must contain 2 values"
                    )

                sx, sy = arc_turn.start_point_u
                ex, ey = arc_turn.end_point_u
                chord_u = math.hypot(float(ex) - float(sx), float(ey) - float(sy))
                if chord_u < 1e-9:
                    raise ValueError(
                        f"path_specs[{index}].segments[{seg_index}].arc_turn start/end must differ"
                    )

                if chord_u > 2.0 * float(arc_turn.radius_u) + 1e-6:
                    raise ValueError(
                        "path_specs["
                        f"{index}].segments[{seg_index}].arc_turn is geometrically impossible: "
                        f"chord={chord_u:.3f} > 2*radius={2.0 * float(arc_turn.radius_u):.3f}"
                    )

                if arc_turn.center_u is not None:
                    cx, cy = arc_turn.center_u
                    ds = math.hypot(float(sx) - float(cx), float(sy) - float(cy))
                    de = math.hypot(float(ex) - float(cx), float(ey) - float(cy))
                    radius_u = float(arc_turn.radius_u)
                    if abs(ds - radius_u) > 1e-3 or abs(de - radius_u) > 1e-3:
                        raise ValueError(
                            f"path_specs[{index}].segments[{seg_index}].arc_turn center/radius mismatch"
                        )


class UAVController:
    """单机 UAV 控制逻辑（设备无关，仅做坐标与角度计算）。"""

    def __init__(
        self,
        cfg,
        uav_id: int = 0,
        start_pos_u: tuple[int, int] | None = None,
        start_time_h: float = 0.0,
    ):
        self.cfg = cfg
        self.uav_id = uav_id

        # 出发时间管理
        self.start_time_h = start_time_h
        self.is_started = start_time_h <= 0.0  # 如果 start_time_h <= 0，立即启动

        # 初始朝向：0.5π，表示沿 +y 方向起飞。
        self.angle = 0.5 * np.pi
        self.is_turning = False
        self.is_turning_clockwise = False
        self.turning_angle_each = 0.0
        self.turn_step_remain = 0

        # 未指定时使用默认起点；多机模式可为每架 UAV 指定独立起点。
        if start_pos_u is None:
            self.pos_u = np.array([self.cfg.derived.uav_scan_radius_u, 0], dtype=np.int64)
        else:
            self.pos_u = np.array([int(start_pos_u[0]), int(start_pos_u[1])], dtype=np.int64)
        self.turn_from_x_u = 0
        self.turn_from_y_u = 0
        self.turn_to_x_u = 0
        self.turn_to_y_u = 0

        # 路径规划字段（支持分段直线 + 圆弧转弯）
        self.segments: list[SegmentSpec] = []
        self.current_segment_idx: int = 0
        self.current_arc_turn: Optional[ArcTurnSpec] = None
        self.state: str = "flying_to_waypoint"  # "flying_to_waypoint" 或 "turning"
        self.auto_gen_type: Optional[str] = None
        self._advance_segment_after_turn: bool = False

    @property
    def position_u(self) -> tuple[int, int]:
        """返回整数单位坐标（用于与粒子系统交互）。"""
        return int(self.pos_u[0]), int(self.pos_u[1])

    def position_km(self) -> tuple[float, float]:
        """返回 km 坐标（用于可视化与日志输出）。"""
        return self.pos_u[0] / self.cfg.numeric.scale, self.pos_u[1] / self.cfg.numeric.scale

    def angle_deg(self) -> float:
        """返回 [0, 360) 的航向角度。"""
        return (self.angle * 180.0 / np.pi) % 360.0

    def is_uav_up(self) -> bool:
        return (self.angle > 0) and (self.angle < np.pi)

    def is_uav_down(self) -> bool:
        return (self.angle > np.pi) and (self.angle < 2 * np.pi)

    def is_uav_outside_top_edge(self) -> bool:
        return self.pos_u[1] >= self.cfg.derived.area_height_u

    def is_uav_outside_bottom_edge(self) -> bool:
        return self.pos_u[1] <= 0

    def get_turn_angle(self, angle_from: float, angle_to: float, clockwise: bool) -> float:
        """按指定方向计算转角，避免不必要的大回转。"""
        diff = (angle_to - angle_from) % self.cfg.derived.two_pi
        if clockwise:
            if diff == 0:
                return 0.0
            return diff - self.cfg.derived.two_pi
        return diff

    def uav_turn_start(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        start_angle: float,
        end_angle: float,
        is_clockwise: bool,
        radius_override_u: float | None = None,
    ) -> None:
        """
        启动一次转向段。

        核心几何：用弦长与圆心角推算转弯半径，再由弧长估算转向步数。
        """
        self.turn_from_x_u = int(start_point[0])
        self.turn_from_y_u = int(start_point[1])
        self.turn_to_x_u = int(end_point[0])
        self.turn_to_y_u = int(end_point[1])

        total_angle = self.get_turn_angle(start_angle, end_angle, is_clockwise)

        dx = float(self.turn_to_x_u - self.turn_from_x_u)
        dy = float(self.turn_to_y_u - self.turn_from_y_u)
        chord_u = math.hypot(dx, dy)
        theta = abs(total_angle)

        # 极小角/极短弦视为无需转向，直接退出转向状态。
        if theta < 1e-9 or chord_u < 1e-9:
            self.turn_step_remain = 0
            self.turning_angle_each = 0.0
            self.is_turning = False
            self.is_turning_clockwise = is_clockwise
            return

        den = 2.0 * math.sin(theta * 0.5)
        if radius_override_u is not None and radius_override_u > 0:
            turn_radius_u = float(radius_override_u)
        elif abs(den) < 1e-9:
            turn_radius_u = float(self.cfg.derived.uav_scan_radius_u)
        else:
            turn_radius_u = chord_u / abs(den)

        arc_length_u = abs(total_angle) * turn_radius_u
        self.turn_step_remain = max(1, int(round(arc_length_u / self.cfg.derived.uav_step_u)))
        self.turning_angle_each = total_angle / self.turn_step_remain
        self.is_turning = True
        self.is_turning_clockwise = is_clockwise

    def is_uav_at_end_corner(self) -> bool:
        """判定是否已完成条带覆盖（右上角向上 / 右下角向下）。"""
        if (
            self.is_uav_outside_top_edge()
            and (self.pos_u[0] + self.cfg.derived.uav_scan_radius_u) >= self.cfg.derived.area_width_u
            and self.is_uav_up()
        ):
            return True
        if (
            self.is_uav_outside_bottom_edge()
            and (self.pos_u[0] + self.cfg.derived.uav_scan_radius_u) >= self.cfg.derived.area_width_u
            and self.is_uav_down()
        ):
            return True
        return False

    def _is_waypoint_reached(self, target_pos_u: tuple[int, int], tolerance_u: int = 1000) -> bool:
        """
        判断是否已到达目标点（使用距离容差）。
        
        参数：
            target_pos_u - 目标点 (x, y)
            tolerance_u - 容差值（整数单位）
        """
        dx = float(self.pos_u[0]) - float(target_pos_u[0])
        dy = float(self.pos_u[1]) - float(target_pos_u[1])
        distance = math.hypot(dx, dy)
        return distance <= tolerance_u

    def _compute_heading_to_waypoint(self, target_pos_u: tuple[int, int]) -> float:
        """计算指向目标点的航向角度（弧度）。"""
        dx = float(target_pos_u[0]) - float(self.pos_u[0])
        dy = float(target_pos_u[1]) - float(self.pos_u[1])
        return math.atan2(dy, dx)

    def update(self, elapsed_time_h: float) -> bool:
        """
        状态机推进一步。支持两种运动模式：
        1. 条带扫描模式（legacy）- 自动掉头逻辑
        2. 自定义路径模式 - 按路径段及圆弧参数运动

        返回：
        - True: 继续扫描
        - False: 扫描完成或达到终止角
        """
        # 检查是否已启动：若未启动且达到出发时间，则激活
        if not self.is_started:
            if elapsed_time_h >= self.start_time_h:
                self.is_started = True
                # 立即执行第一步，避免跳跃
            else:
                # 未到出发时间，保持活跃但不动作
                return True

        # 自定义路径模式处理（直线运动版本）
        if self.segments and self.state != "strip_scan":
            return self._update_custom_path()

        # 条带扫描模式（原有逻辑）
        # 若处于转向阶段，则先累加转角再做位移。
        if self.turn_step_remain > 0:
            self.angle += self.turning_angle_each
            self.turn_step_remain -= 1
            if self.turn_step_remain == 0:
                self.turning_angle_each = 0.0
                self.is_turning = False

        # 角度归一化，避免长期迭代导致数值膨胀。
        self.angle = self.angle % self.cfg.derived.two_pi

        # 基于当前航向推进一步。
        self.pos_u[0] += int(round(self.cfg.derived.uav_step_u * np.cos(self.angle)))
        self.pos_u[1] += int(round(self.cfg.derived.uav_step_u * np.sin(self.angle)))

        if self.is_uav_at_end_corner():
            print(f"UAV#{self.uav_id} scan completed! Elapsed time: {elapsed_time_h:.2f} h")
            return False

        # 到达上下边界后启动掉头，切换到下一条条带。
        if not self.is_turning:
            if self.is_uav_up() and self.is_uav_outside_top_edge():
                self.uav_turn_start(
                    self.pos_u.copy(),
                    np.array([self.pos_u[0] + 2 * self.cfg.derived.uav_scan_radius_u, self.pos_u[1]]),
                    self.angle,
                    1.5 * np.pi,
                    is_clockwise=True,
                )
            elif self.is_uav_down() and self.is_uav_outside_bottom_edge():
                self.uav_turn_start(
                    self.pos_u.copy(),
                    np.array([self.pos_u[0] + 2 * self.cfg.derived.uav_scan_radius_u, self.pos_u[1]]),
                    self.angle,
                    0.5 * np.pi,
                    is_clockwise=False,
                )

        return True

    def _update_custom_path(self) -> bool:
        """
        自定义路径运动更新（直线 + 圆弧的基础版本）。
        
        当前实现：
        - 直线段：计算指向目标点的角度，直线靠近
        - 圆弧转弯：简化处理（保留现有转弯逻辑）
        - 路径完成：到达最后一个段的终点时扫描完成
        """
        # 检查是否所有路段都已完成
        if self.current_segment_idx >= len(self.segments):
            print(f"UAV#{self.uav_id} completed all segments!")
            return False

        current_segment = self.segments[self.current_segment_idx]
        target_point_u = current_segment["end_point_u"]
        arc_turn = current_segment.get("arc_turn")

        step_u = float(self.cfg.derived.uav_step_u)

        # 检查是否处于圆弧转弯中
        if self.is_turning:
            # 继续圆弧转弯
            self.angle += self.turning_angle_each
            self.turn_step_remain -= 1
            if self.turn_step_remain <= 0:
                self.turning_angle_each = 0.0
                self.is_turning = False
                self.turn_step_remain = 0
                if self._advance_segment_after_turn:
                    self.current_segment_idx += 1
                    self._advance_segment_after_turn = False
            
            self.angle = self.angle % self.cfg.derived.two_pi
            self.pos_u[0] += int(round(self.cfg.derived.uav_step_u * np.cos(self.angle)))
            self.pos_u[1] += int(round(self.cfg.derived.uav_step_u * np.sin(self.angle)))
            return True

        # 直线运动：先计算到目标点的距离，避免越点后错过 waypoint。
        dx = float(target_point_u[0]) - float(self.pos_u[0])
        dy = float(target_point_u[1]) - float(self.pos_u[1])
        dist_u = math.hypot(dx, dy)
        approach_angle = math.atan2(dy, dx) if dist_u > 1e-9 else self.angle

        if dist_u <= step_u:
            # 一步内可到达时，直接吸附到 waypoint，确保路径必经。
            self.pos_u[0] = int(target_point_u[0])
            self.pos_u[1] = int(target_point_u[1])
            self.angle = approach_angle

            # 检查是否有圆弧转弯
            if arc_turn is not None:
                # 保证圆弧起点与当前落点对齐，避免几何抖动。
                self.pos_u[0] = int(round(arc_turn.start_point_u[0]))
                self.pos_u[1] = int(round(arc_turn.start_point_u[1]))

                # 启动圆弧转弯
                start_angle = approach_angle
                end_angle = math.atan2(
                    float(arc_turn.end_point_u[1]) - float(arc_turn.start_point_u[1]),
                    float(arc_turn.end_point_u[0]) - float(arc_turn.start_point_u[0]),
                )
                
                self.uav_turn_start(
                    np.array(arc_turn.start_point_u),
                    np.array(arc_turn.end_point_u),
                    start_angle,
                    end_angle,
                    is_clockwise=arc_turn.is_clockwise,
                    radius_override_u=arc_turn.radius_u,
                )
                if self.is_turning:
                    self._advance_segment_after_turn = True
                else:
                    # 若该转弯在几何上退化为 0 步，立即前进到下一段避免停滞。
                    self.current_segment_idx += 1
                    self._advance_segment_after_turn = False
            else:
                # 无转弯，直接进入下一段
                self.current_segment_idx += 1
            return True

        # 尚未到点则按直线方向推进一步。
        target_angle = math.atan2(dy, dx)
        self.angle = target_angle
        self.pos_u[0] += int(round(step_u * np.cos(self.angle)))
        self.pos_u[1] += int(round(step_u * np.sin(self.angle)))

        return True


class UAVFleetController:
    """多机封装：统一初始化、统一步进、统一获取活跃位置。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.controllers: list[UAVController] = []
        self.active_flags: list[bool] = []
        self.last_step_positions_u: list[tuple[int, int]] = []
        self._build_controllers()

    def _build_controllers(self) -> None:
        """
        根据配置模式初始化 UAV 控制器。
        支持两种模式：
        - "auto_strip_scan": 自动条带扫描（默认）
        - "custom_paths": 从 JSON 加载自定义路径
        """
        mode = self.cfg.uav_fleet_mode.mode

        if mode == "custom_paths":
            # 加载自定义路径
            json_file = self.cfg.uav_fleet_mode.custom_paths_json
            if not json_file:
                raise ValueError("custom_paths mode requires custom_paths_json  to be set")
            path_specs = UAVPathGenerator.load_custom_paths_from_json(json_file)
            if self.cfg.uav_fleet_mode.strict_path_validation:
                UAVPathGenerator.validate_path_specs(path_specs)
        else:
            # 默认条带扫描模式
            path_specs = UAVPathGenerator.generate_strip_scan_paths(self.cfg)

        # 从路径规范创建 UAVController
        clamped_count = 0
        overlap_count = 0
        used_x: set[int] = set()

        for spec in path_specs:
            uav = UAVController(
                self.cfg,
                uav_id=spec.uav_id,
                start_pos_u=spec.start_pos_u,
                start_time_h=spec.start_time_h,
            )
            # 设置路径规划字段
            uav.segments = spec.segments
            uav.auto_gen_type = spec.auto_gen_type
            uav.current_segment_idx = 0
            uav.state = "flying_to_waypoint" if spec.segments else "strip_scan"
            # 注：当 segments 为空时，使用旧的条带扫描逻辑

            # 边界检查（仅条带扫描模式）
            x = spec.start_pos_u[0]
            if x not in used_x:
                used_x.add(x)
            else:
                overlap_count += 1

            self.controllers.append(uav)
            self.active_flags.append(True)

        if clamped_count > 0 or overlap_count > 0:
            print(
                "[FLEET][WARN] start positions analyzed: "
                f"overlapped={overlap_count}, "
                f"unique_x={len(used_x)}/{len(self.controllers)}."
            )

    @property
    def primary_controller(self) -> UAVController:
        """返回主 UAV（用于兼容现有日志/可视化接口）。"""
        return self.controllers[0]

    @property
    def active_positions_u(self) -> list[tuple[int, int]]:
        """返回仍在扫描中的 UAV 当前位置列表。"""
        return [
            uav.position_u
            for uav, active in zip(self.controllers, self.active_flags)
            if active
        ]

    @property
    def scan_positions_u(self) -> list[tuple[int, int]]:
        """返回本步应参与扫描的位置（含本步刚结束的 UAV 终点）。"""
        return self.last_step_positions_u

    def update_all(self, elapsed_time_h: float) -> bool:
        """
        推进所有活跃 UAV 一步。

        返回：是否仍至少有一架 UAV 在继续扫描。
        """
        any_active = False
        step_positions: list[tuple[int, int]] = []
        for i, uav in enumerate(self.controllers):
            if not self.active_flags[i]:
                continue
            keep_running = uav.update(elapsed_time_h)
            if uav.is_started:
                step_positions.append(uav.position_u)
            if not keep_running:
                self.active_flags[i] = False
            else:
                any_active = True

        self.last_step_positions_u = step_positions
        return any_active


class UAVFleetBuilder:
    """High-level API surface intended for external scripts and agents.

    Preferred call order for agents:
    1) from_default_config (keep current project behavior)
    2) from_custom_json (when config is file-driven)
    3) from_path_specs (when caller builds objects in-memory)
    """

    @staticmethod
    def from_default_config(cfg) -> UAVFleetController:
        """Build fleet according to cfg.uav_fleet_mode.mode."""
        return UAVFleetController(cfg)

    @staticmethod
    def from_strip_scan(cfg, override_params: dict | None = None) -> UAVFleetController:
        """Build fleet in strip-scan mode with optional temporary overrides.

        Supported override keys:
            - uav_count
            - start_spacing_scan_diameters
        """
        # 保存原始配置
        orig_mode = cfg.uav_fleet_mode.mode
        orig_uav_count = cfg.fleet.uav_count
        orig_spacing = cfg.fleet.start_spacing_scan_diameters

        try:
            # 应用参数覆盖
            cfg.uav_fleet_mode.mode = "auto_strip_scan"
            if override_params:
                if "uav_count" in override_params:
                    cfg.fleet.uav_count = override_params["uav_count"]
                if "start_spacing_scan_diameters" in override_params:
                    cfg.fleet.start_spacing_scan_diameters = override_params["start_spacing_scan_diameters"]
            
            # 重新计算派生参数
            cfg.recompute_derived()

            # 创建控制器
            fleet = UAVFleetController(cfg)
            return fleet
        finally:
            # 恢复原始配置
            cfg.uav_fleet_mode.mode = orig_mode
            cfg.fleet.uav_count = orig_uav_count
            cfg.fleet.start_spacing_scan_diameters = orig_spacing
            cfg.recompute_derived()

    @staticmethod
    def from_custom_json(cfg, json_filepath: str) -> UAVFleetController:
        """Build fleet from custom JSON and switch cfg mode to custom_paths."""
        # 设置配置为自定义路径模式
        cfg.uav_fleet_mode.mode = "custom_paths"
        cfg.uav_fleet_mode.custom_paths_json = json_filepath
        
        return UAVFleetController(cfg)

    @staticmethod
    def from_path_specs(cfg, path_specs: list[UAVPathSpec]) -> UAVFleetController:
        """Build fleet directly from in-memory path specs.

        This path avoids file I/O and is the fastest option for agent pipelines.
        """
        if cfg.uav_fleet_mode.strict_path_validation:
            UAVPathGenerator.validate_path_specs(path_specs)

        # 创建 fleet 对象但跳过 _build_controllers（手动初始化）
        fleet = UAVFleetController.__new__(UAVFleetController)
        fleet.cfg = cfg
        fleet.controllers = []
        fleet.active_flags = []
        fleet.last_step_positions_u = []

        # 手动创建控制器
        for spec in path_specs:
            uav = UAVController(
                cfg,
                uav_id=spec.uav_id,
                start_pos_u=spec.start_pos_u,
                start_time_h=spec.start_time_h,
            )
            uav.segments = spec.segments
            uav.auto_gen_type = spec.auto_gen_type
            uav.current_segment_idx = 0
            uav.state = "flying_to_waypoint" if spec.segments else "strip_scan"

            fleet.controllers.append(uav)
            fleet.active_flags.append(True)
            fleet.last_step_positions_u.append(uav.position_u)

        return fleet
