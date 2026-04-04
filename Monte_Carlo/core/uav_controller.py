"""UAV 控制器：负责外置路径驱动下的分段运动学。

Agent-Oriented API Notes:
- This module exposes stable external APIs via UAVFleetBuilder.
- Data contracts are represented by ArcTurnSpec, SegmentSpec, UAVPathSpec.
- All coordinates are in integer units (u), not km.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Literal, Optional, TypedDict

import numpy as np

__all__ = [
    "ArcTurnSpec",
    "SegmentSpec",
    "UAVPathSpec",
    "UAVPathGenerator",
    "UAVController",
    "UAVFleetController",
    "UAVFleetBuilder",
    "ReplanningMetadata",
]

_POINT_TOL_U = 1e-6
_ANGLE_TOL_RAD = 1e-12
_SEGMENT_CONTINUITY_TOL_U = 100.0

_COMMON_ANGLE_DEG_TO_RAD = {
    0: 0.0,
    15: math.pi / 12.0,
    30: math.pi / 6.0,
    45: math.pi / 4.0,
    60: math.pi / 3.0,
    75: 5.0 * math.pi / 12.0,
    90: math.pi / 2.0,
    105: 7.0 * math.pi / 12.0,
    120: 2.0 * math.pi / 3.0,
    135: 3.0 * math.pi / 4.0,
    150: 5.0 * math.pi / 6.0,
    165: 11.0 * math.pi / 12.0,
    180: math.pi,
    195: 13.0 * math.pi / 12.0,
    210: 7.0 * math.pi / 6.0,
    225: 5.0 * math.pi / 4.0,
    240: 4.0 * math.pi / 3.0,
    255: 17.0 * math.pi / 12.0,
    270: 3.0 * math.pi / 2.0,
    285: 19.0 * math.pi / 12.0,
    300: 5.0 * math.pi / 3.0,
    315: 7.0 * math.pi / 4.0,
    330: 11.0 * math.pi / 6.0,
    345: 23.0 * math.pi / 12.0,
}


class LineSegmentSpec(TypedDict):
    """直线段：从当前点飞到 end_point_u。"""

    segment_type: Literal["line"]
    end_point_u: tuple[int, int]


@dataclass
class ArcTurnSpec:
    """圆弧段：由起点、半径、方向、旋转角度定义。"""

    start_point_u: tuple[float, float]
    radius_u: float
    is_clockwise: bool
    rotation_angle_deg: float


class ArcSegmentSpec(TypedDict):
    """圆弧段：由 arc 定义独立段。"""

    segment_type: Literal["arc"]
    arc: ArcTurnSpec


SegmentSpec = LineSegmentSpec | ArcSegmentSpec


@dataclass
class ReplanningTriggerEvent:
    """动态重规划触发事件记录"""
    step_idx: int                           # 触发时的仿真步数
    time_h: float                           # 触发时的仿真时间（小时）
    trigger_pos_u: tuple[int, int]         # 触发位置
    trigger_boundary: str                   # 触发的边界："top" 或 "bottom"
    candidate_strips: list[tuple[int, int, int]] = field(default_factory=list)  # [(strip_id, score, rank), ...]
    selected_strip_id: int = -1            # 最终选中的条带ID
    new_segments_count: int = 0            # 插入的新段数量
    tolerance_switched: bool = False       # 是否因容差规则改选了相邻条带
    tolerance_best_strip_id: int = -1      # 容差判定前的best条带
    tolerance_adjacent_avg: float = 0.0    # 相邻可用条带平均得分
    tolerance_relative_gain: float = 0.0   # best/adj_avg - 1
    tolerance_threshold: float = 0.0       # 容差阈值


@dataclass
class ReplanningMetadata:
    """单个UAV的重规划元数据与触发历史"""
    uav_id: int
    assigned_x_range_u: tuple[int, int]    # 该UAV负责的x区间 (x_min, x_max)
    last_replan_step: int = -1             # 上次重规划的步数
    boundary_latch: Optional[str] = None   # 边界锁存：top/bottom，离开边界区后清除
    replan_events: list[ReplanningTriggerEvent] = field(default_factory=list)


@dataclass
class UAVPathSpec:
    """Full per-UAV path contract used by UAVFleetBuilder.from_path_specs."""

    uav_id: int
    start_time_h: float
    start_pos_u: tuple[int, int]
    segments: list[SegmentSpec]
    auto_gen_type: Optional[str] = None


class UAVPathGenerator:
    """Path-spec factory for agent-driven orchestration."""

    @staticmethod
    def generate_strip_scan_paths(cfg) -> list[UAVPathSpec]:
        """Legacy API removed: strip-scan paths are no longer supported."""

        raise NotImplementedError(
            "generate_strip_scan_paths() has been removed. "
            "Please use external paths via from_custom_json() or from_path_specs()."
        )

    @staticmethod
    def _normalize_angle_0_2pi(angle_rad: float) -> float:
        angle = angle_rad % (2.0 * math.pi)
        if angle < 0:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _deg_to_rad_stable(angle_deg: float) -> float:
        """稳定角度换算：常见角优先精确映射，其他走 radians。"""

        if not math.isfinite(angle_deg):
            raise ValueError("rotation_angle_deg must be finite")
        if abs(angle_deg) <= 1e-12:
            return 0.0

        rounded = round(angle_deg)
        if abs(angle_deg - rounded) <= 1e-12:
            full_turns = int(rounded // 360)
            rem = int(rounded % 360)
            if rem in _COMMON_ANGLE_DEG_TO_RAD:
                return full_turns * (2.0 * math.pi) + _COMMON_ANGLE_DEG_TO_RAD[rem]

        return math.radians(angle_deg)

    @staticmethod
    def _distance_u(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))

    @staticmethod
    def _parse_point2(name: str, value) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must be a 2D point")
        x = float(value[0])
        y = float(value[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{name} must contain finite values")
        return (x, y)

    @staticmethod
    def _build_arc_center_from_heading(
        start_point_u: tuple[float, float],
        start_heading_rad: float,
        radius_u: float,
        is_clockwise: bool,
    ) -> tuple[float, float]:
        hx = math.cos(start_heading_rad)
        hy = math.sin(start_heading_rad)
        if is_clockwise:
            nx, ny = hy, -hx
        else:
            nx, ny = -hy, hx
        return (start_point_u[0] + nx * radius_u, start_point_u[1] + ny * radius_u)

    @staticmethod
    def _compute_arc_end_from_state(
        start_point_u: tuple[float, float],
        start_heading_rad: float,
        arc_spec: ArcTurnSpec,
    ) -> tuple[tuple[float, float], float]:
        total_abs = UAVPathGenerator._deg_to_rad_stable(float(arc_spec.rotation_angle_deg))
        if total_abs <= _ANGLE_TOL_RAD:
            return start_point_u, UAVPathGenerator._normalize_angle_0_2pi(start_heading_rad)

        signed = -total_abs if arc_spec.is_clockwise else total_abs
        center_u = UAVPathGenerator._build_arc_center_from_heading(
            start_point_u,
            start_heading_rad,
            float(arc_spec.radius_u),
            bool(arc_spec.is_clockwise),
        )
        start_radial = math.atan2(
            float(start_point_u[1]) - float(center_u[1]),
            float(start_point_u[0]) - float(center_u[0]),
        )
        end_radial = start_radial + signed
        end_point = (
            float(center_u[0]) + float(arc_spec.radius_u) * math.cos(end_radial),
            float(center_u[1]) + float(arc_spec.radius_u) * math.sin(end_radial),
        )
        end_heading = UAVPathGenerator._normalize_angle_0_2pi(start_heading_rad + signed)
        return end_point, end_heading

    @staticmethod
    def _migration_error_message() -> str:
        return (
            "Legacy arc schema is not compatible anymore. "
            "Detected old key `arc_turn` (or arc end/center fields). "
            "Please migrate segment to: "
            "{'segment_type':'arc','arc':{'start_point_u':[x,y],'radius_u':r,'is_clockwise':true,'rotation_angle_deg':90}}"
        )

    @staticmethod
    def load_custom_paths_from_json(filepath: str) -> list[UAVPathSpec]:
        """Load custom path specs from JSON file (new typed segment schema)."""

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("custom path JSON root must be a list")

        path_specs: list[UAVPathSpec] = []
        for item_idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"json[{item_idx}] must be an object")
            raw_segments = item.get("segments", [])
            if not isinstance(raw_segments, list):
                raise ValueError(f"json[{item_idx}].segments must be a list")

            segments: list[SegmentSpec] = []
            for seg_idx, seg in enumerate(raw_segments):
                if not isinstance(seg, dict):
                    raise ValueError(f"json[{item_idx}].segments[{seg_idx}] must be an object")
                if "arc_turn" in seg:
                    raise ValueError(UAVPathGenerator._migration_error_message())

                seg_type = seg.get("segment_type")
                if seg_type == "line":
                    end_point = UAVPathGenerator._parse_point2(
                        f"json[{item_idx}].segments[{seg_idx}].end_point_u",
                        seg.get("end_point_u"),
                    )
                    segments.append(
                        {
                            "segment_type": "line",
                            "end_point_u": (int(round(end_point[0])), int(round(end_point[1]))),
                        }
                    )
                    continue

                if seg_type == "arc":
                    arc_raw = seg.get("arc")
                    if not isinstance(arc_raw, dict):
                        raise ValueError(f"json[{item_idx}].segments[{seg_idx}].arc must be object")
                    if "end_point_u" in arc_raw or "center_u" in arc_raw:
                        raise ValueError(UAVPathGenerator._migration_error_message())

                    start_point = UAVPathGenerator._parse_point2(
                        f"json[{item_idx}].segments[{seg_idx}].arc.start_point_u",
                        arc_raw.get("start_point_u"),
                    )
                    arc = ArcTurnSpec(
                        start_point_u=start_point,
                        radius_u=float(arc_raw["radius_u"]),
                        is_clockwise=bool(arc_raw.get("is_clockwise", True)),
                        rotation_angle_deg=float(arc_raw["rotation_angle_deg"]),
                    )
                    segments.append({"segment_type": "arc", "arc": arc})
                    continue

                raise ValueError(
                    f"json[{item_idx}].segments[{seg_idx}].segment_type must be 'line' or 'arc'"
                )

            spec = UAVPathSpec(
                uav_id=int(item["uav_id"]),
                start_time_h=float(item.get("start_time_h", 0.0)),
                start_pos_u=tuple(int(v) for v in item["start_pos_u"]),
                segments=segments,
                auto_gen_type=item.get("auto_gen_type"),
            )
            path_specs.append(spec)

        return path_specs

    @staticmethod
    def validate_path_specs(path_specs: list[UAVPathSpec]) -> None:
        """严格校验新 schema：字段、连续性、与可构造性。"""

        if not path_specs:
            raise ValueError("path_specs must not be empty")

        for i, spec in enumerate(path_specs):
            if not isinstance(spec.uav_id, int):
                raise TypeError(f"path_specs[{i}].uav_id must be int")
            if not math.isfinite(float(spec.start_time_h)) or spec.start_time_h < 0:
                raise ValueError(f"path_specs[{i}].start_time_h must be finite and >= 0")
            if len(spec.start_pos_u) != 2:
                raise ValueError(f"path_specs[{i}].start_pos_u must contain 2 values")
            if not spec.segments:
                raise ValueError(f"path_specs[{i}].segments must not be empty")

            current_point = (float(spec.start_pos_u[0]), float(spec.start_pos_u[1]))
            current_heading = 0.5 * math.pi

            for j, segment in enumerate(spec.segments):
                if not isinstance(segment, dict):
                    raise TypeError(f"path_specs[{i}].segments[{j}] must be dict")
                if "arc_turn" in segment:
                    raise ValueError(UAVPathGenerator._migration_error_message())

                seg_type = segment.get("segment_type")
                if seg_type not in ("line", "arc"):
                    raise ValueError(
                        f"path_specs[{i}].segments[{j}].segment_type must be 'line' or 'arc'"
                    )

                if seg_type == "line":
                    if "end_point_u" not in segment:
                        raise KeyError(f"path_specs[{i}].segments[{j}] missing end_point_u")
                    end = UAVPathGenerator._parse_point2(
                        f"path_specs[{i}].segments[{j}].end_point_u",
                        segment["end_point_u"],
                    )
                    dx = end[0] - current_point[0]
                    dy = end[1] - current_point[1]
                    if abs(dx) > _POINT_TOL_U and abs(dy) > _POINT_TOL_U:
                        raise ValueError(
                            f"path_specs[{i}].segments[{j}] line must be axis-aligned"
                        )
                    if math.hypot(dx, dy) > _POINT_TOL_U:
                        if abs(dx) > _POINT_TOL_U:
                            current_heading = 0.0 if dx > 0 else math.pi
                        else:
                            current_heading = 0.5 * math.pi if dy > 0 else 1.5 * math.pi
                    current_point = end
                    continue

                if "arc" not in segment:
                    raise KeyError(f"path_specs[{i}].segments[{j}] missing arc")
                arc = segment["arc"]
                if not isinstance(arc, ArcTurnSpec):
                    raise TypeError(f"path_specs[{i}].segments[{j}].arc must be ArcTurnSpec")
                if arc.radius_u <= 0:
                    raise ValueError(f"path_specs[{i}].segments[{j}].arc.radius_u must be > 0")
                if not math.isfinite(float(arc.rotation_angle_deg)):
                    raise ValueError(
                        f"path_specs[{i}].segments[{j}].arc.rotation_angle_deg must be finite"
                    )
                if arc.rotation_angle_deg < 0:
                    raise ValueError(
                        f"path_specs[{i}].segments[{j}].arc.rotation_angle_deg must be >= 0"
                    )
                rot_mod_90 = float(arc.rotation_angle_deg) % 90.0
                if not math.isclose(rot_mod_90, 0.0, abs_tol=1e-9):
                    raise ValueError(
                        f"path_specs[{i}].segments[{j}].arc.rotation_angle_deg must be a multiple of 90"
                    )
                if len(arc.start_point_u) != 2:
                    raise ValueError(
                        f"path_specs[{i}].segments[{j}].arc.start_point_u must contain 2 values"
                    )

                start = UAVPathGenerator._parse_point2(
                    f"path_specs[{i}].segments[{j}].arc.start_point_u",
                    arc.start_point_u,
                )
                if UAVPathGenerator._distance_u(start, current_point) > _SEGMENT_CONTINUITY_TOL_U:
                    if j == 0:
                        raise ValueError(
                            f"path_specs[{i}].segments[{j}] arc.start_point_u must equal start_pos_u"
                        )
                    raise ValueError(
                        f"path_specs[{i}].segments[{j}] arc.start_point_u is not continuous with previous segment end"
                    )

                end_point, end_heading = UAVPathGenerator._compute_arc_end_from_state(
                    start,
                    current_heading,
                    arc,
                )
                current_point = end_point
                current_heading = end_heading


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

        self.start_time_h = start_time_h
        self.is_started = start_time_h <= 0.0

        self.angle = 0.5 * np.pi
        self.is_turning = False
        self.is_turning_clockwise = False
        self.turning_angle_each = 0.0
        self.turn_step_remain = 0

        if start_pos_u is None:
            self.pos_u = np.array([self.cfg.derived.uav_scan_radius_u, 0], dtype=np.int64)
        else:
            self.pos_u = np.array([int(start_pos_u[0]), int(start_pos_u[1])], dtype=np.int64)

        self.segments: list[SegmentSpec] = []
        self.current_segment_idx: int = 0
        self.state: str = "flying_to_waypoint"
        self.auto_gen_type: Optional[str] = None

        # 圆弧段运行时状态
        self.current_arc_turn: Optional[ArcTurnSpec] = None
        self._arc_center_u: Optional[np.ndarray] = None
        self._arc_start_radial_angle: float = 0.0
        self._arc_total_angle: float = 0.0
        self._arc_progress_angle: float = 0.0
        self._arc_start_heading: float = 0.0
        self._arc_radius_u: float = 0.0
        
        # 动态重规划元数据
        self.replanning_metadata: ReplanningMetadata = ReplanningMetadata(
            uav_id=uav_id,
            assigned_x_range_u=(0, self.cfg.derived.area_width_u),
        )

    @property
    def position_u(self) -> tuple[int, int]:
        return int(self.pos_u[0]), int(self.pos_u[1])

    def position_km(self) -> tuple[float, float]:
        return self.pos_u[0] / self.cfg.numeric.scale, self.pos_u[1] / self.cfg.numeric.scale

    def angle_deg(self) -> float:
        return (self.angle * 180.0 / np.pi) % 360.0

    def _is_waypoint_reached(self, target_pos_u: tuple[float, float], tolerance_u: int = 1000) -> bool:
        dx = float(self.pos_u[0]) - float(target_pos_u[0])
        dy = float(self.pos_u[1]) - float(target_pos_u[1])
        return math.hypot(dx, dy) <= tolerance_u

    def _compute_heading_to_waypoint(self, target_pos_u: tuple[float, float]) -> float:
        dx = float(target_pos_u[0]) - float(self.pos_u[0])
        dy = float(target_pos_u[1]) - float(self.pos_u[1])
        return math.atan2(dy, dx)

    def _check_boundary_trigger(
        self, current_step: int
    ) -> tuple[bool, Optional[str]]:
        """
        检查是否触及上/下边界，是否应该触发重规划。
        返回：(should_trigger, boundary_type)，boundary_type 为 "top"/"bottom"/None
        """
        if not self.cfg.dynamic_replanning.enable:
            return False, None

        # 仅在直线飞行阶段允许触发；转弯或弧段执行时禁止触发，避免重复注入重规划段。
        if self.is_turning:
            return False, None
        if self.current_segment_idx < len(self.segments):
            seg = self.segments[self.current_segment_idx]
            if isinstance(seg, dict) and seg.get("segment_type") == "arc":
                return False, None
        
        # 节流检查：避免同一UAV高频重规划
        steps_since_last = current_step - self.replanning_metadata.last_replan_step
        if steps_since_last < self.cfg.dynamic_replanning.min_steps_between_replans:
            return False, None
        
        d = self.cfg.derived
        scan_radius = d.uav_scan_radius_u
        tolerance = max(500, min(self.cfg.dynamic_replanning.trigger_tolerance_u, d.uav_step_u))
        
        y_pos = int(self.pos_u[1])
        heading_sin = math.sin(float(self.angle))
        heading_gate = 0.1  # 仅当有明确朝向边界趋势时触发
        top_trigger_y = d.area_height_u - scan_radius
        bottom_trigger_y = scan_radius
        in_bottom_zone = abs(y_pos - bottom_trigger_y) <= tolerance
        in_top_zone = abs(y_pos - top_trigger_y) <= tolerance

        # 边界锁存复位：只有离开上下边界触发区后才允许再次触发。
        if not in_top_zone and not in_bottom_zone:
            self.replanning_metadata.boundary_latch = None
        
        # 检查下边界 (y ≈ 0)
        if in_bottom_zone and heading_sin < -heading_gate:
            if self.replanning_metadata.boundary_latch == "bottom":
                return False, None
            self.replanning_metadata.boundary_latch = "bottom"
            return True, "bottom"
        
        # 检查上边界 (y ≈ area_height_u)
        if in_top_zone and heading_sin > heading_gate:
            if self.replanning_metadata.boundary_latch == "top":
                return False, None
            self.replanning_metadata.boundary_latch = "top"
            return True, "top"
        
        return False, None

    def record_replanning_trigger(
        self,
        step_idx: int,
        time_h: float,
        boundary_type: str,
        candidate_strips: list[tuple[int, int, int]],
        selected_strip_id: int,
        new_segments_count: int = 0,
        tolerance_switched: bool = False,
        tolerance_best_strip_id: int = -1,
        tolerance_adjacent_avg: float = 0.0,
        tolerance_relative_gain: float = 0.0,
        tolerance_threshold: float = 0.0,
    ) -> None:
        """记录一次重规划触发事件"""
        event = ReplanningTriggerEvent(
            step_idx=step_idx,
            time_h=time_h,
            trigger_pos_u=self.position_u,
            trigger_boundary=boundary_type,
            candidate_strips=candidate_strips,
            selected_strip_id=selected_strip_id,
            new_segments_count=new_segments_count,
            tolerance_switched=bool(tolerance_switched),
            tolerance_best_strip_id=int(tolerance_best_strip_id),
            tolerance_adjacent_avg=float(tolerance_adjacent_avg),
            tolerance_relative_gain=float(tolerance_relative_gain),
            tolerance_threshold=float(tolerance_threshold),
        )
        self.replanning_metadata.replan_events.append(event)
        self.replanning_metadata.last_replan_step = step_idx
        print(
            f"[REPLAN] UAV#{self.uav_id} triggered at step={step_idx}, "
            f"boundary={boundary_type}, selected_strip={selected_strip_id}"
        )

    def inject_segments_after_current(
        self, new_segments: list[SegmentSpec]
    ) -> None:
        """
        将新段注入到当前段之后（不改变当前处理段）。
        保持段索引和状态的一致性。
        """
        if not new_segments:
            return
        for idx, seg in enumerate(new_segments):
            if not isinstance(seg, dict):
                raise TypeError(f"Injected segment[{idx}] must be dict")

            seg_type = seg.get("segment_type")
            if seg_type == "line":
                if "end_point_u" not in seg:
                    raise KeyError(f"Injected line segment[{idx}] missing end_point_u")
            elif seg_type == "arc":
                if "arc" not in seg:
                    raise KeyError(f"Injected arc segment[{idx}] missing arc")
                arc = self._normalize_arc_turn_spec(seg["arc"])
                if not math.isclose(float(arc.rotation_angle_deg) % 360.0, 90.0, abs_tol=1e-9):
                    raise ValueError("Injected arc segment rotation_angle_deg must be 90")
                seg["arc"] = arc
            else:
                raise ValueError(f"Injected segment[{idx}] has invalid segment_type")

        # 在 current_segment_idx 之后替换后续段，避免新旧路径混合导致不连续。
        insert_pos = self.current_segment_idx + 1
        self.segments = self.segments[:insert_pos] + new_segments
        print(
            f"[INJECT] UAV#{self.uav_id} injected {len(new_segments)} segments "
            f"after index {self.current_segment_idx} (tail replaced)"
        )

    def _move_toward_point(self, target_pos_u: tuple[float, float], step_u: float) -> bool:
        """向目标点推进一步；返回是否已在本步到达。"""

        dx = float(target_pos_u[0]) - float(self.pos_u[0])
        dy = float(target_pos_u[1]) - float(self.pos_u[1])
        dist_u = math.hypot(dx, dy)
        if dist_u <= step_u:
            self.pos_u[0] = int(round(target_pos_u[0]))
            self.pos_u[1] = int(round(target_pos_u[1]))
            if dist_u > _POINT_TOL_U:
                self.angle = math.atan2(dy, dx)
            self.angle = UAVPathGenerator._normalize_angle_0_2pi(self.angle)
            return True

        self.angle = math.atan2(dy, dx)
        self.pos_u[0] += int(round(step_u * math.cos(self.angle)))
        self.pos_u[1] += int(round(step_u * math.sin(self.angle)))
        self.angle = UAVPathGenerator._normalize_angle_0_2pi(self.angle)
        return False

    def _move_axis_aligned_toward_point(self, target_pos_u: tuple[float, float], step_u: float) -> bool:
        """沿坐标轴推进线段，禁止斜向运动。"""

        tx = float(target_pos_u[0])
        ty = float(target_pos_u[1])
        cx = float(self.pos_u[0])
        cy = float(self.pos_u[1])
        dx = tx - cx
        dy = ty - cy
        axis_tol = 1.0

        if abs(dx) <= axis_tol and abs(dy) <= axis_tol:
            self.pos_u[0] = int(round(tx))
            self.pos_u[1] = int(round(ty))
            return True

        if abs(dx) <= axis_tol:
            self.pos_u[0] = int(round(tx))
            step = min(step_u, abs(dy))
            if dy > 0:
                self.pos_u[1] += int(round(step))
                self.angle = 0.5 * math.pi
            else:
                self.pos_u[1] -= int(round(step))
                self.angle = 1.5 * math.pi
            if abs(float(self.pos_u[1]) - ty) <= axis_tol:
                self.pos_u[1] = int(round(ty))
                return True
            return False

        if abs(dy) <= axis_tol:
            self.pos_u[1] = int(round(ty))
            step = min(step_u, abs(dx))
            if dx > 0:
                self.pos_u[0] += int(round(step))
                self.angle = 0.0
            else:
                self.pos_u[0] -= int(round(step))
                self.angle = math.pi
            if abs(float(self.pos_u[0]) - tx) <= axis_tol:
                self.pos_u[0] = int(round(tx))
                return True
            return False

        raise RuntimeError(
            f"UAV#{self.uav_id} line segment is not axis-aligned: "
            f"current={self.position_u}, target=({int(round(tx))}, {int(round(ty))})"
        )

    def uav_turn_start(self, arc_spec: ArcTurnSpec, start_heading_rad: float) -> None:
        """启动圆弧段：由起点、半径、方向、旋转角驱动。"""

        total_abs = UAVPathGenerator._deg_to_rad_stable(float(arc_spec.rotation_angle_deg))
        if total_abs <= _ANGLE_TOL_RAD:
            self.is_turning = False
            self.turn_step_remain = 0
            self.turning_angle_each = 0.0
            self.current_arc_turn = None
            return

        start_point = np.array([float(arc_spec.start_point_u[0]), float(arc_spec.start_point_u[1])], dtype=float)
        center = UAVPathGenerator._build_arc_center_from_heading(
            (float(arc_spec.start_point_u[0]), float(arc_spec.start_point_u[1])),
            start_heading_rad,
            float(arc_spec.radius_u),
            bool(arc_spec.is_clockwise),
        )
        center_arr = np.array([center[0], center[1]], dtype=float)

        signed_total = -total_abs if arc_spec.is_clockwise else total_abs
        arc_length_u = abs(signed_total) * float(arc_spec.radius_u)
        step_u = float(self.cfg.derived.uav_step_u)
        steps = max(1, int(math.ceil(arc_length_u / step_u)))

        self.is_turning = True
        self.is_turning_clockwise = bool(arc_spec.is_clockwise)
        self.turn_step_remain = steps
        self.turning_angle_each = signed_total / steps

        self.current_arc_turn = arc_spec
        self._arc_center_u = center_arr
        self._arc_radius_u = float(arc_spec.radius_u)
        self._arc_start_radial_angle = math.atan2(start_point[1] - center_arr[1], start_point[0] - center_arr[0])
        self._arc_total_angle = signed_total
        self._arc_progress_angle = 0.0
        self._arc_start_heading = UAVPathGenerator._normalize_angle_0_2pi(start_heading_rad)
        self.state = "turning"

    def _finish_current_arc_segment(self) -> None:
        if self.current_arc_turn is None:
            return

        arc = self.current_arc_turn
        signed_total = self._arc_total_angle
        end_radial = self._arc_start_radial_angle + signed_total
        end_x = float(self._arc_center_u[0]) + self._arc_radius_u * math.cos(end_radial)
        end_y = float(self._arc_center_u[1]) + self._arc_radius_u * math.sin(end_radial)

        self.pos_u[0] = int(round(end_x))
        self.pos_u[1] = int(round(end_y))
        self.angle = UAVPathGenerator._normalize_angle_0_2pi(self._arc_start_heading + signed_total)

        self.is_turning = False
        self.turn_step_remain = 0
        self.turning_angle_each = 0.0
        self.current_arc_turn = None
        self._arc_center_u = None
        self._arc_start_radial_angle = 0.0
        self._arc_total_angle = 0.0
        self._arc_progress_angle = 0.0
        self._arc_start_heading = 0.0
        self._arc_radius_u = 0.0

        self.current_segment_idx += 1
        self.state = "flying_to_waypoint"

    def _advance_arc_step(self) -> bool:
        if not self.is_turning or self.current_arc_turn is None:
            return True

        remaining = self._arc_total_angle - self._arc_progress_angle
        if abs(remaining) <= _ANGLE_TOL_RAD:
            self._finish_current_arc_segment()
            return True

        delta = self.turning_angle_each
        if abs(delta) > abs(remaining):
            delta = remaining

        self._arc_progress_angle += delta
        self.turn_step_remain -= 1

        radial = self._arc_start_radial_angle + self._arc_progress_angle
        px = float(self._arc_center_u[0]) + self._arc_radius_u * math.cos(radial)
        py = float(self._arc_center_u[1]) + self._arc_radius_u * math.sin(radial)
        self.pos_u[0] = int(round(px))
        self.pos_u[1] = int(round(py))

        tangent_offset = -0.5 * math.pi if self.is_turning_clockwise else 0.5 * math.pi
        self.angle = UAVPathGenerator._normalize_angle_0_2pi(radial + tangent_offset)

        if self.turn_step_remain <= 0 or abs(self._arc_total_angle - self._arc_progress_angle) <= _ANGLE_TOL_RAD:
            self._finish_current_arc_segment()

        return True

    @staticmethod
    def _normalize_arc_turn_spec(arc_raw) -> ArcTurnSpec:
        """Normalize arc payload to ArcTurnSpec for runtime compatibility."""

        if isinstance(arc_raw, ArcTurnSpec):
            return arc_raw
        if not isinstance(arc_raw, dict):
            raise TypeError(f"arc segment payload must be ArcTurnSpec or dict, got {type(arc_raw)!r}")

        if "start_point_u" not in arc_raw:
            raise KeyError("arc segment missing required key: start_point_u")
        if "radius_u" not in arc_raw:
            raise KeyError("arc segment missing required key: radius_u")
        if "rotation_angle_deg" not in arc_raw:
            raise KeyError("arc segment missing required key: rotation_angle_deg")

        start_point = UAVPathGenerator._parse_point2("arc.start_point_u", arc_raw["start_point_u"])
        return ArcTurnSpec(
            start_point_u=start_point,
            radius_u=float(arc_raw["radius_u"]),
            is_clockwise=bool(arc_raw.get("is_clockwise", True)),
            rotation_angle_deg=float(arc_raw["rotation_angle_deg"]),
        )

    def update(self, elapsed_time_h: float) -> bool:
        if not self.is_started:
            if elapsed_time_h >= self.start_time_h:
                self.is_started = True
            else:
                return True

        if not self.segments:
            raise RuntimeError(
                "UAVController requires non-empty segments. "
                "Strip-scan mode has been removed; provide external path specs."
            )
        return self._update_custom_path()

    def _update_custom_path(self) -> bool:
        if self.current_segment_idx >= len(self.segments):
            print(f"UAV#{self.uav_id} completed all segments!")
            return False

        if self.is_turning:
            return self._advance_arc_step()

        segment = self.segments[self.current_segment_idx]
        seg_type = segment["segment_type"]
        step_u = float(self.cfg.derived.uav_step_u)

        if seg_type == "line":
            target = segment["end_point_u"]
            reached = self._move_axis_aligned_toward_point((float(target[0]), float(target[1])), step_u)
            if reached:
                self.current_segment_idx += 1
            return True

        arc = self._normalize_arc_turn_spec(segment["arc"])
        segment["arc"] = arc
        start = (float(arc.start_point_u[0]), float(arc.start_point_u[1]))

        if not self._is_waypoint_reached(start, tolerance_u=max(1000, int(step_u))):
            raise RuntimeError(
                f"UAV#{self.uav_id} arc start is not continuous with current position: "
                f"current={self.position_u}, arc_start=({int(round(start[0]))}, {int(round(start[1]))})"
            )

        self.pos_u[0] = int(round(start[0]))
        self.pos_u[1] = int(round(start[1]))

        total_abs = UAVPathGenerator._deg_to_rad_stable(float(arc.rotation_angle_deg))
        if total_abs <= _ANGLE_TOL_RAD:
            self.current_segment_idx += 1
            return True

        self.uav_turn_start(arc_spec=arc, start_heading_rad=self.angle)
        if self.is_turning:
            return self._advance_arc_step()

        self.current_segment_idx += 1
        return True


class UAVFleetController:
    """多机封装：统一初始化、统一步进、统一获取活跃位置。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.controllers: list[UAVController] = []
        self.active_flags: list[bool] = []
        self.last_step_positions_u: list[tuple[int, int]] = []
        
        # 动态重规划状态追踪
        self.pending_replans: list[tuple[int, str]] = []  # [(uav_id, boundary_type), ...]
        self.current_step_idx: int = 0
        
        self._build_controllers()

    def _build_controllers(self) -> None:
        json_file = self.cfg.uav_fleet_mode.custom_paths_json
        if not json_file:
            raise ValueError("custom_paths_json must be set")
        path_specs = UAVPathGenerator.load_custom_paths_from_json(json_file)
        if self.cfg.uav_fleet_mode.strict_path_validation:
            UAVPathGenerator.validate_path_specs(path_specs)

        overlap_count = 0
        used_x: set[int] = set()
        x_positions: dict[int, int] = {}  # uav_id -> x_pos 映射

        for spec in path_specs:
            uav = UAVController(
                self.cfg,
                uav_id=spec.uav_id,
                start_pos_u=spec.start_pos_u,
                start_time_h=spec.start_time_h,
            )
            uav.segments = spec.segments
            uav.auto_gen_type = spec.auto_gen_type
            uav.current_segment_idx = 0
            uav.state = "flying_to_waypoint"

            x = spec.start_pos_u[0]
            if x in used_x:
                overlap_count += 1
            used_x.add(x)
            x_positions[spec.uav_id] = x

            self.controllers.append(uav)
            self.active_flags.append(True)

        if overlap_count > 0:
            print(
                "[FLEET][WARN] start positions analyzed: "
                f"overlapped={overlap_count}, unique_x={len(used_x)}/{len(self.controllers)}."
            )
        
        # 计算每个UAV的负责x区间（基于排序的起始位置）
        self._assign_responsible_x_ranges(x_positions)

    def _assign_responsible_x_ranges(self, x_positions: dict[int, int]) -> None:
        """按 Figure 3 等时分区公式计算负责区间（与 pictures/3.py 一致）。"""

        n = len(self.controllers)
        if n <= 0:
            return

        e = self.cfg.environment
        m = self.cfg.motion
        s = self.cfg.simulation
        scale = self.cfg.numeric.scale

        W = float(e.area_width_km)
        L = float(e.area_height_km)
        w = float(m.uav_scan_radius_km)
        v = float(m.uav_speed_km_h)
        kappa = 1.0
        base_x = -314.0
        base_y = -323.0

        widths = np.full(n, W / n, dtype=float)
        min_width = min(2.0 * w, 0.5 * W / n)
        x_edges = np.linspace(0.0, W, n + 1)
        d_in = np.zeros(n, dtype=float)

        def _entry_offset(width_km: float) -> float:
            return min(w, 0.5 * width_km)

        def _enforce_width_constraints(widths_km: np.ndarray) -> np.ndarray:
            if n * min_width >= W:
                return np.full(n, W / n, dtype=float)

            x = np.maximum(widths_km.astype(float), min_width)
            excess = float(np.sum(x) - W)
            if excess <= 1e-9:
                return x * (W / float(np.sum(x)))

            for _ in range(20):
                free = x > (min_width + 1e-9)
                if not np.any(free):
                    return np.full(n, W / n, dtype=float)
                reducible = x[free] - min_width
                reducible_sum = float(np.sum(reducible))
                if reducible_sum <= 1e-12:
                    return np.full(n, W / n, dtype=float)
                delta = excess * reducible / reducible_sum
                x[free] -= np.minimum(reducible, delta)
                x = np.maximum(x, min_width)
                excess = float(np.sum(x) - W)
                if abs(excess) <= 1e-6:
                    break

            return x * (W / float(np.sum(x)))

        for _ in range(40):
            for i in range(n):
                seg_w = widths[i]
                entry_x = x_edges[i] + _entry_offset(float(seg_w))
                d_in[i] = math.hypot(entry_x - base_x, 0.0 - base_y)

            t_star = (float(np.mean(d_in)) + (L * W * kappa) / (n * w)) / v
            raw = (w / (L * kappa)) * (v * t_star - d_in)
            raw = np.maximum(raw, 1e-6)
            next_widths = _enforce_width_constraints(raw)

            if float(np.max(np.abs(next_widths - widths))) <= 1e-5:
                widths = next_widths
                break

            widths = next_widths
            x_edges[0] = 0.0
            x_edges[1:] = np.cumsum(widths)

        x_edges[0] = 0.0
        x_edges[1:] = np.cumsum(widths)
        x_edges[-1] = W
        x_edges = np.maximum.accumulate(x_edges)

        sorted_controllers = sorted(self.controllers, key=lambda u: u.uav_id)
        for i, uav in enumerate(sorted_controllers):
            x_min_u = int(round(float(x_edges[i]) * scale))
            x_max_u = int(round(float(x_edges[i + 1]) * scale))
            uav.replanning_metadata.assigned_x_range_u = (x_min_u, x_max_u)
            print(
                f"[FLEET] UAV#{uav.uav_id} assigned x_range=[{x_min_u}, {x_max_u}] km="
                f"[{x_min_u/scale:.3f}, {x_max_u/scale:.3f}]"
            )

    @property
    def primary_controller(self) -> UAVController:
        return self.controllers[0]

    @property
    def active_positions_u(self) -> list[tuple[int, int]]:
        return [uav.position_u for uav, active in zip(self.controllers, self.active_flags) if active]

    @property
    def scan_positions_u(self) -> list[tuple[int, int]]:
        return self.last_step_positions_u

    def update_all(self, elapsed_time_h: float) -> bool:
        any_active = False
        step_positions: list[tuple[int, int]] = []
        
        # 重置本步pending重规划列表
        self.pending_replans.clear()
        
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
            
            # 检查边界触发条件（仅在UAV活跃且启用重规划时检查）
            should_trigger, boundary_type = uav._check_boundary_trigger(self.current_step_idx)
            if should_trigger and boundary_type:
                self.pending_replans.append((uav.uav_id, boundary_type))

        self.last_step_positions_u = step_positions
        self.current_step_idx += 1
        return any_active


class UAVFleetBuilder:
    """High-level API surface intended for external scripts and agents."""

    @staticmethod
    def from_default_config(cfg) -> UAVFleetController:
        raise NotImplementedError(
            "from_default_config() has been removed. "
            "Use from_custom_json(cfg, json_filepath) or from_path_specs(cfg, path_specs)."
        )

    @staticmethod
    def from_strip_scan(cfg, override_params: dict | None = None) -> UAVFleetController:
        raise NotImplementedError(
            "from_strip_scan() has been removed. "
            "Please provide external path specs via from_custom_json()/from_path_specs()."
        )

    @staticmethod
    def from_custom_json(cfg, json_filepath: str) -> UAVFleetController:
        cfg.uav_fleet_mode.custom_paths_json = json_filepath
        return UAVFleetController(cfg)

    @staticmethod
    def from_path_specs(cfg, path_specs: list[UAVPathSpec]) -> UAVFleetController:
        if cfg.uav_fleet_mode.strict_path_validation:
            UAVPathGenerator.validate_path_specs(path_specs)

        fleet = UAVFleetController.__new__(UAVFleetController)
        fleet.cfg = cfg
        fleet.controllers = []
        fleet.active_flags = []
        fleet.last_step_positions_u = []

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
            uav.state = "flying_to_waypoint"

            fleet.controllers.append(uav)
            fleet.active_flags.append(True)
            fleet.last_step_positions_u.append(uav.position_u)

        return fleet
