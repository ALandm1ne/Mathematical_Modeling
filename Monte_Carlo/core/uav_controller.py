"""UAV 控制器：负责外置路径驱动下的分段运动学。

Agent-Oriented API Notes:
- This module exposes stable external APIs via UAVFleetBuilder.
- Data contracts are represented by ArcTurnSpec, SegmentSpec, UAVPathSpec.
- All coordinates are in integer units (u), not km.
"""

import json
import math
from dataclasses import dataclass
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
]

_POINT_TOL_U = 1e-6
_ANGLE_TOL_RAD = 1e-12

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
                    if math.hypot(dx, dy) > _POINT_TOL_U:
                        current_heading = math.atan2(dy, dx)
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
                if len(arc.start_point_u) != 2:
                    raise ValueError(
                        f"path_specs[{i}].segments[{j}].arc.start_point_u must contain 2 values"
                    )

                start = UAVPathGenerator._parse_point2(
                    f"path_specs[{i}].segments[{j}].arc.start_point_u",
                    arc.start_point_u,
                )
                if UAVPathGenerator._distance_u(start, current_point) > 1e-3:
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
            reached = self._move_toward_point((float(target[0]), float(target[1])), step_u)
            if reached:
                self.current_segment_idx += 1
            return True

        arc = segment["arc"]
        start = (float(arc.start_point_u[0]), float(arc.start_point_u[1]))

        if not self._is_waypoint_reached(start, tolerance_u=max(1000, int(step_u))):
            self._move_toward_point(start, step_u)
            return True

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

            self.controllers.append(uav)
            self.active_flags.append(True)

        if overlap_count > 0:
            print(
                "[FLEET][WARN] start positions analyzed: "
                f"overlapped={overlap_count}, unique_x={len(used_x)}/{len(self.controllers)}."
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
