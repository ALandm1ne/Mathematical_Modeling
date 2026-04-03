"""UAV 控制器：负责条带扫描状态机与转向几何。"""

import math

import numpy as np


class UAVController:
    """单机 UAV 控制逻辑（设备无关，仅做坐标与角度计算）。"""

    def __init__(self, cfg, uav_id: int = 0, start_pos_u: tuple[int, int] | None = None):
        self.cfg = cfg
        self.uav_id = uav_id

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
        if abs(den) < 1e-9:
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

    def update(self, elapsed_time_h: float) -> bool:
        """
        状态机推进一步。

        返回：
        - True: 继续扫描
        - False: 扫描完成或达到终止角
        """
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


class UAVFleetController:
    """多机封装：统一初始化、统一步进、统一获取活跃位置。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.controllers: list[UAVController] = []
        self.active_flags: list[bool] = []
        self.last_step_positions_u: list[tuple[int, int]] = []
        self._build_controllers()

    def _build_controllers(self) -> None:
        d = self.cfg.derived
        # 防止极小 spacing 被 round 成 0，导致所有 UAV 初始重叠。
        spacing_u = max(1, int(round(self.cfg.fleet.start_spacing_scan_diameters * 2 * d.uav_scan_radius_u)))
        base_x = d.uav_scan_radius_u
        y0 = 0
        clamped_count = 0
        used_x: set[int] = set()
        overlap_count = 0

        for i in range(self.cfg.fleet.uav_count):
            x = base_x + i * spacing_u
            x_clamped = max(0, min(d.area_width_u, x))
            if x_clamped != x:
                clamped_count += 1
            if x_clamped in used_x:
                overlap_count += 1
            used_x.add(x_clamped)

            x = x_clamped
            uav = UAVController(self.cfg, uav_id=i, start_pos_u=(x, y0))
            self.controllers.append(uav)
            self.active_flags.append(True)

        if clamped_count > 0 or overlap_count > 0:
            print(
                "[FLEET][WARN] start positions compressed by boundary constraints: "
                f"clamped={clamped_count}, overlapped={overlap_count}, "
                f"unique_x={len(used_x)}/{self.cfg.fleet.uav_count}."
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
            step_positions.append(uav.position_u)
            if not keep_running:
                self.active_flags[i] = False
            else:
                any_active = True

        self.last_step_positions_u = step_positions
        return any_active
