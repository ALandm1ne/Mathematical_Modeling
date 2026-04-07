"""
动态路径重规划引擎：边界触发 + 概率驱动条带选择 + 90度接入路径生成。

关键职责：
1. 根据粒子密度热力图评估各条带的覆盖潜力
2. 按配置的启发式选择最优条带（粒子数最多 or 概率最高）
3. 生成从当前位置向目标条带的90度接入路径
4. 安全注入到UAV的段队列中

接入点：main.py主循环的C阶段（扫描后）
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ReplanningRequest:
    """单个重规划请求"""
    uav_id: int
    boundary_type: str  # "top" or "bottom"
    current_pos_u: tuple[int, int]
    current_heading_rad: float
    assigned_x_range_u: tuple[int, int]


class StripEvaluator:
    """条带评估器：计算各竖直条带的覆盖潜力"""
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def evaluate_strips_by_density(
        self,
        density_grid: np.ndarray,  # [n_y_bins, n_x_bins] 的粒子密度矩阵
        x_range_u: tuple[int, int],  # 负责区域的x范围
    ) -> list[tuple[int, float]]:
        """
        按当前粒子密度评估各条带的得分。
        
        参数：
        - density_grid: 粒子密度热力图
        - x_range_u: 要评估的x区间 (x_min, x_max)
        
        返回：
        - [(strip_id, score), ...] 按得分降序排列
        """
        d = self.cfg.derived
        grid_size_u = d.grid_size_u
        strip_width_u = 2 * d.uav_scan_radius_u
        x_min, x_max = x_range_u
        
        # 候选 x 采用“条带左边界 x”，允许左边界越过区域/整体边界；
        # 但条带中心线仍必须落在责任区内。
        x_min_left = x_min - d.uav_scan_radius_u
        x_max_left = x_max - d.uav_scan_radius_u
        x_min_bin = int(math.floor(x_min_left / grid_size_u))
        x_max_bin = int(math.floor(x_max_left / grid_size_u))
        window_bins = max(1, int(round(strip_width_u / grid_size_u)))
        
        strip_scores = []
        
        for left_bin in range(x_min_bin, x_max_bin + 1):
            strip_left_x_u = int(left_bin * grid_size_u)
            left_idx = max(0, int(math.floor(strip_left_x_u / grid_size_u)))
            right_idx = min(density_grid.shape[1], int(math.ceil((strip_left_x_u + strip_width_u) / grid_size_u)))
            if right_idx <= left_idx:
                continue

            strip_density = int(np.sum(density_grid[:, left_idx:right_idx]))
            strip_scores.append((strip_left_x_u, float(strip_density)))
        
        # 按得分降序排列
        strip_scores.sort(key=lambda x: x[1], reverse=True)
        return strip_scores
    
    def get_best_strips(
        self,
        strip_scores: list[tuple[int, float]],
        count: int = 5,
    ) -> list[tuple[int, float]]:
        """获取前N个得分最高的条带"""
        return strip_scores[:count]


class PathGenerator:
    """90度接入路径生成器：从当前位置到目标条带的三段式路径"""
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def generate_90degree_approach(
        self,
        current_line_x_u: int,
        target_strip_left_x_u: int,
        boundary_type: str,
    ) -> Optional[list[dict]]:
        """
        生成90度接入路径：转向 -> 直线接近 -> 再次转向。
        
        参数：
        - current_pos_u: 当前位置
        - current_heading_rad: 当前朝向（弧度）
        - target_strip_id: 目标条带编号（网格列号）
        - scan_direction: 扫描方向说明
        - max_turn_attempts: 最多转向尝试次数
        
        返回：
        - [segment_spec, ...] 或 None（若无法生成）
        """
        d = self.cfg.derived
        r = d.uav_scan_radius_u
        H = d.area_height_u

        current_line_x_u = int(current_line_x_u)
        target_strip_left_x_u = int(target_strip_left_x_u)
        current_scan_col_u = current_line_x_u
        target_scan_col_u = target_strip_left_x_u + r
        delta = target_scan_col_u - current_scan_col_u

        if abs(delta) < 2:
            return []

        moving_right = delta > 0

        # 自适应半径：优先 r；若用 r 无法到达目标条带，则缩小到可行半径。
        max_reachable_radius = max(1, abs(delta) // 2)
        turn_radius_u = int(min(r, max_reachable_radius))
        if turn_radius_u <= 0:
            return []

        if boundary_type == "top":
            trigger_y = H - r
            end_trigger_y = r
            first_arc_cw = moving_right
            second_arc_cw = moving_right
            boundary_y = H
            boundary_target_x = (
                target_scan_col_u - turn_radius_u if moving_right else target_scan_col_u + turn_radius_u
            )
        else:
            trigger_y = r
            end_trigger_y = H - r
            first_arc_cw = not moving_right
            second_arc_cw = not moving_right
            boundary_y = 0
            boundary_target_x = (
                target_scan_col_u - turn_radius_u if moving_right else target_scan_col_u + turn_radius_u
            )

        segments: list[dict] = [
            {
                "segment_type": "arc",
                "arc": {
                    "start_point_u": (current_line_x_u, trigger_y),
                    "radius_u": turn_radius_u,
                    "is_clockwise": first_arc_cw,
                    "rotation_angle_deg": 90.0,
                },
            }
        ]

        segments.append({"segment_type": "line", "end_point_u": (int(boundary_target_x), int(boundary_y))})
        segments.append(
            {
                "segment_type": "arc",
                "arc": {
                    "start_point_u": (int(boundary_target_x), int(boundary_y)),
                    "radius_u": turn_radius_u,
                    "is_clockwise": second_arc_cw,
                    "rotation_angle_deg": 90.0,
                },
            }
        )
        segments.append(
            {
                "segment_type": "line",
                "end_point_u": (int(target_scan_col_u), int(end_trigger_y)),
            }
        )

        return segments

    def generate_keep_lane_fallback(
        self,
        current_line_x_u: int,
        boundary_type: str,
    ) -> list[dict]:
        """无可行条带时的保底策略：保持当前扫描列继续纵向覆盖。"""

        d = self.cfg.derived
        r = d.uav_scan_radius_u
        H = d.area_height_u
        current_line_x_u = int(current_line_x_u)

        if boundary_type == "top":
            start_y = H - r
            end_y = r
        else:
            start_y = r
            end_y = H - r

        return [
            {
                "segment_type": "line",
                "end_point_u": (current_line_x_u, int(end_y)),
            }
        ]

    def generate_cold_start_path(
        self,
        current_pos_u: tuple[int, int],
        target_strip_left_x_u: int,
    ) -> list[dict]:
        """Generate initial path from arbitrary start position to first scanning strip."""

        d = self.cfg.derived
        r = d.uav_scan_radius_u
        top_y = d.area_height_u - r
        start_x, start_y = int(current_pos_u[0]), int(current_pos_u[1])
        target_scan_col_u = int(target_strip_left_x_u + r)

        segments: list[dict] = []
        if start_x != target_scan_col_u:
            segments.append(
                {
                    "segment_type": "line",
                    "end_point_u": (target_scan_col_u, start_y),
                }
            )

        if start_y != r:
            segments.append(
                {
                    "segment_type": "line",
                    "end_point_u": (target_scan_col_u, int(r)),
                }
            )

        if int(r) != int(top_y):
            segments.append(
                {
                    "segment_type": "line",
                    "end_point_u": (target_scan_col_u, int(top_y)),
                }
            )

        return segments


class ReplanningEngine:
    """动态重规划主引擎"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.evaluator = StripEvaluator(cfg)
        self.path_gen = PathGenerator(cfg)

    @staticmethod
    def _choose_direction_from_top_candidate(current_scan_col_u: int, candidates: list[tuple[int, float]], r_u: int) -> int:
        if not candidates:
            return 1
        top_left_x, _ = candidates[0]
        top_scan_col = int(top_left_x + r_u)
        return 1 if top_scan_col >= current_scan_col_u else -1

    @staticmethod
    def _is_feasible_strip(
        candidate_left_x_u: int,
        current_scan_col_u: int,
        direction_d: int,
        r_u: int,
        area_width_u: int,
        x_range_u: tuple[int, int],
    ) -> bool:
        t = int(candidate_left_x_u + r_u)
        if direction_d * (t - current_scan_col_u) < 2 * r_u:
            return False
        # 只约束 UAV 中心线在整体范围内，扫描半径可跨越整体边界。
        if t < 0 or t > area_width_u:
            return False
        x_min, x_max = x_range_u
        # 允许扫描圆盘覆盖到责任区外，但中心线必须留在责任区内。
        if t < x_min or t > x_max:
            return False
        return True

    def _select_best_feasible_strip(
        self,
        current_scan_col_u: int,
        candidates: list[tuple[int, float]],
        x_range_u: tuple[int, int],
        score_lookup_all: Optional[dict[int, float]] = None,
    ) -> tuple[Optional[int], int, dict]:
        d = self.cfg.derived
        r_u = d.uav_scan_radius_u
        direction = self._choose_direction_from_top_candidate(current_scan_col_u, candidates, r_u)
        adjacent_center_delta_u = 2 * r_u

        def _apply_tolerance_switch(
            best_left_x: int,
            best_score: float,
            direction_d: int,
        ) -> tuple[int, dict]:
            threshold = float(self.cfg.dynamic_replanning.best_strip_tolerance_ratio)
            info = {
                "switched": False,
                "best_left_x": int(best_left_x),
                "adjacent_avg": 0.0,
                "relative_gain": 0.0,
                "threshold": threshold,
            }
            if threshold <= 0:
                return int(best_left_x), info

            score_map = score_lookup_all if score_lookup_all is not None else {int(x): float(s) for x, s in candidates}

            def _nearest_feasible_to_target_left(target_left_x: int) -> Optional[tuple[int, float]]:
                nearest: Optional[tuple[int, float, int]] = None  # (x, score, |x-target|)
                for nx, ns in score_map.items():
                    if not self._is_feasible_strip(nx, current_scan_col_u, direction_d, r_u, d.area_width_u, x_range_u):
                        continue
                    dist = abs(int(nx) - int(target_left_x))
                    if nearest is None or dist < nearest[2] or (dist == nearest[2] and int(nx) < nearest[0]):
                        nearest = (int(nx), float(ns), int(dist))
                if nearest is None:
                    return None
                return (nearest[0], nearest[1])

            adjacent_items: list[tuple[int, float]] = []
            # “相邻条带”定义为以当前扫描中心为基准的 ±2r，对应到最近可行网格条带。
            for center_delta in (-adjacent_center_delta_u, adjacent_center_delta_u):
                target_center_x = int(current_scan_col_u + center_delta)
                target_left_x = int(target_center_x - r_u)
                item = _nearest_feasible_to_target_left(target_left_x)
                if item is None:
                    continue
                if all(existing_x != item[0] for existing_x, _ in adjacent_items):
                    adjacent_items.append(item)

            if not adjacent_items:
                return int(best_left_x), info

            adjacent_avg = float(sum(s for _, s in adjacent_items) / len(adjacent_items))
            info["adjacent_avg"] = adjacent_avg
            if adjacent_avg <= 0:
                return int(best_left_x), info

            relative_gain = float(best_score / adjacent_avg - 1.0)
            info["relative_gain"] = relative_gain
            if relative_gain >= threshold:
                return int(best_left_x), info

            # 容差内改选相邻条带：优先与当前方向一致的邻条带，否则取相邻中得分更高者。
            preferred: list[tuple[int, float]] = []
            for nx, ns in adjacent_items:
                n_col = nx + r_u
                if direction_d * (n_col - current_scan_col_u) > 0:
                    preferred.append((nx, ns))

            choices = preferred if preferred else adjacent_items
            choices.sort(key=lambda it: (-it[1], abs((it[0] + r_u) - current_scan_col_u), it[0]))
            selected_adj = int(choices[0][0])
            info["switched"] = True
            adj_detail = ", ".join(f"{ax}:{ascore:.1f}" for ax, ascore in sorted(adjacent_items, key=lambda t: t[0]))
            print(
                "[REPLANNING] tolerance-switch: "
                f"best_x={best_left_x}, best_score={best_score:.1f}, "
                f"adj_avg={adjacent_avg:.1f}, gain={relative_gain:.4f} < {threshold:.4f}, "
                f"adjacent_2r=[{adj_detail}], use_adjacent_x={selected_adj}"
            )
            return selected_adj, info

        def _pick(direction_d: int) -> tuple[Optional[int], dict]:
            feasible = [
                (left_x, score)
                for left_x, score in candidates
                if self._is_feasible_strip(left_x, current_scan_col_u, direction_d, r_u, d.area_width_u, x_range_u)
            ]
            if not feasible:
                return None, {
                    "switched": False,
                    "best_left_x": -1,
                    "adjacent_avg": 0.0,
                    "relative_gain": 0.0,
                    "threshold": float(self.cfg.dynamic_replanning.best_strip_tolerance_ratio),
                }
            feasible.sort(key=lambda it: (-it[1], abs((it[0] + r_u) - current_scan_col_u), it[0]))
            best_left_x, best_score = feasible[0]
            return _apply_tolerance_switch(int(best_left_x), float(best_score), direction_d)

        selected, info = _pick(direction)
        if selected is not None:
            return selected, direction, info

        selected, info = _pick(-direction)
        if selected is not None:
            return selected, -direction, info

        return None, direction, {
            "switched": False,
            "best_left_x": -1,
            "adjacent_avg": 0.0,
            "relative_gain": 0.0,
            "threshold": float(self.cfg.dynamic_replanning.best_strip_tolerance_ratio),
        }

    def _select_fallback_strip_left_x(
        self,
        current_scan_col_u: int,
        preferred_direction: int,
        x_range_u: tuple[int, int],
    ) -> Optional[int]:
        """保底条带：最小横向位移 2r，避免原地直接反向调头。"""

        d = self.cfg.derived
        r_u = d.uav_scan_radius_u
        x_min, x_max = x_range_u

        def _build(direction_d: int) -> Optional[int]:
            target_scan_col = current_scan_col_u + direction_d * (2 * r_u)
            # 中心线不越责任区边界；扫描覆盖允许越界到相邻责任区。
            low = max(0, x_min)
            high = min(d.area_width_u, x_max)
            target_scan_col = min(high, max(low, target_scan_col))
            if abs(target_scan_col - current_scan_col_u) < 2 * r_u:
                return None
            return int(target_scan_col - r_u)

        selected = _build(preferred_direction)
        if selected is not None:
            return selected
        return _build(-preferred_direction)
    
    def process_pending_replans(
        self,
        pending_replans: list[tuple[int, str]],  # [(uav_id, boundary_type), ...], boundary_type in {top,bottom,cold_start}
        fleet_controller,  # UAVFleetController
        particle_system,   # ParticleSystem
        current_step: int,
        elapsed_time_h: float,
    ) -> None:
        """
        处理本步所有pending的重规划请求。
        
        流程：
        1. 获取粒子密度热力图
        2. 对每个pending的UAV，评估其负责区域内的条带
        3. 选择最优条带并生成接入路径
        4. 将新段注入到UAV的段队列
        5. 记录重规划事件
        """
        if not self.cfg.dynamic_replanning.enable or not pending_replans:
            return
        
        # 获取当前粒子密度热力图
        try:
            density_grid = particle_system.get_counts_in_grids()
        except Exception as e:
            print(f"[REPLANNING][ERROR] Failed to get particle density: {e}")
            return
        
        for uav_id, boundary_type in pending_replans:
            uav = next((u for u in fleet_controller.controllers if u.uav_id == uav_id), None)
            if uav is None:
                continue

            planned_pos_u = uav.position_u
            if (not uav.is_turning) and (0 <= uav.current_segment_idx < len(uav.segments)):
                current_seg = uav.segments[uav.current_segment_idx]
                if isinstance(current_seg, dict) and current_seg.get("segment_type") == "line":
                    end = current_seg.get("end_point_u")
                    if isinstance(end, (list, tuple)) and len(end) == 2:
                        planned_pos_u = (int(end[0]), int(end[1]))
            
            # 评估该UAV负责区域内的条带
            x_range = uav.replanning_metadata.assigned_x_range_u
            r_u = self.cfg.derived.uav_scan_radius_u
            eval_left_range = (x_range[0] - r_u, x_range[1] + r_u)
            strip_scores = self.evaluator.evaluate_strips_by_density(density_grid, eval_left_range)
            
            if not strip_scores:
                print(f"[REPLANNING] UAV#{uav_id}: no candidate strips found")
                continue
            
            # 选择最优条带
            best_strips = self.evaluator.get_best_strips(strip_scores, count=10)
            current_scan_col_u = int(planned_pos_u[0])
            selected_strip_left_x_u, _direction_d, tol_info = self._select_best_feasible_strip(
                current_scan_col_u=current_scan_col_u,
                candidates=best_strips,
                x_range_u=x_range,
                score_lookup_all={int(x): float(s) for x, s in strip_scores},
            )

            if selected_strip_left_x_u is None:
                print(f"[REPLANNING] UAV#{uav_id}: no feasible strip found, fallback to keep current lane")
                if boundary_type == "cold_start":
                    fallback_strip_left_x_u = int(current_scan_col_u - r_u)
                    new_segments = self.path_gen.generate_cold_start_path(
                        current_pos_u=planned_pos_u,
                        target_strip_left_x_u=fallback_strip_left_x_u,
                    )
                else:
                    fallback_strip_left_x_u = self._select_fallback_strip_left_x(
                        current_scan_col_u=current_scan_col_u,
                        preferred_direction=_direction_d,
                        x_range_u=x_range,
                    )
                    if fallback_strip_left_x_u is not None:
                        new_segments = self.path_gen.generate_90degree_approach(
                            current_line_x_u=current_scan_col_u,
                            target_strip_left_x_u=fallback_strip_left_x_u,
                            boundary_type=boundary_type,
                        )
                    else:
                        new_segments = self.path_gen.generate_keep_lane_fallback(
                            current_line_x_u=current_scan_col_u,
                            boundary_type=boundary_type,
                        )
                if not new_segments:
                    print(f"[REPLANNING] UAV#{uav_id}: fallback path generation failed")
                    continue
                uav.inject_segments_after_current(new_segments)
                candidate_list = [(sid, int(score), i) for i, (sid, score) in enumerate(best_strips)]
                uav.record_replanning_trigger(
                    step_idx=current_step,
                    time_h=elapsed_time_h,
                    boundary_type=boundary_type,
                    candidate_strips=candidate_list,
                    selected_strip_id=(fallback_strip_left_x_u if fallback_strip_left_x_u is not None else -1),
                    new_segments_count=len(new_segments),
                    tolerance_switched=bool(tol_info.get("switched", False)),
                    tolerance_best_strip_id=int(tol_info.get("best_left_x", -1)),
                    tolerance_adjacent_avg=float(tol_info.get("adjacent_avg", 0.0)),
                    tolerance_relative_gain=float(tol_info.get("relative_gain", 0.0)),
                    tolerance_threshold=float(tol_info.get("threshold", 0.0)),
                )
                continue

            # 生成接入路径
            derived = self.cfg.derived
            best_left_x_u = int(tol_info.get("best_left_x", -1))
            selected_scan_col_u = int(selected_strip_left_x_u + derived.uav_scan_radius_u)
            best_scan_col_u = int(best_left_x_u + derived.uav_scan_radius_u) if best_left_x_u >= 0 else -1
            print(
                f"[REPLANNING] UAV#{uav_id} select: current_scan_col={current_scan_col_u}, "
                f"best_left_x={best_left_x_u}, best_scan_col={best_scan_col_u}, "
                f"selected_left_x={selected_strip_left_x_u}, selected_scan_col={selected_scan_col_u}, "
                f"switched={bool(tol_info.get('switched', False))}"
            )
            try:
                if boundary_type == "cold_start":
                    new_segments = self.path_gen.generate_cold_start_path(
                        current_pos_u=planned_pos_u,
                        target_strip_left_x_u=selected_strip_left_x_u,
                    )
                else:
                    new_segments = self.path_gen.generate_90degree_approach(
                        current_line_x_u=current_scan_col_u,
                        target_strip_left_x_u=selected_strip_left_x_u,
                        boundary_type=boundary_type,
                    )
            except Exception as e:
                print(
                    f"[REPLANNING] UAV#{uav_id}: failed to generate path to "
                    f"strip_left_x {selected_strip_left_x_u}: {e}"
                )
                continue
            
            if not new_segments:
                print(f"[REPLANNING] UAV#{uav_id}: no segments generated")
                continue
            
            # 注入到UAV段队列
            uav.inject_segments_after_current(new_segments)
            
            # 记录重规划事件
            candidate_list = [(sid, int(score), i) for i, (sid, score) in enumerate(best_strips)]
            uav.record_replanning_trigger(
                step_idx=current_step,
                time_h=elapsed_time_h,
                boundary_type=boundary_type,
                candidate_strips=candidate_list,
                selected_strip_id=selected_strip_left_x_u,
                new_segments_count=len(new_segments),
                tolerance_switched=bool(tol_info.get("switched", False)),
                tolerance_best_strip_id=int(tol_info.get("best_left_x", -1)),
                tolerance_adjacent_avg=float(tol_info.get("adjacent_avg", 0.0)),
                tolerance_relative_gain=float(tol_info.get("relative_gain", 0.0)),
                tolerance_threshold=float(tol_info.get("threshold", 0.0)),
            )
