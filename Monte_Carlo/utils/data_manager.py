"""数据管理模块：负责结果目录创建、轨迹记录与 CSV/Parquet 导出。"""

import csv
import os

import pyarrow as pa
import pyarrow.parquet as pq

from config import build_run_timestamp


class DataLogger:
    """仿真数据记录器。"""

    def __init__(self, cfg):
        self.cfg = cfg

        # 运行开始即创建目录，避免收尾导出时出现 FileNotFound。
        self.run_timestamp = build_run_timestamp(cfg)
        self.run_results_dir = os.path.join(cfg.results_root_dir, self.run_timestamp)
        os.makedirs(self.run_results_dir, exist_ok=True)

        # 每步轨迹记录（完整字段）。
        self.uav_step_trace = {
            "step": [],
            "time_h": [],
            "uav_id": [],
            "is_active": [],
            "x_km": [],
            "y_km": [],
            "angle_deg": [],
            "is_turning": [],
            "remaining_particles": [],
            # 新增字段：路径规划相关
            "current_segment_idx": [],
            "is_on_custom_path": [],
            "auto_gen_type": [],
        }
        self.uav_traj_x_km_by_id: dict[int, list[float]] = {}
        self.uav_traj_y_km_by_id: dict[int, list[float]] = {}

    def init_uav_trace(self, uav_controller) -> None:
        """兼容单机接口：写入单架 UAV 初始位置。"""
        self.init_uav_trace_fleet([uav_controller])

    def init_uav_trace_fleet(self, controllers) -> None:
        """写入机群初始位置，确保每条轨迹从起点开始可视化。"""
        self.uav_traj_x_km_by_id.clear()
        self.uav_traj_y_km_by_id.clear()
        for uav in controllers:
            x_km, y_km = uav.position_km()
            self.uav_traj_x_km_by_id[int(uav.uav_id)] = [x_km]
            self.uav_traj_y_km_by_id[int(uav.uav_id)] = [y_km]

    def record_uav_step_trace(self, step: int, time_h: float, uav_controller, remaining_particles: int) -> None:
        """兼容单机接口：记录单架 UAV 单步轨迹。"""
        self.record_uav_step_trace_fleet(
            step=step,
            time_h=time_h,
            controllers=[uav_controller],
            active_flags=[True],
            remaining_particles=remaining_particles,
        )

    def record_uav_step_trace_fleet(
        self,
        step: int,
        time_h: float,
        controllers,
        active_flags,
        remaining_particles: int,
    ) -> None:
        """记录机群单步轨迹与状态信息（每架 UAV 一行）。"""
        for uav, is_active in zip(controllers, active_flags):
            uav_x_km, uav_y_km = uav.position_km()
            uav_id = int(uav.uav_id)

            self.uav_step_trace["step"].append(step)
            self.uav_step_trace["time_h"].append(time_h)
            self.uav_step_trace["uav_id"].append(uav_id)
            self.uav_step_trace["is_active"].append(bool(is_active))
            self.uav_step_trace["x_km"].append(uav_x_km)
            self.uav_step_trace["y_km"].append(uav_y_km)
            self.uav_step_trace["angle_deg"].append(uav.angle_deg())
            self.uav_step_trace["is_turning"].append(bool(uav.is_turning))
            self.uav_step_trace["remaining_particles"].append(int(remaining_particles))
            # 记录新增的路径规划字段
            self.uav_step_trace["current_segment_idx"].append(int(uav.current_segment_idx))
            self.uav_step_trace["is_on_custom_path"].append(bool(uav.segments and uav.auto_gen_type != "strip_scan"))
            self.uav_step_trace["auto_gen_type"].append(str(uav.auto_gen_type) if uav.auto_gen_type else "unknown")

            if uav_id not in self.uav_traj_x_km_by_id:
                self.uav_traj_x_km_by_id[uav_id] = []
                self.uav_traj_y_km_by_id[uav_id] = []
            self.uav_traj_x_km_by_id[uav_id].append(uav_x_km)
            self.uav_traj_y_km_by_id[uav_id].append(uav_y_km)

    def _get_trajectory_fieldnames(self):
        """按配置返回导出字段集合（基础字段 / 扩展字段）。"""
        base_fields = ["step", "time_h", "uav_id", "is_active", "x_km", "y_km"]
        extended_fields = ["angle_deg", "is_turning", "remaining_particles", "current_segment_idx", "is_on_custom_path", "auto_gen_type"]
        if not self.cfg.export.trajectory_include_extended:
            return base_fields
        return base_fields + extended_fields

    def export_uav_trace(self) -> None:
        """根据配置导出 UAV 轨迹到 CSV/Parquet。"""
        if not self.cfg.runtime.export_uav_trajectory:
            return
        if len(self.uav_step_trace["step"]) == 0:
            print("UAV trajectory export skipped: no recorded steps.")
            return

        fieldnames = self._get_trajectory_fieldnames()
        base_name = self.cfg.export.trajectory_output_basename
        csv_path = os.path.join(self.run_results_dir, f"{base_name}.csv")
        parquet_path = os.path.join(self.run_results_dir, f"{base_name}.parquet")

        fmt = self.cfg.export.trajectory_export_format.strip().lower()
        if fmt == "csv":
            self._write_csv(csv_path, fieldnames)
            return
        if fmt == "parquet":
            self._write_parquet(parquet_path, fieldnames)
            return
        if fmt == "both":
            self._write_parquet(parquet_path, fieldnames)
            self._write_csv(csv_path, fieldnames)
            return

        raise ValueError(
            f"Unknown trajectory_export_format={self.cfg.export.trajectory_export_format}. "
            "Use one of: 'csv', 'parquet', 'both'."
        )

    def _write_csv(self, path: str, fieldnames: list[str]) -> None:
        """写出 CSV 文件。"""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            n_rows = len(self.uav_step_trace["step"])
            for i in range(n_rows):
                row = {k: self.uav_step_trace[k][i] for k in fieldnames}
                writer.writerow(row)

        file_size = os.path.getsize(path)
        print(f"UAV trajectory exported: {path} ({n_rows} rows, {file_size} bytes)")

    def _write_parquet(self, path: str, fieldnames: list[str]) -> None:
        """写出 Parquet 文件。"""
        table_data = {k: self.uav_step_trace[k] for k in fieldnames}
        table = pa.table(table_data)
        pq.write_table(table, path, compression=self.cfg.export.trajectory_parquet_compression)

        n_rows = len(self.uav_step_trace["step"])
        file_size = os.path.getsize(path)
        print(f"UAV trajectory exported: {path} ({n_rows} rows, {file_size} bytes)")

    def export_replanning_events(self, fleet_controller) -> None:
        """
        导出所有UAV的重规划触发事件。
        用于图11的数据来源。
        """
        if not self.cfg.dynamic_replanning.enable:
            return
        
        # 收集所有重规划事件
        all_events = []
        for uav in fleet_controller.controllers:
            for event in uav.replanning_metadata.replan_events:
                event_dict = {
                    "uav_id": uav.uav_id,
                    "step_idx": event.step_idx,
                    "time_h": event.time_h,
                    "trigger_pos_x_km": event.trigger_pos_u[0] / self.cfg.numeric.scale,
                    "trigger_pos_y_km": event.trigger_pos_u[1] / self.cfg.numeric.scale,
                    "trigger_boundary": event.trigger_boundary,
                    "selected_strip_id": event.selected_strip_id,
                    "tolerance_switched": event.tolerance_switched,
                    "tolerance_best_strip_id": event.tolerance_best_strip_id,
                    "tolerance_adjacent_avg": event.tolerance_adjacent_avg,
                    "tolerance_relative_gain": event.tolerance_relative_gain,
                    "tolerance_threshold": event.tolerance_threshold,
                    "num_candidates": len(event.candidate_strips),
                    "top_3_candidates": "|".join(
                        f"{cand[0]}({cand[1]})" for cand in event.candidate_strips[:3]
                    ),
                    "new_segments_count": event.new_segments_count,
                }
                all_events.append(event_dict)
        
        if not all_events:
            print("No replanning events recorded.")
            return
        
        # 导出为 CSV
        csv_path = os.path.join(self.run_results_dir, "replanning_events.csv")
        fieldnames = [
            "uav_id", "step_idx", "time_h",
            "trigger_pos_x_km", "trigger_pos_y_km", "trigger_boundary",
            "tolerance_switched", "tolerance_best_strip_id", "tolerance_adjacent_avg",
            "tolerance_relative_gain", "tolerance_threshold",
            "selected_strip_id", "num_candidates", "top_3_candidates",
            "new_segments_count",
        ]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event_dict in all_events:
                writer.writerow(event_dict)
        
        file_size = os.path.getsize(csv_path)
        print(f"Replanning events exported: {csv_path} ({len(all_events)} events, {file_size} bytes)")

    def export_search_strategy(self, fleet_controller) -> None:
        """
        导出搜索策略对比数据（图11所需的关键数据）：
        静态顺序 vs 动态顺序。
        """
        if not self.cfg.dynamic_replanning.enable:
            return
        
        # 为每个UAV列举其所有重规划后访问的条带顺序
        search_data = []
        
        for uav in fleet_controller.controllers:
            # 构造该UAV的访问顺序
            # 简单版本：按重规划事件的时间顺序列出访问的条带
            for i, event in enumerate(uav.replanning_metadata.replan_events):
                search_data.append({
                    "uav_id": uav.uav_id,
                    "replan_sequence": i + 1,
                    "step_idx": event.step_idx,
                    "time_h": event.time_h,
                    "selected_strip_id": event.selected_strip_id,
                    "boundary_type": event.trigger_boundary,
                    "tolerance_switched": event.tolerance_switched,
                    "tolerance_best_strip_id": event.tolerance_best_strip_id,
                    "tolerance_relative_gain": event.tolerance_relative_gain,
                    "tolerance_threshold": event.tolerance_threshold,
                    "is_high_priority": len(event.candidate_strips) > 0 and event.selected_strip_id == event.candidate_strips[0][0],  # noqa
                })
        
        if not search_data:
            print("No search strategy data to export.")
            return
        
        # 导出为 CSV
        csv_path = os.path.join(self.run_results_dir, "search_strategy_dynamic.csv")
        fieldnames = [
            "uav_id", "replan_sequence", "step_idx", "time_h",
            "selected_strip_id", "boundary_type", "tolerance_switched",
            "tolerance_best_strip_id", "tolerance_relative_gain", "tolerance_threshold",
            "is_high_priority",
        ]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in search_data:
                writer.writerow(entry)
        
        file_size = os.path.getsize(csv_path)
        print(f"Search strategy exported: {csv_path} ({len(search_data)} records, {file_size} bytes)")

