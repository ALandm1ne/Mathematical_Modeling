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
            "x_km": [],
            "y_km": [],
            "angle_deg": [],
            "is_turning": [],
            "remaining_particles": [],
        }
        self.uav_traj_x_km = []
        self.uav_traj_y_km = []

    def init_uav_trace(self, uav_controller) -> None:
        """写入 UAV 初始位置，确保轨迹从起点开始可视化。"""
        x_km, y_km = uav_controller.position_km()
        self.uav_traj_x_km = [x_km]
        self.uav_traj_y_km = [y_km]

    def record_uav_step_trace(self, step: int, time_h: float, uav_controller, remaining_particles: int) -> None:
        """记录单步轨迹与状态信息。"""
        uav_x_km, uav_y_km = uav_controller.position_km()

        self.uav_step_trace["step"].append(step)
        self.uav_step_trace["time_h"].append(time_h)
        self.uav_step_trace["x_km"].append(uav_x_km)
        self.uav_step_trace["y_km"].append(uav_y_km)
        self.uav_step_trace["angle_deg"].append(uav_controller.angle_deg())
        self.uav_step_trace["is_turning"].append(bool(uav_controller.is_turning))
        self.uav_step_trace["remaining_particles"].append(int(remaining_particles))

        self.uav_traj_x_km.append(uav_x_km)
        self.uav_traj_y_km.append(uav_y_km)

    def _get_trajectory_fieldnames(self):
        """按配置返回导出字段集合（基础字段 / 扩展字段）。"""
        base_fields = ["step", "time_h", "x_km", "y_km"]
        if not self.cfg.export.trajectory_include_extended:
            return base_fields
        return base_fields + ["angle_deg", "is_turning", "remaining_particles"]

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
                row = {
                    "step": self.uav_step_trace["step"][i],
                    "time_h": self.uav_step_trace["time_h"][i],
                    "x_km": self.uav_step_trace["x_km"][i],
                    "y_km": self.uav_step_trace["y_km"][i],
                    "angle_deg": self.uav_step_trace["angle_deg"][i],
                    "is_turning": self.uav_step_trace["is_turning"][i],
                    "remaining_particles": self.uav_step_trace["remaining_particles"][i],
                }
                writer.writerow({k: row[k] for k in fieldnames})

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
