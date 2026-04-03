"""状态追踪器：缓存关键时刻的概率密度快照（numpy）。"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Snapshot:
    """单个时刻快照。"""

    step: int
    time_h: float
    density_map: np.ndarray


class StateTracker:
    """保存关键时刻密度快照，不持有 torch 对象。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.snapshots: list[Snapshot] = []
        self._targets_h = sorted(float(t) for t in cfg.snapshot.sample_times_h)
        self._captured_targets: set[float] = set()

    def maybe_capture(self, *, step: int, time_h: float, density_map: np.ndarray) -> bool:
        """当达到尚未采样的阈值时记录快照（首次跨越触发）。"""
        if not self.cfg.snapshot.enable:
            return False

        target = None
        for t in self._targets_h:
            if t in self._captured_targets:
                continue
            if time_h >= t:
                target = t
                break

        if target is None:
            return False

        self._captured_targets.add(target)
        self.snapshots.append(
            Snapshot(
                step=int(step),
                time_h=float(time_h),
                density_map=np.asarray(density_map, dtype=np.float32).copy(),
            )
        )
        return True

    def capture_terminal(self, *, step: int, time_h: float, density_map: np.ndarray) -> bool:
        """在仿真结束时补终态快照，避免与最后一次采样重复。"""
        if not self.cfg.snapshot.enable or not self.cfg.snapshot.include_terminal_snapshot:
            return False

        if self.snapshots:
            last = self.snapshots[-1]
            if last.step == int(step) and abs(last.time_h - float(time_h)) < 1e-9:
                return False

        self.snapshots.append(
            Snapshot(
                step=int(step),
                time_h=float(time_h),
                density_map=np.asarray(density_map, dtype=np.float32).copy(),
            )
        )
        return True

    def export_npz(self, run_results_dir: str) -> str | None:
        """将快照导出为 .npz 文件（矩阵与元数据分开存储）。"""
        if not self.snapshots:
            return None

        os.makedirs(run_results_dir, exist_ok=True)
        out_path = os.path.join(run_results_dir, self.cfg.snapshot.output_filename_npz)

        density_stack = np.stack([snap.density_map for snap in self.snapshots], axis=0)
        steps = np.array([snap.step for snap in self.snapshots], dtype=np.int64)
        times_h = np.array([snap.time_h for snap in self.snapshots], dtype=np.float32)

        np.savez_compressed(
            out_path,
            density_stack=density_stack,
            steps=steps,
            times_h=times_h,
        )
        return out_path
