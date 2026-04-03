"""离线绘图工厂：地理图、概率图、统计图。"""

from __future__ import annotations

import os
from collections import defaultdict

import matplotlib
import numpy as np
from matplotlib.patches import Rectangle


class _BasePlotter:
    def __init__(self, cfg):
        self.cfg = cfg
        self._plt = None

    def _get_plt(self):
        if self._plt is None:
            backend = "QtAgg" if self.cfg.runtime.realtime_visualization else "Agg"
            matplotlib.use(backend)
            import matplotlib.pyplot as plt

            self._plt = plt
        return self._plt

    def _u_to_km(self, value_u: np.ndarray | float) -> np.ndarray:
        return np.asarray(value_u, dtype=np.float64) / float(self.cfg.numeric.scale)

    def _save_close(self, fig, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=self.cfg.plot.dpi, bbox_inches="tight")
        plt = self._get_plt()
        plt.close(fig)


class GeographicPlotter(_BasePlotter):
    """需求 1-4,7：地理底图 + 轨迹 + 密度叠加。"""

    def plot_search_area(self, output_path: str) -> None:
        plt = self._get_plt()
        fig, ax = plt.subplots(figsize=(9, 7))

        w = self.cfg.environment.area_width_km
        h = self.cfg.environment.area_height_km
        rect = Rectangle((0.0, 0.0), w, h, fill=False, linewidth=2.0, edgecolor="black")
        ax.add_patch(rect)

        ax.scatter(
            [self.cfg.plot.airport_x_km],
            [self.cfg.plot.airport_y_km],
            marker="*",
            s=120,
            color="crimson",
            label=self.cfg.plot.airport_label,
        )
        ax.annotate(
            f"{self.cfg.plot.airport_label}\n({self.cfg.plot.airport_x_km:.1f}, {self.cfg.plot.airport_y_km:.1f}) km",
            (self.cfg.plot.airport_x_km, self.cfg.plot.airport_y_km),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_title("Search Area & Wenzhou Airport")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        self._save_close(fig, output_path)

    def plot_uav_trajectories(self, output_path: str, trajectories_by_id_u: dict[int, dict[str, np.ndarray]]) -> None:
        plt = self._get_plt()
        fig, ax = plt.subplots(figsize=(10, 8))

        cmap = plt.cm.get_cmap("tab10", max(1, len(trajectories_by_id_u)))
        for i, (uav_id, traj) in enumerate(sorted(trajectories_by_id_u.items())):
            x_km = self._u_to_km(traj["x_u"])
            y_km = self._u_to_km(traj["y_u"])
            ax.plot(x_km, y_km, color=cmap(i), linewidth=1.3, label=f"UAV#{uav_id}")
            if len(x_km) > 0:
                ax.scatter([x_km[0]], [y_km[0]], color=cmap(i), s=20)

        ax.set_xlim(0, self.cfg.environment.area_width_km)
        ax.set_ylim(0, self.cfg.environment.area_height_km)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_title("UAV Trajectories")
        ax.grid(True, alpha=0.25)
        ax.legend()
        self._save_close(fig, output_path)

    def plot_density_overlay(
        self,
        output_path: str,
        density_map: np.ndarray,
        trajectories_by_id_u: dict[int, dict[str, np.ndarray]],
    ) -> None:
        plt = self._get_plt()
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.imshow(
            density_map,
            origin="lower",
            cmap="viridis",
            extent=(0, self.cfg.environment.area_width_km, 0, self.cfg.environment.area_height_km),
            aspect="auto",
            alpha=0.85,
        )

        cmap = plt.cm.get_cmap("tab10", max(1, len(trajectories_by_id_u)))
        for i, (uav_id, traj) in enumerate(sorted(trajectories_by_id_u.items())):
            x_km = self._u_to_km(traj["x_u"])
            y_km = self._u_to_km(traj["y_u"])
            ax.plot(x_km, y_km, color=cmap(i), linewidth=1.0, label=f"UAV#{uav_id}")

        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_title("Probability Density with UAV Paths")
        ax.legend(loc="upper right")
        self._save_close(fig, output_path)


class ProbabilityPlotter(_BasePlotter):
    """需求 8-11,16：概率演化图。"""

    def _plot_density(self, output_path: str, density_map: np.ndarray, title: str, vmax: float) -> None:
        plt = self._get_plt()
        fig, ax = plt.subplots(figsize=(9, 7))

        im = ax.imshow(
            density_map,
            origin="lower",
            cmap="viridis",
            extent=(0, self.cfg.environment.area_width_km, 0, self.cfg.environment.area_height_km),
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )
        fig.colorbar(im, ax=ax, label="Density")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_title(title)
        self._save_close(fig, output_path)

    def plot_initial_density(self, output_path: str, density_map: np.ndarray, vmax: float) -> None:
        self._plot_density(output_path, density_map, "Initial Probability Density", vmax)

    def plot_predicted_density(self, output_path: str, density_map: np.ndarray | None, vmax: float) -> None:
        if density_map is None:
            density_map = np.zeros((self.cfg.derived.n_y_bins, self.cfg.derived.n_x_bins), dtype=np.float32)
            title = "Predicted Probability (Markov unavailable: fallback placeholder)"
        else:
            title = "Predicted Probability Density (p_pred)"
        self._plot_density(output_path, density_map, title, vmax)

    def plot_residual_density(self, output_path: str, density_map: np.ndarray, vmax: float) -> None:
        self._plot_density(output_path, density_map, "Residual Probability After Scan", vmax)

    def plot_snapshot_grid(self, output_path: str, snapshots: list[dict], vmax: float) -> None:
        plt = self._get_plt()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes_flat = axes.flatten()

        for i in range(4):
            ax = axes_flat[i]
            if i < len(snapshots):
                density = snapshots[i]["density_map"]
                time_h = float(snapshots[i]["time_h"])
                im = ax.imshow(
                    density,
                    origin="lower",
                    cmap="viridis",
                    extent=(0, self.cfg.environment.area_width_km, 0, self.cfg.environment.area_height_km),
                    vmin=0,
                    vmax=vmax,
                    aspect="auto",
                )
                ax.text(
                    0.98,
                    0.98,
                    f"Time: {time_h:.2f} h",
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
                if i == 0:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis("off")

            ax.set_xlabel("X (km)")
            ax.set_ylabel("Y (km)")
            ax.set_title(f"Snapshot #{i + 1}")

        fig.suptitle("Probability Snapshots (2x2)")
        self._save_close(fig, output_path)


class AnalyticPlotter(_BasePlotter):
    """需求 5,6,12-15：统计与对比图。"""

    def plot_remaining_curves(self, output_path: str, runs: list[dict]) -> None:
        plt = self._get_plt()
        fig, ax = plt.subplots(figsize=(10, 7))

        for run in runs:
            steps = np.array(run["steps"], dtype=np.int64)
            remaining = np.array(run["remaining_particles"], dtype=np.int64)
            label = run["run_id"]
            ax.plot(steps, remaining, linewidth=1.1, alpha=0.9, label=label)

        ax.set_xlabel("Step")
        ax.set_ylabel("Remaining Particles")
        ax.set_title("Remaining Particles Across Runs")
        ax.grid(True, alpha=0.25)
        if len(runs) <= 10:
            ax.legend()
        self._save_close(fig, output_path)

    def plot_success_time_vs_n(self, output_path: str, runs: list[dict]) -> None:
        plt = self._get_plt()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        grouped = defaultdict(list)
        for run in runs:
            grouped[int(run["n_particles"])].append(run)

        n_values = sorted(grouped.keys())
        success_rates = []
        mean_hours = []
        for n in n_values:
            items = grouped[n]
            success = [1 if int(it["remaining_particles"][-1]) == 0 else 0 for it in items]
            hours = [float(it["time_h"][-1]) if it["time_h"] else 0.0 for it in items]
            success_rates.append(float(np.mean(success)) if success else 0.0)
            mean_hours.append(float(np.mean(hours)) if hours else 0.0)

        axes[0].plot(n_values, success_rates, marker="o")
        axes[0].set_xlabel("N (Initial Particles)")
        axes[0].set_ylabel("Success Rate")
        axes[0].set_title("Success Rate vs N")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(n_values, mean_hours, marker="s")
        axes[1].set_xlabel("N (Initial Particles)")
        axes[1].set_ylabel("Mean Time (h)")
        axes[1].set_title("Mean Search Time vs N")
        axes[1].grid(True, alpha=0.25)

        self._save_close(fig, output_path)

    def plot_uav_total_time_bar(self, output_path: str, run_rows: list[dict]) -> None:
        plt = self._get_plt()
        fig, ax = plt.subplots(figsize=(9, 6))

        per_uav = defaultdict(float)
        for row in run_rows:
            if row.get("is_active", True):
                per_uav[int(row["uav_id"])] = max(per_uav[int(row["uav_id"])] , float(row["time_h"]))

        ids = sorted(per_uav.keys())
        vals = [per_uav[i] for i in ids]
        ax.bar([str(i) for i in ids], vals, color="steelblue")
        ax.set_xlabel("UAV ID")
        ax.set_ylabel("Total Active Time (h)")
        ax.set_title("UAV Total Active Time Comparison")
        ax.grid(True, axis="y", alpha=0.25)
        self._save_close(fig, output_path)

    def plot_turning_ratio_bar(self, output_path: str, run_rows: list[dict]) -> None:
        plt = self._get_plt()
        fig, ax = plt.subplots(figsize=(9, 6))

        total = defaultdict(int)
        turning = defaultdict(int)
        for row in run_rows:
            uid = int(row["uav_id"])
            total[uid] += 1
            if bool(row.get("is_turning", False)):
                turning[uid] += 1

        ids = sorted(total.keys())
        ratios = [turning[i] / total[i] if total[i] > 0 else 0.0 for i in ids]
        ax.bar([str(i) for i in ids], ratios, color="darkorange")
        ax.set_xlabel("UAV ID")
        ax.set_ylabel("Turning Ratio")
        ax.set_title("UAV Turning Ratio")
        ax.grid(True, axis="y", alpha=0.25)
        self._save_close(fig, output_path)

    def plot_sensitivity_curve(self, output_path: str, records: list[dict], x_key: str, y_key: str, title: str) -> None:
        plt = self._get_plt()
        fig, ax = plt.subplots(figsize=(10, 7))

        xs = np.array([float(r[x_key]) for r in records], dtype=np.float64)
        ys = np.array([float(r[y_key]) for r in records], dtype=np.float64)
        order = np.argsort(xs)

        ax.plot(xs[order], ys[order], marker="o")
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        self._save_close(fig, output_path)
