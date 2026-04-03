"""离线后处理入口：扫描 results 并批量生成 16 项标准图。"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from config import build_default_config
from utils.plotter import AnalyticPlotter, GeographicPlotter, ProbabilityPlotter


def _is_run_dir(path: Path) -> bool:
    name = path.name
    return path.is_dir() and len(name) == 15 and name[8] == "_" and name.replace("_", "").isdigit()


def _scan_run_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted([p for p in results_root.iterdir() if _is_run_dir(p)])


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _read_rows_from_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "step": int(r["step"]),
                    "time_h": float(r["time_h"]),
                    "uav_id": int(r["uav_id"]),
                    "is_active": _to_bool(r["is_active"]),
                    "x_km": float(r["x_km"]),
                    "y_km": float(r["y_km"]),
                    "angle_deg": float(r.get("angle_deg", 0.0)),
                    "is_turning": _to_bool(r.get("is_turning", False)),
                    "remaining_particles": int(float(r.get("remaining_particles", 0))),
                }
            )
    return rows


def _read_rows_from_parquet(path: Path) -> list[dict]:
    table = pq.read_table(path)
    data = table.to_pydict()
    n = len(data["step"]) if "step" in data else 0
    rows = []
    for i in range(n):
        rows.append(
            {
                "step": int(data["step"][i]),
                "time_h": float(data["time_h"][i]),
                "uav_id": int(data["uav_id"][i]),
                "is_active": _to_bool(data["is_active"][i]),
                "x_km": float(data["x_km"][i]),
                "y_km": float(data["y_km"][i]),
                "angle_deg": float(data.get("angle_deg", [0.0] * n)[i]),
                "is_turning": _to_bool(data.get("is_turning", [False] * n)[i]),
                "remaining_particles": int(data.get("remaining_particles", [0] * n)[i]),
            }
        )
    return rows


def _load_trajectory_rows(run_dir: Path) -> list[dict]:
    parquet_path = run_dir / "uav_trajectory.parquet"
    csv_path = run_dir / "uav_trajectory.csv"
    if parquet_path.exists():
        return _read_rows_from_parquet(parquet_path)
    if csv_path.exists():
        return _read_rows_from_csv(csv_path)
    return []


def _build_trajectories_u(rows: list[dict], scale: int) -> dict[int, dict[str, np.ndarray]]:
    grouped: dict[int, dict[str, list]] = {}
    for row in rows:
        uid = int(row["uav_id"])
        if uid not in grouped:
            grouped[uid] = {"x_u": [], "y_u": []}
        # 输入统一转为 units(u)，绘图器内部再做 u->km。
        grouped[uid]["x_u"].append(int(round(float(row["x_km"]) * scale)))
        grouped[uid]["y_u"].append(int(round(float(row["y_km"]) * scale)))

    out: dict[int, dict[str, np.ndarray]] = {}
    for uid, xy in grouped.items():
        out[uid] = {
            "x_u": np.asarray(xy["x_u"], dtype=np.int64),
            "y_u": np.asarray(xy["y_u"], dtype=np.int64),
        }
    return out


def _load_snapshots(run_dir: Path) -> list[dict]:
    npz_path = run_dir / "state_snapshots.npz"
    if not npz_path.exists():
        return []

    data = np.load(npz_path)
    times_h = data.get("times_h", np.array([], dtype=np.float32))
    steps = data.get("steps", np.array([], dtype=np.int64))
    density_stack = data.get("density_stack", np.array([], dtype=np.float32))

    snapshots = []
    n = min(len(times_h), len(steps), int(density_stack.shape[0]) if density_stack.ndim >= 1 else 0)
    for i in range(n):
        snapshots.append(
            {
                "step": int(steps[i]) if i < len(steps) else 0,
                "time_h": float(times_h[i]),
                "density_map": np.asarray(density_stack[i], dtype=np.float32),
            }
        )
    return snapshots


def _build_run_stats(run_id: str, rows: list[dict], default_n: int) -> dict:
    if not rows:
        return {
            "run_id": run_id,
            "steps": [],
            "time_h": [],
            "remaining_particles": [],
            "n_particles": default_n,
        }

    # 聚合到每 step（多 UAV 行中取同一 step 的 remaining 与 time）。
    by_step = {}
    for r in rows:
        step = int(r["step"])
        if step not in by_step:
            by_step[step] = {
                "time_h": float(r["time_h"]),
                "remaining_particles": int(r["remaining_particles"]),
            }

    steps_sorted = sorted(by_step.keys())
    return {
        "run_id": run_id,
        "steps": steps_sorted,
        "time_h": [by_step[s]["time_h"] for s in steps_sorted],
        "remaining_particles": [by_step[s]["remaining_particles"] for s in steps_sorted],
        "n_particles": default_n,
    }


def _make_16_figures_for_run(cfg, run_dir: Path, rows: list[dict], run_stats: dict) -> None:
    plots_dir = run_dir / cfg.plot.output_subdir_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    geo = GeographicPlotter(cfg)
    prob = ProbabilityPlotter(cfg)
    ana = AnalyticPlotter(cfg)

    trajectories_u = _build_trajectories_u(rows, cfg.numeric.scale)
    snapshots = _load_snapshots(run_dir)

    max_density = 1.0
    if snapshots:
        max_density = max(float(np.max(s["density_map"])) for s in snapshots)
        max_density = max(max_density, 1.0)

    initial_density = snapshots[0]["density_map"] if snapshots else np.zeros((cfg.derived.n_y_bins, cfg.derived.n_x_bins), dtype=np.float32)
    residual_density = snapshots[-1]["density_map"] if snapshots else initial_density

    # 1-4,7
    geo.plot_search_area(str(plots_dir / "fig01_search_area.png"))
    geo.plot_search_area(str(plots_dir / "fig02_search_area_airport.png"))
    geo.plot_uav_trajectories(str(plots_dir / "fig03_uav_trajectories.png"), trajectories_u)
    geo.plot_uav_trajectories(str(plots_dir / "fig04_uav_trajectories_colored.png"), trajectories_u)
    geo.plot_density_overlay(str(plots_dir / "fig07_density_with_paths.png"), residual_density, trajectories_u)

    # 8-11,16
    prob.plot_initial_density(str(plots_dir / "fig08_initial_probability.png"), initial_density, max_density)
    prob.plot_predicted_density(str(plots_dir / "fig09_predicted_probability.png"), None, max_density)
    prob.plot_residual_density(str(plots_dir / "fig10_residual_probability.png"), residual_density, max_density)
    prob.plot_snapshot_grid(str(plots_dir / "fig16_snapshot_grid.png"), snapshots[:4], max_density)

    # 5,6,12-15（单 run 版本用于占位 + 局部分析）
    ana.plot_remaining_curves(str(plots_dir / "fig05_remaining_curve_single_run.png"), [run_stats])
    ana.plot_uav_total_time_bar(str(plots_dir / "fig06_uav_total_time.png"), rows)
    ana.plot_turning_ratio_bar(str(plots_dir / "fig12_uav_turning_ratio.png"), rows)

    # 单 run 的参数图使用 run 本身统计构建可复现实例。
    if run_stats["steps"]:
        records = [
            {"x": i + 1, "final_step": float(run_stats["steps"][-1]), "final_time_h": float(run_stats["time_h"][-1])}
            for i in range(5)
        ]
        ana.plot_sensitivity_curve(
            str(plots_dir / "fig13_success_time_vs_n_proxy.png"),
            [{"n_particles": run_stats["n_particles"], "value": run_stats["time_h"][-1]}],
            "n_particles",
            "value",
            "Proxy: Time vs N (single run)",
        )
        ana.plot_sensitivity_curve(
            str(plots_dir / "fig14_final_step_proxy.png"),
            records,
            "x",
            "final_step",
            "Proxy: Final Step Trend",
        )
        ana.plot_sensitivity_curve(
            str(plots_dir / "fig15_sensitivity_proxy.png"),
            records,
            "x",
            "final_time_h",
            "Proxy: Sensitivity Curve",
        )

    # 补齐第11张（概率横截面对比）
    if residual_density.size > 0:
        mid_row = residual_density[residual_density.shape[0] // 2, :]
        x_u = np.arange(mid_row.shape[0]) * cfg.derived.grid_size_u
        x_km = x_u / float(cfg.numeric.scale)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x_km, mid_row)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Density")
        ax.set_title("fig11 Cross-section of Residual Density")
        fig.savefig(plots_dir / "fig11_density_cross_section.png", dpi=cfg.plot.dpi, bbox_inches="tight")
        plt.close(fig)


def _make_cross_run_summary(cfg, results_root: Path, run_stats_all: list[dict]) -> None:
    if not run_stats_all:
        return

    summary_dir = results_root / cfg.analysis.summary_subdir_name
    summary_dir.mkdir(parents=True, exist_ok=True)

    ana = AnalyticPlotter(cfg)
    ana.plot_remaining_curves(str(summary_dir / "summary_remaining_curves.png"), run_stats_all)
    ana.plot_success_time_vs_n(str(summary_dir / "summary_success_time_vs_n.png"), run_stats_all)

    # 灵敏度示例：横轴使用 run 索引。
    sens_records = []
    for i, st in enumerate(run_stats_all):
        if not st["steps"]:
            continue
        sens_records.append(
            {
                "run_index": i,
                "final_step": st["steps"][-1],
                "final_time_h": st["time_h"][-1],
            }
        )

    if sens_records:
        ana.plot_sensitivity_curve(
            str(summary_dir / "summary_sensitivity_final_step.png"),
            sens_records,
            "run_index",
            "final_step",
            "Sensitivity: Final Step Across Runs",
        )
        ana.plot_sensitivity_curve(
            str(summary_dir / "summary_sensitivity_final_time.png"),
            sens_records,
            "run_index",
            "final_time_h",
            "Sensitivity: Final Time Across Runs",
        )


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = build_default_config(script_dir=script_dir, require_cuda_override=False)

    results_root = Path(cfg.results_root_dir)
    run_dirs = _scan_run_dirs(results_root)
    if not run_dirs:
        print(f"No run directories found in: {results_root}")
        return

    run_stats_all = []
    for run_dir in run_dirs:
        rows = _load_trajectory_rows(run_dir)
        if not rows:
            print(f"Skip {run_dir.name}: no trajectory file")
            continue

        run_stats = _build_run_stats(run_dir.name, rows, cfg.simulation.n_particles)
        run_stats_all.append(run_stats)
        _make_16_figures_for_run(cfg, run_dir, rows, run_stats)
        print(f"Generated figures for run: {run_dir.name}")

    _make_cross_run_summary(cfg, results_root, run_stats_all)
    print("Analysis completed.")


if __name__ == "__main__":
    main()
