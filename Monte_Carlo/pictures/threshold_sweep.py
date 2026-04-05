from __future__ import annotations

import csv
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

MONTE_CARLO_DIR = Path(__file__).resolve().parent.parent
if str(MONTE_CARLO_DIR) not in sys.path:
    sys.path.insert(0, str(MONTE_CARLO_DIR))

from config import build_default_config
from core.simulation_gpu import ParticleSystem
from core.uav_controller import UAVFleetBuilder


THRESHOLD_LEVELS: tuple[float, float, float] = (0.90, 0.95, 0.99)


@dataclass(frozen=True)
class SweepSpec:
    figure_id: str
    x_label: str
    x_unit: str
    x_values: tuple[float, ...]
    fixed_scan_radius_km: float | None = None
    fixed_uav_speed_km_h: float | None = None
    fixed_target_speed_km_h: float | None = None


@dataclass(frozen=True)
class SweepRunConfig:
    runs_per_setting: int = 5
    timeout_s: float = 1800.0
    n_particles: int = 10_000_000
    dt_h: float = 0.01
    safety_max_steps: int = 2_000_000
    seed_base: int = 20260406


def _validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite, got {value}")


def validate_run_config(cfg: SweepRunConfig) -> None:
    if cfg.runs_per_setting <= 0:
        raise ValueError("runs_per_setting must be > 0")
    if cfg.n_particles <= 0:
        raise ValueError("n_particles must be > 0")
    if cfg.safety_max_steps <= 0:
        raise ValueError("safety_max_steps must be > 0")
    _validate_positive("timeout_s", cfg.timeout_s)
    _validate_positive("dt_h", cfg.dt_h)


def validate_sweep_spec(spec: SweepSpec) -> None:
    if not spec.figure_id:
        raise ValueError("figure_id must not be empty")
    if not spec.x_values:
        raise ValueError("x_values must not be empty")
    for x in spec.x_values:
        _validate_positive("x value", float(x))
    if spec.fixed_scan_radius_km is not None:
        _validate_positive("fixed_scan_radius_km", spec.fixed_scan_radius_km)
    if spec.fixed_uav_speed_km_h is not None:
        _validate_positive("fixed_uav_speed_km_h", spec.fixed_uav_speed_km_h)
    if spec.fixed_target_speed_km_h is not None:
        _validate_positive("fixed_target_speed_km_h", spec.fixed_target_speed_km_h)


def _build_one_run(
    *,
    uav_speed_km_h: float,
    scan_radius_km: float,
    target_speed_km_h: float,
    timeout_s: float,
    n_particles: int,
    dt_h: float,
    safety_max_steps: int,
    random_seed: int,
) -> dict[str, object]:
    _validate_positive("uav_speed_km_h", uav_speed_km_h)
    _validate_positive("scan_radius_km", scan_radius_km)
    _validate_positive("target_speed_km_h", target_speed_km_h)

    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed) % (2**32 - 1))

    sim_cfg = build_default_config(script_dir=str(MONTE_CARLO_DIR), require_cuda_override=False)
    sim_cfg.dynamic_replanning.enable = False
    sim_cfg.runtime.realtime_visualization = False
    sim_cfg.runtime.export_simulation_video = False
    sim_cfg.runtime.export_uav_trajectory = False
    sim_cfg.runtime.api_demo_enable = False

    sim_cfg.simulation.n_particles = int(n_particles)
    sim_cfg.simulation.dt_h = float(dt_h)
    sim_cfg.simulation.max_steps = int(max(1_000_000, safety_max_steps))

    sim_cfg.motion.uav_speed_km_h = float(uav_speed_km_h)
    sim_cfg.motion.uav_scan_radius_km = float(scan_radius_km)
    sim_cfg.motion.target_speed_km_h = float(target_speed_km_h)

    sim_cfg.recompute_derived()
    sim_cfg.validate()

    path_file = MONTE_CARLO_DIR / "config_templates" / "uav_paths_n1.json"
    if not path_file.exists():
        raise FileNotFoundError(f"Single-UAV template not found: {path_file}")

    particle_system = ParticleSystem(sim_cfg)
    fleet = UAVFleetBuilder.from_custom_json(sim_cfg, str(path_file))
    if len(fleet.controllers) != 1:
        raise RuntimeError(f"Expected single UAV, got {len(fleet.controllers)} from {path_file}")

    initial_particles = int(sim_cfg.simulation.n_particles)
    threshold_times = {0.90: math.nan, 0.95: math.nan, 0.99: math.nan}

    elapsed_h = 0.0
    step_idx = 0
    wall_start = time.perf_counter()

    final_status = "runtime_error"
    fail_reason = "unknown"
    final_remaining = particle_system.active_count

    while True:
        wall_elapsed = time.perf_counter() - wall_start
        if wall_elapsed > timeout_s:
            final_status = "failed"
            fail_reason = "timeout"
            break
        if step_idx > safety_max_steps:
            final_status = "failed"
            fail_reason = "safety_max_steps"
            break
        if not fleet.active_positions_u:
            final_status = "failed"
            fail_reason = "no_active_uav"
            break

        particle_system.update_particles()
        any_uav_active = fleet.update_all(elapsed_h)

        remaining = particle_system.active_count
        for pos_u in fleet.scan_positions_u:
            remaining = particle_system.remove_scanned_particles(
                pos_u,
                sim_cfg.motion.uav_detection_probability,
            )

        final_remaining = int(remaining)
        if final_remaining < 0 or final_remaining > initial_particles:
            final_status = "failed"
            fail_reason = "invalid_remaining_particles"
            break

        elimination_ratio = 1.0 - (float(final_remaining) / float(initial_particles))
        for level in THRESHOLD_LEVELS:
            if math.isnan(threshold_times[level]) and elimination_ratio >= level:
                threshold_times[level] = float(elapsed_h)

        if elimination_ratio >= 0.99:
            final_status = "success_99"
            fail_reason = ""
            break

        if not any_uav_active:
            final_status = "failed"
            fail_reason = "uav_inactive_before_99"
            break

        elapsed_h += sim_cfg.simulation.dt_h
        step_idx += 1

    return {
        "t90_h": threshold_times[0.90],
        "t95_h": threshold_times[0.95],
        "t99_h": threshold_times[0.99],
        "status": final_status,
        "fail_reason": fail_reason,
        "steps": int(step_idx),
        "wall_time_s": float(time.perf_counter() - wall_start),
        "final_remaining": int(final_remaining),
    }


def _param_triplet(spec: SweepSpec, x_value: float) -> tuple[float, float, float]:
    if spec.figure_id == "16":
        return float(x_value), float(spec.fixed_scan_radius_km), float(spec.fixed_target_speed_km_h)
    if spec.figure_id == "17":
        return float(spec.fixed_uav_speed_km_h), float(x_value), float(spec.fixed_target_speed_km_h)
    if spec.figure_id == "18":
        return float(spec.fixed_uav_speed_km_h), float(spec.fixed_scan_radius_km), float(x_value)
    raise ValueError(f"Unknown figure_id: {spec.figure_id}")


def run_threshold_sweep(spec: SweepSpec, run_cfg: SweepRunConfig) -> list[dict[str, object]]:
    validate_sweep_spec(spec)
    validate_run_config(run_cfg)

    rows: list[dict[str, object]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for x_idx, x_value in enumerate(spec.x_values):
        for repeat_id in range(1, run_cfg.runs_per_setting + 1):
            run_id = f"{timestamp}_{spec.figure_id}_{x_idx:03d}_{repeat_id:02d}"
            seed = run_cfg.seed_base + x_idx * 1000 + repeat_id
            uav_speed, scan_radius, target_speed = _param_triplet(spec, float(x_value))

            try:
                result = _build_one_run(
                    uav_speed_km_h=uav_speed,
                    scan_radius_km=scan_radius,
                    target_speed_km_h=target_speed,
                    timeout_s=run_cfg.timeout_s,
                    n_particles=run_cfg.n_particles,
                    dt_h=run_cfg.dt_h,
                    safety_max_steps=run_cfg.safety_max_steps,
                    random_seed=seed,
                )
            except Exception as exc:
                result = {
                    "t90_h": math.nan,
                    "t95_h": math.nan,
                    "t99_h": math.nan,
                    "status": "failed",
                    "fail_reason": f"exception:{type(exc).__name__}",
                    "steps": 0,
                    "wall_time_s": 0.0,
                    "final_remaining": -1,
                }

            rows.append(
                {
                    "run_id": run_id,
                    "figure_id": spec.figure_id,
                    "x_value": float(x_value),
                    "x_label": spec.x_label,
                    "x_unit": spec.x_unit,
                    "repeat_id": repeat_id,
                    "uav_speed_km_h": uav_speed,
                    "scan_radius_km": scan_radius,
                    "turn_radius_km": scan_radius,
                    "target_speed_km_h": target_speed,
                    **result,
                }
            )
            print(
                f"[Figure {spec.figure_id}] x={x_value:.3f} run={repeat_id}/{run_cfg.runs_per_setting} "
                f"status={result['status']} t99={result['t99_h']}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return rows


def _nan_stats(values: Iterable[float]) -> tuple[float, float, float]:
    arr = np.array([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    avg = float(np.mean(arr))
    if arr.size > 1:
        std = float(np.std(arr, ddof=1))
        ua = float(std / math.sqrt(arr.size))
    else:
        std = 0.0
        ua = 0.0
    return avg, std, ua


def summarize_runs(spec: SweepSpec, rows: list[dict[str, object]], runs_per_setting: int) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []

    for x in spec.x_values:
        x_rows = [r for r in rows if float(r["x_value"]) == float(x)]
        fail_counter = Counter(str(r["fail_reason"]) for r in x_rows if str(r["fail_reason"]))
        fail_desc = "|".join(f"{k}:{v}" for k, v in sorted(fail_counter.items()))

        t90_vals = [float(r["t90_h"]) for r in x_rows]
        t95_vals = [float(r["t95_h"]) for r in x_rows]
        t99_vals = [float(r["t99_h"]) for r in x_rows]

        t90_avg, t90_std, t90_ua = _nan_stats(t90_vals)
        t95_avg, t95_std, t95_ua = _nan_stats(t95_vals)
        t99_avg, t99_std, t99_ua = _nan_stats(t99_vals)

        reach90 = float(sum(math.isfinite(v) for v in t90_vals) / runs_per_setting)
        reach95 = float(sum(math.isfinite(v) for v in t95_vals) / runs_per_setting)
        reach99 = float(sum(math.isfinite(v) for v in t99_vals) / runs_per_setting)
        valid_runs = int(sum(1 for r in x_rows if str(r["status"]) == "success_99"))

        summary_rows.append(
            {
                "figure_id": spec.figure_id,
                "x_value": float(x),
                "x_label": spec.x_label,
                "x_unit": spec.x_unit,
                "runs_per_setting": int(runs_per_setting),
                "valid_runs": valid_runs,
                "reach_rate_90": reach90,
                "t90_avg_h": t90_avg,
                "t90_std_h": t90_std,
                "t90_uA_h": t90_ua,
                "reach_rate_95": reach95,
                "t95_avg_h": t95_avg,
                "t95_std_h": t95_std,
                "t95_uA_h": t95_ua,
                "reach_rate_99": reach99,
                "t99_avg_h": t99_avg,
                "t99_std_h": t99_std,
                "t99_uA_h": t99_ua,
                "fail_reason_counts": fail_desc,
            }
        )

    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def set_paper_plot_style() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


def draw_threshold_figure(
    *,
    spec: SweepSpec,
    summary_rows: list[dict[str, object]],
    output_png: Path,
    output_pdf: Path,
    title: str,
    subtitle: str,
) -> None:
    set_paper_plot_style()

    x = np.array([float(r["x_value"]) for r in summary_rows], dtype=float)
    t90 = np.array([float(r["t90_avg_h"]) for r in summary_rows], dtype=float)
    t95 = np.array([float(r["t95_avg_h"]) for r in summary_rows], dtype=float)
    t99 = np.array([float(r["t99_avg_h"]) for r in summary_rows], dtype=float)

    e90 = np.array([float(r["t90_uA_h"]) for r in summary_rows], dtype=float)
    e95 = np.array([float(r["t95_uA_h"]) for r in summary_rows], dtype=float)
    e99 = np.array([float(r["t99_uA_h"]) for r in summary_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=300)

    series = [
        (t90, e90, "T90 (h)", "o", "#1f77b4"),
        (t95, e95, "T95 (h)", "s", "#2ca02c"),
        (t99, e99, "T99 (h)", "^", "#d62728"),
    ]

    for y, yerr, label, marker, color in series:
        mask = np.isfinite(y)
        ax.errorbar(
            x[mask],
            y[mask],
            yerr=yerr[mask],
            marker=marker,
            markersize=4.5,
            linewidth=1.4,
            elinewidth=1.0,
            capsize=2.5,
            color=color,
            label=label,
        )

    ax.set_title(title, fontsize=11.5, pad=8)
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=9)
    ax.set_xlabel(f"{spec.x_label} ({spec.x_unit})", fontsize=10)
    ax.set_ylabel("Completion Time (h)", fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.30)
    ax.legend(loc="best", fontsize=8.8, frameon=True)

    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
