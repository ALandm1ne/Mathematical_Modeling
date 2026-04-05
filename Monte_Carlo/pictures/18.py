from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from threshold_sweep import (
    SweepRunConfig,
    SweepSpec,
    draw_threshold_figure,
    run_threshold_sweep,
    summarize_runs,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 18: completion times vs target speed")
    parser.add_argument("--runs", type=int, default=5, help="Runs per target-speed value")
    parser.add_argument("--timeout-s", type=float, default=1800.0, help="Wall-time timeout per run (seconds)")
    parser.add_argument("--n-particles", type=int, default=10_000_000, help="Particle count per run")
    parser.add_argument("--dt-h", type=float, default=0.01, help="Simulation time step in hours")
    parser.add_argument("--safety-max-steps", type=int, default=2_000_000, help="Safety cap for simulation loop")
    parser.add_argument("--seed-base", type=int, default=20260418, help="Base random seed")

    parser.add_argument("--target-min", type=float, default=10.0, help="Minimum target speed (km/h)")
    parser.add_argument("--target-max", type=float, default=50.0, help="Maximum target speed (km/h)")
    parser.add_argument("--target-step", type=float, default=5.0, help="Step of target speed (km/h)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_values = tuple(
        float(v)
        for v in np.arange(args.target_min, args.target_max + 0.5 * args.target_step, args.target_step)
    )

    spec = SweepSpec(
        figure_id="18",
        x_label="Target Speed",
        x_unit="km/h",
        x_values=target_values,
        fixed_scan_radius_km=10.0,
        fixed_uav_speed_km_h=150.0,
    )
    run_cfg = SweepRunConfig(
        runs_per_setting=int(args.runs),
        timeout_s=float(args.timeout_s),
        n_particles=int(args.n_particles),
        dt_h=float(args.dt_h),
        safety_max_steps=int(args.safety_max_steps),
        seed_base=int(args.seed_base),
    )

    rows = run_threshold_sweep(spec, run_cfg)
    summary = summarize_runs(spec, rows, runs_per_setting=run_cfg.runs_per_setting)

    script_dir = Path(__file__).resolve().parent
    runs_path = script_dir / "18_runs.csv"
    summary_path = script_dir / "18_summary.csv"

    run_fields = [
        "run_id", "figure_id", "x_value", "x_label", "x_unit", "repeat_id",
        "uav_speed_km_h", "scan_radius_km", "turn_radius_km", "target_speed_km_h",
        "t90_h", "t95_h", "t99_h", "status", "fail_reason", "steps", "wall_time_s", "final_remaining",
    ]
    summary_fields = [
        "figure_id", "x_value", "x_label", "x_unit", "runs_per_setting", "valid_runs",
        "reach_rate_90", "t90_avg_h", "t90_std_h", "t90_uA_h",
        "reach_rate_95", "t95_avg_h", "t95_std_h", "t95_uA_h",
        "reach_rate_99", "t99_avg_h", "t99_std_h", "t99_uA_h",
        "fail_reason_counts",
    ]

    write_csv(runs_path, rows, run_fields)
    write_csv(summary_path, summary, summary_fields)

    draw_threshold_figure(
        spec=spec,
        summary_rows=summary,
        output_png=script_dir / "18.png",
        output_pdf=script_dir / "18.pdf",
        title="Figure 18 Completion Time vs Target Speed (Single UAV, Non-Dynamic)",
        subtitle="Fixed scan radius = 10 km, fixed UAV speed = 150 km/h; error bars show ±uA",
    )

    print(f"[Figure 18] runs CSV: {runs_path}")
    print(f"[Figure 18] summary CSV: {summary_path}")
    print(f"[Figure 18] figure PNG: {script_dir / '18.png'}")
    print(f"[Figure 18] figure PDF: {script_dir / '18.pdf'}")


if __name__ == "__main__":
    main()
