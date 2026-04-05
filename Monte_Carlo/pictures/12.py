"""Figure 12: Monte Carlo evaluation comparison across search strategies."""

from __future__ import annotations

import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


STRATEGY_ORDER = [
    "static_strip",
    "strip_markov",
    "strip_markov_dynamic",
]

STRATEGY_LABELS = {
    "static_strip": "Static Strip",
    "strip_markov": "Strip + Markov",
    "strip_markov_dynamic": "Strip + Markov + Dynamic Replanning",
}


@dataclass(frozen=True)
class Figure12Params:
    default_time_limit_h: float = 10.0
    figsize: tuple[float, float] = (15.0, 6.6)
    dpi: int = 200


def _normalize_strategy(raw: str) -> str:
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    alias = {
        "static": "static_strip",
        "static_strip": "static_strip",
        "strip": "static_strip",
        "markov": "strip_markov",
        "strip_markov": "strip_markov",
        "strip+markov": "strip_markov",
        "dynamic": "strip_markov_dynamic",
        "dp": "strip_markov_dynamic",
        "replanning": "strip_markov_dynamic",
        "strip_markov_dynamic": "strip_markov_dynamic",
    }
    return alias.get(key, key)


def load_strategy_map(map_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not map_path.exists():
        return mapping

    with map_path.open("r", newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        for row in rows:
            run_dir = str(row.get("run_dir", "")).strip()
            strategy = str(row.get("strategy", "")).strip()
            if not run_dir or not strategy:
                continue
            mapping[run_dir] = _normalize_strategy(strategy)
    return mapping


def has_dynamic_replanning_events(run_dir: Path) -> bool:
    csv_path = run_dir / "search_strategy_dynamic.csv"
    if not csv_path.exists():
        return False

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for _ in reader:
            return True
    return False


def infer_strategy(run_dir: Path, strategy_map: dict[str, str]) -> str:
    run_name = run_dir.name
    if run_name in strategy_map:
        return strategy_map[run_name]

    lowered = run_name.lower()
    if "static" in lowered:
        return "static_strip"
    if "dynamic" in lowered or "replan" in lowered or "dp" in lowered:
        return "strip_markov_dynamic"
    if "markov" in lowered:
        return "strip_markov"

    if has_dynamic_replanning_events(run_dir):
        return "strip_markov_dynamic"
    return "strip_markov"


def read_run_metrics(run_dir: Path, time_limit_h: float) -> dict[str, float | bool] | None:
    traj_path = run_dir / "uav_trajectory.csv"
    if not traj_path.exists():
        return None

    run_end_time = 0.0
    first_found_time = math.inf

    with traj_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["time_h"])
            remaining = int(row["remaining_particles"])
            if t > run_end_time:
                run_end_time = t
            if remaining <= 0 and t < first_found_time:
                first_found_time = t

    found = math.isfinite(first_found_time)
    discovery_time_h = first_found_time if found else run_end_time
    success_within_limit = bool(found and discovery_time_h <= time_limit_h)

    return {
        "found": found,
        "discovery_time_h": float(discovery_time_h),
        "success_within_limit": success_within_limit,
        "run_end_time_h": float(run_end_time),
    }


def collect_metrics(results_root: Path, time_limit_h: float, strategy_map: dict[str, str]) -> tuple[list[dict], dict[str, list[dict]]]:
    run_records: list[dict] = []
    grouped: dict[str, list[dict]] = {k: [] for k in STRATEGY_ORDER}

    run_dirs = [p for p in sorted(results_root.iterdir()) if p.is_dir()]
    for run_dir in run_dirs:
        metrics = read_run_metrics(run_dir, time_limit_h)
        if metrics is None:
            continue

        strategy = infer_strategy(run_dir, strategy_map)
        if strategy not in grouped:
            grouped[strategy] = []

        record = {
            "run_dir": run_dir.name,
            "strategy": strategy,
            **metrics,
        }
        run_records.append(record)
        grouped[strategy].append(record)

    return run_records, grouped


def aggregate(grouped: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for strategy in STRATEGY_ORDER:
        entries = grouped.get(strategy, [])
        n_runs = len(entries)
        if n_runs == 0:
            rows.append(
                {
                    "strategy": strategy,
                    "label": STRATEGY_LABELS[strategy],
                    "n_runs": 0,
                    "avg_discovery_time_h": math.nan,
                    "success_rate": math.nan,
                    "success_count": 0,
                }
            )
            continue

        times = np.array([float(e["discovery_time_h"]) for e in entries], dtype=float)
        success = np.array([bool(e["success_within_limit"]) for e in entries], dtype=bool)
        rows.append(
            {
                "strategy": strategy,
                "label": STRATEGY_LABELS[strategy],
                "n_runs": n_runs,
                "avg_discovery_time_h": float(np.mean(times)),
                "success_rate": float(np.mean(success)),
                "success_count": int(np.sum(success)),
            }
        )
    return rows


def write_metrics_csv(path: Path, aggregated: list[dict], run_records: list[dict], time_limit_h: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy_key",
            "strategy_label",
            "n_runs",
            "avg_discovery_time_h",
            "success_rate_within_limit",
            "success_count",
            "time_limit_h",
        ])
        for row in aggregated:
            writer.writerow([
                row["strategy"],
                row["label"],
                row["n_runs"],
                "" if math.isnan(row["avg_discovery_time_h"]) else f"{row['avg_discovery_time_h']:.6f}",
                "" if math.isnan(row["success_rate"]) else f"{row['success_rate']:.6f}",
                row["success_count"],
                f"{time_limit_h:.6f}",
            ])

    per_run_path = path.with_name("12_run_metrics.csv")
    with per_run_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_dir",
            "strategy_key",
            "strategy_label",
            "found",
            "discovery_time_h",
            "success_within_limit",
            "run_end_time_h",
        ])
        for rec in run_records:
            key = str(rec["strategy"])
            writer.writerow([
                rec["run_dir"],
                key,
                STRATEGY_LABELS.get(key, key),
                rec["found"],
                f"{float(rec['discovery_time_h']):.6f}",
                rec["success_within_limit"],
                f"{float(rec['run_end_time_h']):.6f}",
            ])


def draw_figure(path: Path, aggregated: list[dict], time_limit_h: float, params: Figure12Params) -> None:
    labels = [row["label"] for row in aggregated]
    n_runs = [int(row["n_runs"]) for row in aggregated]
    avg_times = [float(row["avg_discovery_time_h"]) for row in aggregated]
    success_rates = [float(row["success_rate"]) for row in aggregated]

    x = np.arange(len(labels), dtype=float)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=params.figsize, dpi=params.dpi)

    left_colors = ["#6a994e", "#1d3557", "#bc4749"]
    right_colors = ["#90be6d", "#457b9d", "#e76f51"]

    # Left panel: average discovery time.
    bars_l = []
    for i, t in enumerate(avg_times):
        if math.isnan(t):
            bar = ax_l.bar(x[i], 0.0, color="#dadada", edgecolor="#7f7f7f", hatch="//", width=0.62)[0]
            ax_l.text(x[i], 0.15, "No runs", ha="center", va="bottom", fontsize=9, color="#666666")
        else:
            bar = ax_l.bar(x[i], t, color=left_colors[i], edgecolor="#2f2f2f", width=0.62)[0]
            ax_l.text(x[i], t + 0.25, f"{t:.2f} h", ha="center", va="bottom", fontsize=9)
        bars_l.append(bar)

    ax_l.set_title("Average Discovery Time", fontsize=13)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, rotation=16, ha="right")
    ax_l.set_ylabel("Hours")
    ax_l.grid(axis="y", alpha=0.25, linestyle="--")

    # Right panel: success rate within time limit.
    bars_r = []
    for i, p in enumerate(success_rates):
        if math.isnan(p):
            bar = ax_r.bar(x[i], 0.0, color="#dadada", edgecolor="#7f7f7f", hatch="//", width=0.62)[0]
            ax_r.text(x[i], 3.0, "No runs", ha="center", va="bottom", fontsize=9, color="#666666")
        else:
            pct = 100.0 * p
            bar = ax_r.bar(x[i], pct, color=right_colors[i], edgecolor="#2f2f2f", width=0.62)[0]
            ax_r.text(x[i], pct + 1.7, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
        bars_r.append(bar)

    ax_r.set_title(f"Success Rate Within {time_limit_h:.1f} h", fontsize=13)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, rotation=16, ha="right")
    ax_r.set_ylabel("Success Rate (%)")
    ax_r.set_ylim(0.0, 105.0)
    ax_r.grid(axis="y", alpha=0.25, linestyle="--")

    # Show sample size for each strategy on both panels.
    for i, n in enumerate(n_runs):
        ax_l.text(x[i], 0.02, f"n={n}", transform=ax_l.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
        ax_r.text(x[i], 0.02, f"n={n}", transform=ax_r.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Figure 12 Monte Carlo Evaluation Comparison Across Search Strategies",
        fontsize=15,
        y=0.98,
    )
    fig.text(
        0.5,
        0.04,
        "Left: average discovery time. Right: probability of discovering target within the time limit.",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    params = Figure12Params()
    script_dir = Path(__file__).resolve().parent
    default_results_root = script_dir.parent / "results"

    results_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else default_results_root.resolve()
    output_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else (script_dir / "12.png").resolve()
    time_limit_h = float(sys.argv[3]) if len(sys.argv) > 3 else params.default_time_limit_h

    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    strategy_map = load_strategy_map(script_dir / "12_strategy_map.csv")
    run_records, grouped = collect_metrics(results_root, time_limit_h, strategy_map)
    aggregated = aggregate(grouped)

    write_metrics_csv(script_dir / "12_metrics.csv", aggregated, run_records, time_limit_h)
    draw_figure(output_path, aggregated, time_limit_h, params)

    used_runs = len(run_records)
    print(f"[Figure12] used runs: {used_runs}")
    for row in aggregated:
        label = row["label"]
        n_runs = row["n_runs"]
        if n_runs == 0:
            print(f"[Figure12] {label}: n=0")
            continue
        print(
            f"[Figure12] {label}: n={n_runs}, "
            f"avg_time={row['avg_discovery_time_h']:.3f} h, "
            f"success@{time_limit_h:.1f}h={100.0 * row['success_rate']:.2f}%"
        )

    print(f"[Figure12] metrics: {(script_dir / '12_metrics.csv').resolve()}")
    print(f"[Figure12] figure: {output_path}")


if __name__ == "__main__":
    main()
