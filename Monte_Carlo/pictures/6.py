"""Figure 6: Per-UAV task-time balance comparison across partition strategies."""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Figure6Params:
	x_min: float = 0.0
	x_max: float = 306.0
	y_min: float = 0.0
	y_max: float = 444.0
	base_x: float = -314.0
	base_y: float = -323.0
	uav_speed_km_h: float = 150.0
	scan_radius_km: float = 20.0
	kappa: float = 1.0
	compare_n: int = 8

	@property
	def area_width(self) -> float:
		return self.x_max - self.x_min

	@property
	def area_length(self) -> float:
		return self.y_max - self.y_min


def _entry_offset(width: float, w: float) -> float:
	return min(w, 0.5 * width)


def _enforce_width_constraints(widths: np.ndarray, total_width: float, min_width: float) -> np.ndarray:
	n = len(widths)
	if n == 0:
		return widths
	if n * min_width >= total_width:
		return np.full(n, total_width / n, dtype=float)

	x = np.maximum(widths.astype(float), min_width)
	excess = float(np.sum(x) - total_width)
	if excess <= 1e-9:
		scale = total_width / float(np.sum(x))
		return x * scale

	for _ in range(20):
		free = x > (min_width + 1e-9)
		if not np.any(free):
			return np.full(n, total_width / n, dtype=float)

		reducible = x[free] - min_width
		reducible_sum = float(np.sum(reducible))
		if reducible_sum <= 1e-12:
			return np.full(n, total_width / n, dtype=float)

		delta = excess * reducible / reducible_sum
		x[free] -= np.minimum(reducible, delta)
		x = np.maximum(x, min_width)
		excess = float(np.sum(x) - total_width)
		if abs(excess) <= 1e-6:
			break

	x *= total_width / float(np.sum(x))
	return x


def solve_equal_time_widths(params: Figure6Params, n_uav: int, max_iter: int = 40) -> np.ndarray:
	w = params.scan_radius_km
	v = params.uav_speed_km_h
	L = params.area_length
	W = params.area_width
	kappa = params.kappa

	widths = np.full(n_uav, W / n_uav, dtype=float)
	min_width = min(2.0 * w, 0.5 * W / n_uav)
	x_edges = np.linspace(params.x_min, params.x_max, n_uav + 1)
	d_in = np.zeros(n_uav, dtype=float)

	for _ in range(max_iter):
		for i in range(n_uav):
			seg_w = widths[i]
			entry_x = x_edges[i] + _entry_offset(seg_w, w)
			d_in[i] = math.hypot(entry_x - params.base_x, params.y_min - params.base_y)

		t_star = (np.mean(d_in) + (L * W * kappa) / (n_uav * w)) / v
		raw = (w / (L * kappa)) * (v * t_star - d_in)
		raw = np.maximum(raw, 1e-6)
		next_widths = _enforce_width_constraints(raw, W, min_width)

		if np.max(np.abs(next_widths - widths)) <= 1e-5:
			widths = next_widths
			break

		widths = next_widths
		x_edges[0] = params.x_min
		x_edges[1:] = params.x_min + np.cumsum(widths)

	return widths


def widths_to_edges(params: Figure6Params, widths: np.ndarray) -> np.ndarray:
	n_uav = len(widths)
	x_edges = np.zeros(n_uav + 1, dtype=float)
	x_edges[0] = params.x_min
	x_edges[1:] = params.x_min + np.cumsum(widths)
	x_edges[-1] = params.x_max
	return x_edges


def compute_uav_entry_distances(params: Figure6Params, widths: np.ndarray) -> np.ndarray:
	x_edges = widths_to_edges(params, widths)
	d_in = np.zeros(len(widths), dtype=float)
	for i, w_i in enumerate(widths):
		entry_x = x_edges[i] + _entry_offset(float(w_i), params.scan_radius_km)
		d_in[i] = math.hypot(entry_x - params.base_x, params.y_min - params.base_y)
	return d_in


def compute_uav_task_times_model_h(params: Figure6Params, widths: np.ndarray) -> np.ndarray:
	v = params.uav_speed_km_h
	L = params.area_length
	w = params.scan_radius_km
	kappa = params.kappa
	d_in = compute_uav_entry_distances(params, widths)
	return (d_in + (L * widths * kappa / w)) / v


def main() -> None:
	params = Figure6Params()
	n = params.compare_n
	out_dir = Path(__file__).resolve().parent

	equal_widths = np.full(n, params.area_width / n, dtype=float)
	equal_time_widths = solve_equal_time_widths(params, n)

	t_equal_width = compute_uav_task_times_model_h(params, equal_widths)
	t_equal_time = compute_uav_task_times_model_h(params, equal_time_widths)

	x = np.arange(n)
	bar_w = 0.38

	fig, ax = plt.subplots(figsize=(11, 6.8))
	ax.bar(x - bar_w / 2.0, t_equal_width, width=bar_w, color="#adb5bd", edgecolor="#6c757d", label="Equal-width partition")
	ax.bar(x + bar_w / 2.0, t_equal_time, width=bar_w, color="#4cc9f0", edgecolor="#1d3557", label="Equal-time partition")

	ax.set_title("Figure 6 Per-UAV Task Time under Different Partition Strategies", fontsize=14)
	ax.set_xlabel("UAV index")
	ax.set_ylabel("Task time (h)")
	ax.set_xticks(x)
	ax.set_xticklabels([f"UAV{i + 1}" for i in range(n)])
	ax.grid(axis="y", linestyle="--", alpha=0.25)
	ax.legend(loc="upper right")

	std_equal_width = float(np.std(t_equal_width, ddof=0))
	std_equal_time = float(np.std(t_equal_time, ddof=0))
	range_equal_width = float(np.max(t_equal_width) - np.min(t_equal_width))
	range_equal_time = float(np.max(t_equal_time) - np.min(t_equal_time))

	ax.text(
		0.02,
		0.97,
		(
			f"N = {n}\\n"
			f"Std (equal-width) = {std_equal_width:.3f} h\\n"
			f"Std (equal-time) = {std_equal_time:.3f} h\\n"
			f"Range (equal-width) = {range_equal_width:.3f} h\\n"
			f"Range (equal-time) = {range_equal_time:.3f} h"
		),
		transform=ax.transAxes,
		va="top",
		ha="left",
		fontsize=9,
		bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="gray"),
	)

	out_png = out_dir / "6.png"
	fig.tight_layout()
	fig.savefig(out_png, dpi=260)
	plt.close(fig)

	out_csv = out_dir / "6_metrics.csv"
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["uav_id", "equal_width_time_h", "equal_time_time_h", "equal_width_km", "equal_time_km", "equal_width_d_in_km", "equal_time_d_in_km"])
		din_width = compute_uav_entry_distances(params, equal_widths)
		din_time = compute_uav_entry_distances(params, equal_time_widths)
		for i in range(n):
			writer.writerow(
				[
					i + 1,
					f"{t_equal_width[i]:.6f}",
					f"{t_equal_time[i]:.6f}",
					f"{equal_widths[i]:.6f}",
					f"{equal_time_widths[i]:.6f}",
					f"{din_width[i]:.6f}",
					f"{din_time[i]:.6f}",
				]
			)

	print(f"[Figure6] saved: {out_png}")
	print(f"[Figure6] metrics: {out_csv}")
	print(f"[Figure6] std equal-width: {std_equal_width:.6f} h")
	print(f"[Figure6] std equal-time: {std_equal_time:.6f} h")


if __name__ == "__main__":
	main()
