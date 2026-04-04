"""Figure 5: System completion time vs number of UAVs."""

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
class Figure5Params:
	x_min: float = 0.0
	x_max: float = 306.0
	y_min: float = 0.0
	y_max: float = 444.0
	base_x: float = -314.0
	base_y: float = -323.0
	uav_speed_km_h: float = 150.0
	scan_radius_km: float = 20.0
	kappa: float = 1.0
	t_max_h: float = 40.0
	t_limit_h: float = 10.0
	max_n: int = 16

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


def solve_partition_widths(params: Figure5Params, n_uav: int, max_iter: int = 40) -> np.ndarray:
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


def _strip_centers(x_left: float, x_right: float, turn_radius: float) -> list[float]:
	usable = x_right - x_left
	if usable <= 2.0 * turn_radius + 1e-9:
		return [0.5 * (x_left + x_right)]

	start = x_left + turn_radius
	end = x_right - turn_radius
	centers: list[float] = []
	x = start
	step = 2.0 * turn_radius
	while x <= end + 1e-9:
		centers.append(x)
		x += step

	if centers and (centers[-1] + turn_radius < x_right - 1e-9):
		centers.append(centers[-1] + step)
	if not centers:
		centers.append(0.5 * (x_left + x_right))
	return centers


def _connection_length(dx: float, turn_radius: float) -> float:
	if abs(dx - 2.0 * turn_radius) <= 1e-6:
		return math.pi * turn_radius
	return abs(dx)


def compute_uav_done_time_h(params: Figure5Params, xs: list[float]) -> float:
	v = params.uav_speed_km_h
	L = params.area_length
	r = params.scan_radius_km
	entry_x = xs[0]

	transit_km = math.hypot(entry_x - params.base_x, params.y_min - params.base_y)
	scan_km = len(xs) * L
	for i in range(len(xs) - 1):
		scan_km += _connection_length(xs[i + 1] - xs[i], r)

	return (transit_km + scan_km) / v


def compute_system_time_h(params: Figure5Params, n_uav: int) -> float:
	widths = solve_partition_widths(params, n_uav)
	x_edges = np.zeros(n_uav + 1, dtype=float)
	x_edges[0] = params.x_min
	x_edges[1:] = params.x_min + np.cumsum(widths)
	x_edges[-1] = params.x_max

	done_times: list[float] = []
	for i in range(n_uav):
		xs = _strip_centers(x_edges[i], x_edges[i + 1], params.scan_radius_km)
		done_times.append(compute_uav_done_time_h(params, xs))

	return float(max(done_times))


def main() -> None:
	params = Figure5Params()
	out_dir = Path(__file__).resolve().parent

	n_values = list(range(1, params.max_n + 1))
	t_sys_values = [compute_system_time_h(params, n) for n in n_values]

	n_meet_10 = None
	for n, t_sys in zip(n_values, t_sys_values):
		if t_sys <= params.t_limit_h:
			n_meet_10 = n
			break

	fig, ax = plt.subplots(figsize=(10, 6.5))
	ax.plot(n_values, t_sys_values, color="#1d3557", linewidth=2.4, marker="o", markersize=5.5, label=r"$T_{sys}(N)$")
	ax.axhline(params.t_max_h, color="#6c757d", linestyle="--", linewidth=1.5, label=r"$T_{max}=40\,h$")
	ax.axhline(params.t_limit_h, color="#e63946", linestyle="--", linewidth=1.5, label="10-hour limit")

	if n_meet_10 is not None:
		t_star = t_sys_values[n_meet_10 - 1]
		ax.scatter([n_meet_10], [t_star], color="#e63946", s=55, zorder=3.0)
		ax.annotate(
			f"First feasible N = {n_meet_10}",
			xy=(n_meet_10, t_star),
			xytext=(n_meet_10 + 0.7, t_star + 2.2),
			arrowprops=dict(arrowstyle="->", color="#e63946", lw=1.2),
			fontsize=10,
			color="#e63946",
		)

	ax.set_title("Figure 5 System Completion Time vs Number of UAVs", fontsize=14)
	ax.set_xlabel("Number of UAVs N")
	ax.set_ylabel(r"System Completion Time $T_{sys}$ (h)")
	ax.set_xticks(n_values)
	ax.set_xlim(1, params.max_n)
	y_top = max(max(t_sys_values) * 1.08, params.t_max_h * 1.10)
	ax.set_ylim(0.0, y_top)
	ax.grid(alpha=0.25, linestyle="--")
	ax.legend(loc="upper right")

	out_png = out_dir / "5.png"
	fig.tight_layout()
	fig.savefig(out_png, dpi=260)
	plt.close(fig)

	out_csv = out_dir / "5_metrics.csv"
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["N", "T_sys_h", "T_max_h", "T_limit_h"])
		for n, t_sys in zip(n_values, t_sys_values):
			writer.writerow([n, f"{t_sys:.6f}", f"{params.t_max_h:.6f}", f"{params.t_limit_h:.6f}"])

	print(f"[Figure5] saved: {out_png}")
	print(f"[Figure5] metrics: {out_csv}")
	if n_meet_10 is not None:
		print(f"[Figure5] first N meeting 10h: N={n_meet_10}")
	else:
		print("[Figure5] no N in current range meets 10h")


if __name__ == "__main__":
	main()
