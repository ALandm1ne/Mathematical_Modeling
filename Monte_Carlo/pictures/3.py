"""Figure 3: Multi-UAV cooperative strip coverage (batch N=1..8).

Conventions:
1) Subregion widths are solved by the equal-time partition formula.
2) Transit segments are dashed lines.
3) Scanning segments follow Y-axis boustrophedon motion.
4) Top/bottom turns are semicircular arcs with turning radius R_turn = w.
5) Outputs include 3_1.png ... 3_8.png and formula metrics CSV.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Figure3Params:
	x_min: float = 0.0
	x_max: float = 306.0
	y_min: float = 0.0
	y_max: float = 444.0
	base_x: float = -314.0
	base_y: float = -323.0
	uav_speed_km_h: float = 150.0
	scan_radius_km: float = 20.0
	kappa: float = 1.0

	@property
	def area_width(self) -> float:
		return self.x_max - self.x_min

	@property
	def area_length(self) -> float:
		return self.y_max - self.y_min


def _entry_offset(width: float, w: float) -> float:
	# 入口点偏置不超过子区中线，避免窄子区越界。
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


def solve_partition_widths(params: Figure3Params, n_uav: int, max_iter: int = 40) -> dict:
	w = params.scan_radius_km
	v = params.uav_speed_km_h
	L = params.area_length
	W = params.area_width
	kappa = params.kappa

	widths = np.full(n_uav, W / n_uav, dtype=float)
	min_width = min(2.0 * w, 0.5 * W / n_uav)
	x_edges = np.linspace(params.x_min, params.x_max, n_uav + 1)

	d_in = np.zeros(n_uav, dtype=float)
	t_star = 0.0

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

	x_edges[0] = params.x_min
	x_edges[1:] = params.x_min + np.cumsum(widths)
	x_edges[-1] = params.x_max
	x_edges = np.maximum.accumulate(x_edges)

	return {
		"widths": widths,
		"x_edges": x_edges,
		"d_in": d_in,
		"t_star": t_star,
	}


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

	# If the last strip cannot cover the right edge, add one more strip center.
	# This may exceed the allocated subregion, but guarantees full coverage.
	if centers and (centers[-1] + turn_radius < x_right - 1e-9):
		centers.append(centers[-1] + step)
	if not centers:
		centers.append(0.5 * (x_left + x_right))
	return centers


def _arc_points(cx: float, cy: float, r: float, theta_start: float, theta_end: float, n_pts: int = 60):
	theta = np.linspace(theta_start, theta_end, n_pts)
	return cx + r * np.cos(theta), cy + r * np.sin(theta)


def _connection_length(dx: float, turn_radius: float) -> float:
	if abs(dx - 2.0 * turn_radius) <= 1e-6:
		return math.pi * turn_radius
	return abs(dx)


def compute_uav_timing(params: Figure3Params, xs: list[float]) -> dict:
	v = params.uav_speed_km_h
	L = params.area_length
	r = params.scan_radius_km
	entry_x = xs[0]

	transit_km = math.hypot(entry_x - params.base_x, params.y_min - params.base_y)
	scan_km = len(xs) * L
	for i in range(len(xs) - 1):
		scan_km += _connection_length(xs[i + 1] - xs[i], r)

	total_km = transit_km + scan_km
	return {
		"entry_x": entry_x,
		"transit_km": transit_km,
		"scan_km": scan_km,
		"total_km": total_km,
		"t_in_h": transit_km / v,
		"t_done_h": total_km / v,
	}


def generate_uav_path(
	params: Figure3Params,
	x_left: float,
	x_right: float,
	color: tuple,
	ax,
	xs: list[float] | None = None,
) -> list[float]:
	y0 = params.y_min
	y1 = params.y_max
	r = params.scan_radius_km

	if xs is None:
		xs = _strip_centers(x_left, x_right, r)
	entry_x = xs[0]

	# 航渡段：基地到入口点。
	ax.plot([params.base_x, entry_x], [params.base_y, y0], linestyle="--", color=color, linewidth=1.4, alpha=0.9)

	going_up = True
	for i, x in enumerate(xs):
		if going_up:
			ax.plot([x, x], [y0, y1], linestyle="-", color=color, linewidth=2.0)
			ax.annotate("", xy=(x, 0.60 * y1), xytext=(x, 0.40 * y1), arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
		else:
			ax.plot([x, x], [y1, y0], linestyle="-", color=color, linewidth=2.0)
			ax.annotate("", xy=(x, 0.40 * y1), xytext=(x, 0.60 * y1), arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

		if i == len(xs) - 1:
			break

		x_next = xs[i + 1]
		dx = x_next - x
		if abs(dx - 2.0 * r) <= 1e-6:
			if going_up:
				cx, cy = x + r, y1
				arc_x, arc_y = _arc_points(cx, cy, r, math.pi, 0.0)
			else:
				cx, cy = x + r, y0
				arc_x, arc_y = _arc_points(cx, cy, r, math.pi, 2.0 * math.pi)
			ax.plot(arc_x, arc_y, linestyle="-", color=color, linewidth=2.0)
		else:
			# 末端宽度非 2R 时，用贴边直线衔接，避免几何不连续。
			y_edge = y1 if going_up else y0
			ax.plot([x, x_next], [y_edge, y_edge], linestyle="-", color=color, linewidth=2.0)

		going_up = not going_up

	return xs


def render_for_n(params: Figure3Params, n_uav: int, output_path: Path, metrics_rows: list[dict]) -> None:
	solved = solve_partition_widths(params, n_uav)
	widths = solved["widths"]
	x_edges = solved["x_edges"]
	d_in = solved["d_in"]
	t_star = solved["t_star"]

	fig, ax = plt.subplots(figsize=(11, 9))
	ax.add_patch(
		Rectangle(
			(params.x_min, params.y_min),
			params.area_width,
			params.area_length,
			linewidth=1.8,
			edgecolor="black",
			facecolor="#f8fbff",
			alpha=0.95,
			zorder=0.0,
		)
	)

	cmap = plt.get_cmap("tab10", n_uav)
	timing_annotations: list[tuple[int, tuple, float, float]] = []
	for b in x_edges[1:-1]:
		ax.axvline(b, color="gray", linestyle=":", linewidth=0.9, alpha=0.75, zorder=1.5)
		ax.text(
			b,
			params.y_min - 8.0,
			f"x={b:.1f} km",
			rotation=90,
			va="top",
			ha="center",
			fontsize=8,
			color="dimgray",
			bbox=dict(boxstyle="round", facecolor="white", alpha=0.65, edgecolor="none"),
			clip_on=False,
		)

	for i in range(n_uav):
		color = cmap(i)
		# Fill each assigned partition with UAV-matched background color.
		ax.add_patch(
			Rectangle(
				(x_edges[i], params.y_min),
				x_edges[i + 1] - x_edges[i],
				params.area_length,
				linewidth=0.0,
				facecolor=color,
				alpha=0.48,
				zorder=1.0,
			)
		)
		xs = _strip_centers(x_edges[i], x_edges[i + 1], params.scan_radius_km)
		timing = compute_uav_timing(params, xs)
		generate_uav_path(params, x_edges[i], x_edges[i + 1], color, ax, xs=xs)

		label_x = timing["entry_x"]
		ax.text(
			label_x,
			params.y_min + 10.0,
			f"UAV {i + 1}",
			color=color,
			fontsize=9,
			fontweight="bold",
			ha="center",
			va="bottom",
			zorder=3.0,
		)

		timing_annotations.append((i + 1, color, timing["t_in_h"], timing["t_done_h"]))

		metrics_rows.append(
			{
				"N": n_uav,
				"uav_id": i,
				"d_in_km": float(d_in[i]),
				"W_i_km": float(widths[i]),
				"x_left_km": float(x_edges[i]),
				"x_right_km": float(x_edges[i + 1]),
				"T_star_h": float(t_star),
				"t_in_h": float(timing["t_in_h"]),
				"t_done_h": float(timing["t_done_h"]),
			}
		)

	ax.scatter(params.base_x, params.base_y, marker="*", s=150, color="black", label="Base Station")

	formula_text = (
		r"$W_i^*=\frac{w}{L\kappa}(v_uT_N^*-d_i^{in})$" + "\n"
		r"$T_N^*=\frac{1}{v_u}(\frac{1}{N}\sum d_j^{in}+\frac{LW\kappa}{Nw})$"
	)
	ax.text(
		0.02,
		0.98,
		formula_text,
		transform=ax.transAxes,
		va="top",
		ha="left",
		fontsize=10,
		bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
	)

	ax.set_title(f"Figure 3 Multi-UAV Cooperative Strip Coverage (N={n_uav})", fontsize=14)
	ax.set_xlabel("X (km)")
	ax.set_ylabel("Y (km)")
	ax.set_xlim(params.base_x - 20.0, params.x_max + 20.0)
	ax.set_ylim(params.base_y - 20.0, params.y_max + 20.0)
	ax.set_aspect("equal", adjustable="box")
	ax.grid(alpha=0.25, linestyle="--")
	ax.legend(loc="lower right")

	ax.text(
		1.02,
		0.98,
		"UAV Timing (from base)",
		transform=ax.transAxes,
		ha="left",
		va="top",
		fontsize=9,
		fontweight="bold",
		clip_on=False,
	)
	for idx, (uav_id, color, t_in_h, t_done_h) in enumerate(timing_annotations):
		y = 0.93 - 0.085 * idx
		ax.text(
			1.02,
			y,
			f"UAV {uav_id}: t_in={t_in_h:.2f} h, t_done={t_done_h:.2f} h",
			transform=ax.transAxes,
			ha="left",
			va="top",
			fontsize=8,
			color=color,
			bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"),
			clip_on=False,
		)

	fig.text(
		0.5,
		0.02,
		"Note: Arc-turn convention follows the current simulator, with turning radius R_turn = w.",
		ha="center",
		fontsize=10,
	)

	fig.tight_layout(rect=(0.0, 0.05, 0.84, 1.0))
	fig.savefig(output_path, dpi=220)
	plt.close(fig)


def export_metrics_csv(rows: list[dict], out_csv: Path) -> None:
	fields = [
		"N",
		"uav_id",
		"d_in_km",
		"W_i_km",
		"x_left_km",
		"x_right_km",
		"T_star_h",
		"t_in_h",
		"t_done_h",
	]
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fields)
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	params = Figure3Params()
	out_dir = Path(__file__).resolve().parent
	metrics: list[dict] = []

	for n in range(1, 9):
		out_png = out_dir / f"3_{n}.png"
		render_for_n(params, n, out_png, metrics)
		print(f"[Figure3] saved: {out_png}")

	metrics_csv = out_dir / "3_metrics.csv"
	export_metrics_csv(metrics, metrics_csv)
	print(f"[Figure3] metrics: {metrics_csv}")


if __name__ == "__main__":
	main()
