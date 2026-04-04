"""Figure 7: Grid mapping of strip coverage results."""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Figure7Params:
	x_min: float = 0.0
	x_max: float = 306.0
	y_min: float = 0.0
	y_max: float = 444.0
	base_x: float = -314.0
	base_y: float = -323.0
	uav_speed_km_h: float = 150.0
	scan_radius_km: float = 20.0
	kappa: float = 1.0
	n_uav: int = 8
	grid_km: float = 1.0

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


def solve_equal_time_widths(params: Figure7Params, max_iter: int = 40) -> np.ndarray:
	n_uav = params.n_uav
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


def build_edges(v_min: float, v_max: float, step: float) -> np.ndarray:
	edges = np.arange(v_min, v_max + 1e-9, step)
	if edges[-1] < v_max:
		edges = np.append(edges, v_max)
	return edges


def compute_owner_by_partition(x_centers: np.ndarray, x_edges_uav: np.ndarray) -> np.ndarray:
	# Use partition ownership for stable color regions in the global view.
	owner = np.searchsorted(x_edges_uav, x_centers, side="right") - 1
	owner = np.clip(owner, 0, len(x_edges_uav) - 2)
	return owner.astype(int)


def draw_grid(ax, x_edges: np.ndarray, y_edges: np.ndarray, color: str = "#b7b7b7", lw: float = 0.35) -> None:
	for x in x_edges:
		ax.plot([x, x], [y_edges[0], y_edges[-1]], color=color, linewidth=lw, alpha=0.85, zorder=0.2)
	for y in y_edges:
		ax.plot([x_edges[0], x_edges[-1]], [y, y], color=color, linewidth=lw, alpha=0.85, zorder=0.2)


def main() -> None:
	params = Figure7Params()
	out_dir = Path(__file__).resolve().parent

	widths = solve_equal_time_widths(params)
	x_edges_uav = np.zeros(params.n_uav + 1, dtype=float)
	x_edges_uav[0] = params.x_min
	x_edges_uav[1:] = params.x_min + np.cumsum(widths)
	x_edges_uav[-1] = params.x_max

	strips_by_uav: list[list[float]] = []
	for i in range(params.n_uav):
		strips = _strip_centers(float(x_edges_uav[i]), float(x_edges_uav[i + 1]), params.scan_radius_km)
		strips_by_uav.append(strips)

	x_edges = build_edges(params.x_min, params.x_max, params.grid_km)
	y_edges = build_edges(params.y_min, params.y_max, params.grid_km)
	x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
	y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

	owner_by_x = compute_owner_by_partition(x_centers, x_edges_uav)
	owner_grid = np.tile(owner_by_x, (len(y_centers), 1))

	fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15.2, 6.9))
	cmap = plt.get_cmap("tab10", params.n_uav)
	norm = BoundaryNorm(np.arange(-0.5, params.n_uav + 0.5, 1.0), params.n_uav)

	# Left subplot: global mapping with all strips and covered grids.
	masked_owner = np.ma.masked_where(owner_grid < 0, owner_grid)
	ax_l.pcolormesh(x_edges, y_edges, masked_owner, cmap=cmap, norm=norm, shading="flat", alpha=0.45, zorder=0.5)
	draw_grid(ax_l, x_edges, y_edges)

	for u, strips in enumerate(strips_by_uav):
		for xc in strips:
			ax_l.plot([xc, xc], [params.y_min, params.y_max], color=cmap(u), linewidth=1.4, zorder=1.2)

	for x in x_edges_uav[1:-1]:
		ax_l.plot([x, x], [params.y_min, params.y_max], color="#6c757d", linestyle=":", linewidth=0.9, alpha=0.9, zorder=1.1)

	legend_handles = [Patch(facecolor=cmap(i), edgecolor="none", alpha=0.55, label=f"UAV{i + 1} covered grids") for i in range(params.n_uav)]
	ax_l.legend(handles=legend_handles, loc="upper right", fontsize=7.5, ncol=2, framealpha=0.9)
	ax_l.set_title("Global grid mapping of all strip coverage", fontsize=12)
	ax_l.set_xlabel("X (km)")
	ax_l.set_ylabel("Y (km)")
	ax_l.set_xlim(params.x_min, params.x_max)
	ax_l.set_ylim(params.y_min, params.y_max)
	ax_l.set_aspect("equal", adjustable="box")

	# Right subplot: local mapping for one strip at one sweep moment.
	focus_uav = params.n_uav // 2
	focus_strips = strips_by_uav[focus_uav]
	focus_strip = focus_strips[len(focus_strips) // 2]
	x_local_min = max(params.x_min, focus_strip - 2.6 * params.scan_radius_km)
	x_local_max = min(params.x_max, focus_strip + 2.6 * params.scan_radius_km)
	y_local_min = params.y_max - 110.0
	y_local_max = params.y_max

	Xc, Yc = np.meshgrid(x_centers, y_centers)
	local_window = (
		(Xc >= x_local_min)
		& (Xc <= x_local_max)
		& (Yc >= y_local_min)
		& (Yc <= y_local_max)
	)
	local_mask = (
		local_window
		& (np.abs(Xc - focus_strip) <= params.scan_radius_km + 1e-9)
	)
	local_state = np.full(Xc.shape, np.nan, dtype=float)
	local_state[local_window] = 0.0
	local_state[local_mask] = 1.0
	local_state = np.ma.masked_invalid(local_state)
	local_cmap = ListedColormap(["#d3d7dc", "#0096c7"])
	local_norm = BoundaryNorm([-0.5, 0.5, 1.5], local_cmap.N)
	ax_r.pcolormesh(x_edges, y_edges, local_state, cmap=local_cmap, norm=local_norm, shading="flat", alpha=0.88, zorder=0.6)
	draw_grid(ax_r, x_edges, y_edges, color="#8f96a3", lw=0.45)
	ax_r.plot(
		[focus_strip - params.scan_radius_km, focus_strip - params.scan_radius_km],
		[y_local_min, y_local_max],
		color="#023047",
		linestyle="--",
		linewidth=1.1,
		alpha=0.85,
		zorder=1.2,
	)
	ax_r.plot(
		[focus_strip + params.scan_radius_km, focus_strip + params.scan_radius_km],
		[y_local_min, y_local_max],
		color="#023047",
		linestyle="--",
		linewidth=1.1,
		alpha=0.85,
		zorder=1.2,
	)

	for s in focus_strips:
		ax_r.plot([s, s], [y_local_min, y_local_max], color="#8d99ae", linewidth=0.8, alpha=0.7, zorder=1.0)
	ax_r.plot([focus_strip, focus_strip], [y_local_min, y_local_max], color="#d62828", linewidth=2.1, zorder=1.4)
	ax_r.text(
		focus_strip + 1.2,
		y_local_max - 10.0,
		f"Current strip centerline (UAV{focus_uav + 1})",
		fontsize=8.5,
		color="#d62828",
	)
	ax_r.text(
		x_local_min + 1.0,
		y_local_min + 5.0,
		f"Swath half-width r = {params.scan_radius_km:.0f} km",
		fontsize=8.5,
		color="#1d3557",
	)
	ax_r.legend(
		handles=[
			Patch(facecolor="#0096c7", edgecolor="none", alpha=0.88, label="Scanned cells"),
			Patch(facecolor="#d3d7dc", edgecolor="none", alpha=0.88, label="Unscanned cells"),
		],
		loc="upper left",
		fontsize=8,
		framealpha=0.9,
	)
	ax_r.set_title("Local mapping of one strip at a given sweep", fontsize=12)
	ax_r.set_xlabel("X (km)")
	ax_r.set_ylabel("Y (km)")
	ax_r.set_xlim(x_local_min, x_local_max)
	ax_r.set_ylim(y_local_min, y_local_max)
	ax_r.set_aspect("equal", adjustable="box")

	covered_cells = int(np.sum(owner_grid >= 0))
	total_cells = int(owner_grid.size)
	coverage_ratio = covered_cells / total_cells

	fig.suptitle("Figure 7 Grid Mapping of Strip-Coverage Results", fontsize=15)
	fig.text(
		0.5,
		0.02,
		(
			"Left: global strip-to-grid coverage mapping. "
			"Right: local mapping for a representative strip (bridge to grid-based probability update)."
		),
		ha="center",
		fontsize=10,
	)
	fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

	out_png = out_dir / "7.png"
	fig.savefig(out_png, dpi=260)
	plt.close(fig)

	out_csv = out_dir / "7_metrics.csv"
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["n_uav", "grid_km", "covered_cells", "total_cells", "coverage_ratio"])
		writer.writerow([params.n_uav, f"{params.grid_km:.6f}", covered_cells, total_cells, f"{coverage_ratio:.6f}"])

	print(f"[Figure7] saved: {out_png}")
	print(f"[Figure7] metrics: {out_csv}")
	print(f"[Figure7] covered cells: {covered_cells}/{total_cells} ({coverage_ratio:.2%})")


if __name__ == "__main__":
	main()
