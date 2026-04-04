"""Figure 10: Posterior probability heatmap after one strip search."""

from __future__ import annotations

import csv
import logging
import math
import runpy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


FIG3 = runpy.run_path(str(Path(__file__).resolve().with_name("3.py")))
Figure3Params = FIG3["Figure3Params"]
solve_partition_widths = FIG3["solve_partition_widths"]
_strip_centers = FIG3["_strip_centers"]


@dataclass(frozen=True)
class Figure10Params:
	x_min: float = 0.0
	x_max: float = 306.0
	y_min: float = 0.0
	y_max: float = 444.0
	grid_km: float = 2.0
	focus_uav_index: int = 1
	focus_strip_index: int = 0
	target_speed_km_h: float = 30.0
	pred_sigma_km: float = 10.0
	post_flow_sigma_km: float = 14.0
	prior_floor: float = 0.02
	prior_mix_primary: float = 0.72
	prior_mix_secondary: float = 0.28
	prior_primary_sigma_x: float = 28.0
	prior_primary_sigma_y: float = 62.0
	prior_secondary_sigma_x: float = 36.0
	prior_secondary_sigma_y: float = 46.0
	prior_secondary_offset_x: float = 82.0
	prior_secondary_offset_y: float = -120.0

	@property
	def area_width(self) -> float:
		return self.x_max - self.x_min

	@property
	def area_length(self) -> float:
		return self.y_max - self.y_min


def build_edges(v_min: float, v_max: float, step: float) -> np.ndarray:
	edges = np.arange(v_min, v_max + 1e-9, step)
	if edges[-1] < v_max:
		edges = np.append(edges, v_max)
	return edges


def gaussian_2d(x: np.ndarray, y: np.ndarray, cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
	sx = max(1e-9, float(sx))
	sy = max(1e-9, float(sy))
	return np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def normalize_density(density: np.ndarray) -> np.ndarray:
	density = np.maximum(density, 0.0)
	total = float(np.sum(density))
	if total <= 0.0:
		return density
	return density / total


def gaussian_kernel_1d(sigma_cells: float) -> np.ndarray:
	sigma_cells = max(1e-6, float(sigma_cells))
	radius = max(1, int(math.ceil(3.0 * sigma_cells)))
	x = np.arange(-radius, radius + 1, dtype=np.float64)
	kernel = np.exp(-0.5 * (x / sigma_cells) ** 2)
	kernel /= np.sum(kernel)
	return kernel


def convolve_reflect_1d(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
	radius = kernel.size // 2
	pad_width = [(0, 0)] * arr.ndim
	pad_width[axis] = (radius, radius)
	padded = np.pad(arr, pad_width, mode="reflect")
	result = np.zeros_like(arr, dtype=np.float64)

	if axis == 0:
		for offset, weight in enumerate(kernel):
			result += weight * padded[offset : offset + arr.shape[0], :]
	elif axis == 1:
		for offset, weight in enumerate(kernel):
			result += weight * padded[:, offset : offset + arr.shape[1]]
	else:
		raise ValueError("axis must be 0 or 1")
	return result


def gaussian_blur(field: np.ndarray, sigma_cells: float) -> np.ndarray:
	kernel = gaussian_kernel_1d(sigma_cells)
	blurred = convolve_reflect_1d(field, kernel, axis=0)
	blurred = convolve_reflect_1d(blurred, kernel, axis=1)
	return normalize_density(blurred)


def main() -> None:
	params = Figure10Params()
	out_dir = Path(__file__).resolve().parent

	fig3_params = Figure3Params()
	solved = solve_partition_widths(fig3_params, 4)
	x_edges_uav = solved["x_edges"]
	focus_uav = params.focus_uav_index
	focus_strips = _strip_centers(
		float(x_edges_uav[focus_uav]),
		float(x_edges_uav[focus_uav + 1]),
		fig3_params.scan_radius_km,
	)
	focus_strip = float(focus_strips[params.focus_strip_index])
	search_left = focus_strip - fig3_params.scan_radius_km
	search_right = focus_strip + fig3_params.scan_radius_km

	x_edges = build_edges(params.x_min, params.x_max, params.grid_km)
	y_edges = build_edges(params.y_min, params.y_max, params.grid_km)
	x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
	y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
	Xc, Yc = np.meshgrid(x_centers, y_centers)

	# Initial belief: a bimodal prior that makes the posterior redistribution easy to see.
	prior = (
		params.prior_mix_primary
		* gaussian_2d(
			Xc,
			Yc,
			focus_strip,
			292.0,
			params.prior_primary_sigma_x,
			params.prior_primary_sigma_y,
		)
		+ params.prior_mix_secondary
		* gaussian_2d(
			Xc,
			Yc,
			focus_strip + params.prior_secondary_offset_x,
			292.0 + params.prior_secondary_offset_y,
			params.prior_secondary_sigma_x,
			params.prior_secondary_sigma_y,
		)
		+ params.prior_floor
	)
	prior = normalize_density(prior)

	# Markov prediction: one diffusion-like step on the belief field.
	predicted = gaussian_blur(prior, sigma_cells=params.pred_sigma_km / params.grid_km)

	# Search update: one strip is searched and the target is not found.
	search_mask = (Xc >= search_left) & (Xc <= search_right)
	posterior = np.where(search_mask, 0.0, predicted)
	posterior = normalize_density(posterior)

	# A short time after search: propagate posterior once more to visualize flow.
	posterior_later = gaussian_blur(posterior, sigma_cells=params.post_flow_sigma_km / params.grid_km)
	delta_t_h = params.post_flow_sigma_km / max(1e-9, params.target_speed_km_h)

	# Display scale in a readable range.
	scale = 1e4
	prior_vis = prior * scale
	pred_vis = predicted * scale
	post_vis = posterior * scale
	post_later_vis = posterior_later * scale
	vmax = max(
		float(np.max(prior_vis)),
		float(np.max(pred_vis)),
		float(np.max(post_vis)),
		float(np.max(post_later_vis)),
	)

	fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2), sharex=True, sharey=True)
	cmap = "magma"
	panel_data = [
		(prior_vis, "(a) Initial probability"),
		(post_vis, "(b) Posterior after strip search"),
		(post_later_vis, f"(c) Short-time posterior evolution (~{delta_t_h:.2f} h after scan)"),
	]

	for ax, (data, title) in zip(axes, panel_data):
		im = ax.imshow(
			data,
			extent=(params.x_min, params.x_max, params.y_min, params.y_max),
			origin="lower",
			cmap=cmap,
			vmin=0,
			vmax=vmax,
			interpolation="nearest",
			aspect="auto",
		)
		ax.add_patch(
			Rectangle(
				(search_left, params.y_min),
				search_right - search_left,
				params.area_length,
				facecolor="none",
				edgecolor="#4cc9f0",
				linestyle="--",
				linewidth=1.6,
				zorder=3.0,
			)
		)
		ax.axvline(focus_strip, color="white", linestyle=":", linewidth=1.0, alpha=0.8, zorder=3.1)
		ax.set_title(title, fontsize=12)
		ax.set_xlabel("X (km)")
		ax.set_xlim(params.x_min, params.x_max)
		ax.set_ylim(params.y_min, params.y_max)
		ax.grid(alpha=0.12, linestyle="--")

	axes[0].set_ylabel("Y (km)")

	cax = fig.add_axes([0.895, 0.19, 0.016, 0.64])
	cbar = fig.colorbar(im, cax=cax)
	cbar.set_label(r"Probability density $\times 10^{-4}$")

	fig.suptitle("Figure 10 Posterior Probability Heatmap after One Strip Search", fontsize=15)
	fig.text(
		0.5,
		0.06,
		f"Dashed cyan frame: searched strip. (b) is immediate posterior; (c) shows about {delta_t_h:.2f} h after UAV strip scan.",
		ha="center",
		fontsize=10,
	)
	fig.subplots_adjust(left=0.05, right=0.87, bottom=0.14, top=0.88, wspace=0.14)

	prior_mass = float(np.sum(prior[search_mask]))
	pred_mass = float(np.sum(predicted[search_mask]))
	post_mass = float(np.sum(posterior[search_mask]))
	post_later_mass = float(np.sum(posterior_later[search_mask]))
	def entropy(arr: np.ndarray) -> float:
		positive = arr[arr > 0]
		if positive.size == 0:
			return 0.0
		return float(-np.sum(positive * np.log(positive)))

	out_png = out_dir / "10.png"
	fig.savefig(out_png, dpi=260)
	plt.close(fig)

	out_csv = out_dir / "10_metrics.csv"
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow([
			"focus_uav",
			"focus_strip_center_km",
			"delta_t_h_after_scan",
			"search_left_km",
			"search_right_km",
			"prior_strip_mass",
			"predicted_strip_mass",
			"posterior_strip_mass",
			"posterior_later_strip_mass",
			"prior_entropy",
			"predicted_entropy",
			"posterior_entropy",
			"posterior_later_entropy",
		])
		writer.writerow([
			focus_uav + 1,
			f"{focus_strip:.6f}",
			f"{delta_t_h:.6f}",
			f"{search_left:.6f}",
			f"{search_right:.6f}",
			f"{prior_mass:.8f}",
			f"{pred_mass:.8f}",
			f"{post_mass:.8f}",
			f"{post_later_mass:.8f}",
			f"{entropy(prior):.8f}",
			f"{entropy(predicted):.8f}",
			f"{entropy(posterior):.8f}",
			f"{entropy(posterior_later):.8f}",
		])

	print(f"[Figure10] saved: {out_png}")
	print(f"[Figure10] metrics: {out_csv}")
	print(
		"[Figure10] searched strip mass: "
		f"prior={prior_mass:.6f}, predicted={pred_mass:.6f}, "
		f"posterior={post_mass:.6f}, posterior_later={post_later_mass:.6f}"
	)


if __name__ == "__main__":
	main()
