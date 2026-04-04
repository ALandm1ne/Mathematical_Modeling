"""Figure 4: Single-UAV strip scanning and turning detail."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def _arc_points(cx: float, cy: float, r: float, theta_start: float, theta_end: float, n_pts: int = 80):
	theta = np.linspace(theta_start, theta_end, n_pts)
	return cx + r * np.cos(theta), cy + r * np.sin(theta)


def main() -> None:
	# Local demonstration parameters.
	y_min = 0.0
	y_max = 160.0
	strip_width_w = 20.0
	strip_spacing = 20.0
	turn_radius = 10.0

	# Three strip centerlines.
	x1 = 70.0
	x2 = x1 + strip_spacing
	x3 = x2 + strip_spacing
	centers = [x1, x2, x3]

	entry = (x1, y_min)

	fig, ax = plt.subplots(figsize=(11, 7))

	x_min = x1 - 35.0
	x_max = x3 + 35.0
	ax.add_patch(
		Rectangle(
			(x_min, y_min),
			x_max - x_min,
			y_max - y_min,
			linewidth=1.6,
			edgecolor="black",
			facecolor="#f7fbff",
			alpha=0.95,
			zorder=0.0,
		)
	)

	# Effective strip width swaths.
	for x in centers:
		ax.add_patch(
			Rectangle(
				(x - strip_width_w / 2.0, y_min),
				strip_width_w,
				y_max - y_min,
				linewidth=0.0,
				facecolor="#8ecae6",
				alpha=0.35,
				zorder=1.0,
			)
		)

	# Strip centerlines.
	for idx, x in enumerate(centers, start=1):
		ax.plot([x, x], [y_min, y_max], linestyle="--", color="#1d3557", linewidth=1.2, zorder=2.0)
		ax.text(x, y_min + 4.0, f"", ha="center", va="bottom", fontsize=9, color="#1d3557")

	# Boustrophedon path and turning arcs.
	ax.plot([x1, x1], [y_min, y_max], color="#d62828", linewidth=2.4, zorder=3.0)
	ax.plot(*_arc_points(x1 + turn_radius, y_max, turn_radius, math.pi, 0.0), color="#d62828", linewidth=2.4, zorder=3.0)
	ax.plot([x2, x2], [y_max, y_min], color="#d62828", linewidth=2.4, zorder=3.0)
	ax.plot(*_arc_points(x2 + turn_radius, y_min, turn_radius, math.pi, 2.0 * math.pi), color="#d62828", linewidth=2.4, zorder=3.0)
	ax.plot([x3, x3], [y_min, y_max], color="#d62828", linewidth=2.4, zorder=3.0)

	# Entry point.
	ax.scatter([entry[0]], [entry[1]], s=55, color="black", zorder=4.0)
	ax.text(entry[0] + 2.0, entry[1] + 8.0, "Entry point E", fontsize=9, color="black")

	# Width annotation: w = 20 km.
	xw = x2
	y_w = y_max * 0.55
	ax.annotate(
		"",
		xy=(xw - strip_width_w / 2.0, y_w),
		xytext=(xw + strip_width_w / 2.0, y_w),
		arrowprops=dict(arrowstyle="<->", color="#2a9d8f", lw=1.8),
	)
	ax.text(xw, y_w + 5.0, "w = 20 km", ha="center", va="bottom", fontsize=10, color="#2a9d8f", fontweight="bold")

	# Adjacent strip spacing annotation.
	ys = y_min + 15.0
	ax.annotate("", xy=(x1, ys), xytext=(x2, ys), arrowprops=dict(arrowstyle="<->", color="#e76f51", lw=1.6))
	ax.text((x1 + x2) / 2.0, ys + 4.0, "Adjacent strip spacing = 20 km", ha="center", va="bottom", fontsize=9, color="#e76f51")

	# Direction arrows.
	ax.annotate("", xy=(x1, y_max * 0.75), xytext=(x1, y_max * 0.55), arrowprops=dict(arrowstyle="->", color="#d62828", lw=1.6))
	ax.annotate("", xy=(x2, y_max * 0.25), xytext=(x2, y_max * 0.45), arrowprops=dict(arrowstyle="->", color="#d62828", lw=1.6))
	ax.annotate("", xy=(x3, y_max * 0.75), xytext=(x3, y_max * 0.55), arrowprops=dict(arrowstyle="->", color="#d62828", lw=1.6))

	# Local zoom panel (placed outside the main plotting region).
	axins = inset_axes(
		ax,
		width="52%",
		height="64%",
		loc="upper left",
		bbox_to_anchor=(1.02, 0.02, 1.0, 1.0),
		bbox_transform=ax.transAxes,
		borderpad=0.0,
	)
	axins.set_xlim(x1 - 10.0, x2 + 10.0)
	axins.set_ylim(y_max - 40.0, y_max + 30.0)
	# Keep region color swaths in the zoom panel as well.
	for x in centers:
		axins.add_patch(
			Rectangle(
				(x - strip_width_w / 2.0, y_min),
				strip_width_w,
				y_max - y_min,
				linewidth=0.0,
				facecolor="#8ecae6",
				alpha=0.35,
				zorder=1.0,
			)
		)
	# Include region boundary lines in the zoomed panel.
	left_boundary = x1 - strip_width_w / 2.0
	right_boundary = x2 + strip_width_w / 2.0
	axins.plot([left_boundary, left_boundary], [y_max - 40.0, y_max], color="black", linewidth=1.2, alpha=0.9)
	axins.plot([right_boundary, right_boundary], [y_max - 40.0, y_max], color="black", linewidth=1.2, alpha=0.9)
	axins.plot([left_boundary, right_boundary], [y_max, y_max], color="black", linewidth=1.2, alpha=0.9)
	axins.plot([x1, x1], [y_max - 40.0, y_max], color="#d62828", linewidth=2.0)
	axins.plot(*_arc_points(x1 + turn_radius, y_max, turn_radius, math.pi, 0.0), color="#d62828", linewidth=2.0)
	axins.plot([x2, x2], [y_max, y_max - 40.0], color="#d62828", linewidth=2.0)
	# Mark turning center and radius on the zoom panel.
	turn_center_top = (x1 + turn_radius, y_max)
	axins.scatter([turn_center_top[0]], [turn_center_top[1]], s=24, color="#1d3557", zorder=4.0)
	axins.text(
		turn_center_top[0] - 5.0,
		turn_center_top[1] - 5.0,
		f"Center ({turn_center_top[0]:.0f}, {turn_center_top[1]:.0f})",
		fontsize=7,
		color="#1d3557",
	)
	axins.plot([turn_center_top[0], x1], [y_max, y_max], linestyle="--", color="#1d3557", linewidth=1.0)
	axins.text(
		(turn_center_top[0] + x1) / 2.0 + 1.0,
		y_max + 1.8,
		f"R = {turn_radius:.0f} km",
		ha="center",
		va="bottom",
		fontsize=7,
		color="#1d3557",
	)
	axins.set_aspect("equal", adjustable="box")
	axins.set_xticks([])
	axins.set_yticks([])
	axins.set_title("Local zoom: path and region boundary", fontsize=8)
	mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=1.0)

	ax.set_title("Figure 4 Single-UAV Strip Scanning and Turning Detail", fontsize=14)
	ax.set_xlabel("X (km)")
	ax.set_ylabel("Y (km)")
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min - 12.0, y_max + 10.0)
	ax.grid(alpha=0.25, linestyle="--")
	ax.set_aspect("equal", adjustable="box")

	out_path = Path(__file__).resolve().parent / "4.png"
	fig.subplots_adjust(left=0.08, right=0.70, bottom=0.10, top=0.92)
	fig.savefig(out_path, dpi=260)
	plt.close(fig)
	print(f"[Figure4] saved: {out_path}")


if __name__ == "__main__":
	main()
