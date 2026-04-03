"""Figure 3: Multi-UAV collaborative strip-coverage path map.

运行:
    uv run pictures/figure3_multi_uav_strip_total.py
或:
    python pictures/figure3_multi_uav_strip_total.py

输出:
    pictures/Fig3_Multi_UAV_Strip_Coverage.png
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import build_default_config


# Journal-style typography: prefer Times New Roman.
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
mpl.rcParams["axes.unicode_minus"] = False


@dataclass
class Figure3Style:
    title: str = "Figure 3. Multi-UAV Collaborative Strip-Coverage Mission Paths"
    output_name: str = "3.png"
    figure_size: tuple[float, float] = (12.5, 9.5)
    dpi: int = 320


def _sample_arc_points(
    center_x: float,
    center_y: float,
    radius: float,
    theta_start: float,
    theta_end: float,
    n: int = 40,
) -> np.ndarray:
    """按角度采样圆弧点，角度单位为弧度。"""
    thetas = np.linspace(theta_start, theta_end, n)
    x = center_x + radius * np.cos(thetas)
    y = center_y + radius * np.sin(thetas)
    return np.column_stack((x, y))


def _build_serpentine_path(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    lane_pitch: float,
) -> np.ndarray:
    """Generate a serpentine strip path.

    Design constraints:
    1) Entry point lies exactly on the boundary (top boundary y=y_min).
    2) U-turn arcs are outside the target boundary (y<y_min or y>y_max).
    """
    width = x_max - x_min
    if width <= 1e-6:
        return np.zeros((0, 2), dtype=np.float64)

    margin = max(3.0, lane_pitch * 0.30)
    xs = np.arange(x_min + margin, x_max - margin + 1e-9, lane_pitch)
    if xs.size < 2:
        xs = np.array([x_min + margin, x_max - margin], dtype=np.float64)

    points: list[tuple[float, float]] = []

    # Entry point exactly on the top boundary.
    points.append((float(xs[0]), y_min))

    go_down = True
    for i, x in enumerate(xs):
        start_y = y_min if go_down else y_max
        end_y = y_max if go_down else y_min

        # Keep lane endpoints on the boundary.
        if points[-1] != (float(x), start_y):
            points.append((float(x), start_y))
        points.append((float(x), end_y))

        if i == len(xs) - 1:
            break

        x_next = float(xs[i + 1])
        r = abs(x_next - float(x)) / 2.0
        cx = (float(x) + x_next) / 2.0

        if go_down:
            # Bottom turn outside boundary: y >= y_max.
            arc = _sample_arc_points(
                center_x=cx,
                center_y=y_max,
                radius=r,
                theta_start=np.pi,
                theta_end=0.0,
            )
        else:
            # Top turn outside boundary: y <= y_min.
            arc = _sample_arc_points(
                center_x=cx,
                center_y=y_min,
                radius=r,
                theta_start=np.pi,
                theta_end=2.0 * np.pi,
            )

        # Skip the first arc point to avoid duplicating the current boundary endpoint.
        points.extend([(float(px), float(py)) for px, py in arc[1:]])
        go_down = not go_down

    return np.asarray(points, dtype=np.float64)


def _add_path_arrows(ax, path_xy: np.ndarray, color, every_n: int = 70) -> None:
    """沿路径均匀加方向箭头。"""
    if path_xy.shape[0] < 3:
        return

    for i in range(every_n, path_xy.shape[0] - 1, every_n):
        p0 = path_xy[i - 1]
        p1 = path_xy[i + 1]
        ax.annotate(
            "",
            xy=(p1[0], p1[1]),
            xytext=(p0[0], p0[1]),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.0,
                color=color,
                mutation_scale=9,
                alpha=0.9,
            ),
            zorder=4,
        )


def plot_figure3_total() -> str:
    cfg = build_default_config(script_dir=PROJECT_ROOT, require_cuda_override=False)
    style = Figure3Style()

    # Target-particle motion bounds (this is what the black frame must represent).
    particle_x_min = 0.0
    particle_y_min = 0.0
    particle_w = float(cfg.environment.area_width_km)
    particle_h = float(cfg.environment.area_height_km)
    particle_x_max = particle_x_min + particle_w
    particle_y_max = particle_y_min + particle_h

    area_w = particle_w
    area_h = particle_h
    n_uav = int(cfg.fleet.uav_count)
    scan_r = float(cfg.motion.uav_scan_radius_km)

    # If the configured base lies inside the search region, shift it outside to show ferry segments clearly.
    base_x = float(cfg.plot.airport_x_km)
    base_y = float(cfg.plot.airport_y_km)
    if 0.0 <= base_x <= area_w and 0.0 <= base_y <= area_h:
        base_x = -0.22 * area_w
        base_y = -0.18 * area_h

    colors = mpl.colormaps.get_cmap("tab10").resampled(max(2, n_uav))
    fig, ax = plt.subplots(figsize=style.figure_size, dpi=style.dpi)

    # Black frame: target-particle motion range.
    ax.add_patch(
        Rectangle(
            (particle_x_min, particle_y_min),
            particle_w,
            particle_h,
            fill=False,
            lw=2.2,
            ec="black",
            label="Target Particle Motion Range",
        )
    )

    # Base location (red star marker).
    ax.scatter([base_x], [base_y], marker="*", s=260, c="red", edgecolors="black", linewidths=0.8, zorder=6)
    ax.annotate(
        "Base (Wenzhou Longwan Airport)",
        xy=(base_x, base_y),
        xytext=(10, 8),
        textcoords="offset points",
        color="red",
        fontsize=10,
        weight="bold",
    )

    sub_w = area_w / n_uav
    lane_pitch = max(8.0, 2.0 * scan_r * 0.90)

    legend_handles = []
    legend_labels = []

    for uid in range(n_uav):
        c = colors(uid)
        x0 = uid * sub_w
        x1 = (uid + 1) * sub_w

        # Light tint for each sub-region.
        ax.axvspan(x0, x1, facecolor=c, alpha=0.06, lw=0)

        path_xy = _build_serpentine_path(
            x_min=x0,
            x_max=x1,
            y_min=particle_y_min,
            y_max=particle_y_max,
            lane_pitch=lane_pitch,
        )
        if path_xy.shape[0] == 0:
            continue

        entry_x, entry_y = float(path_xy[0, 0]), float(path_xy[0, 1])

        # Ferry segment (dashed).
        ferry_line, = ax.plot(
            [base_x, entry_x],
            [base_y, entry_y],
            linestyle="--",
            lw=1.6,
            color=c,
            alpha=0.95,
            zorder=2,
        )
        ax.annotate(
            "",
            xy=(entry_x, entry_y),
            xytext=(base_x, base_y),
            arrowprops=dict(arrowstyle="->", lw=1.3, color=c),
            zorder=3,
        )

        # Entry point.
        ax.scatter([entry_x], [entry_y], s=52, color=c, edgecolors="white", linewidths=0.8, zorder=6)
        ax.annotate(
            f"Entry Point UAV{uid + 1}",
            xy=(entry_x, entry_y),
            xytext=(7, -14),
            textcoords="offset points",
            fontsize=8.8,
            color=c,
        )

        # Strip-search segment (solid).
        search_line, = ax.plot(path_xy[:, 0], path_xy[:, 1], linestyle="-", lw=2.0, color=c, zorder=4)
        _add_path_arrows(ax, path_xy, color=c, every_n=80)

        legend_handles.extend([search_line, ferry_line])
        legend_labels.extend([f"UAV{uid + 1} Strip-Search Segment", f"UAV{uid + 1} Ferry Segment"])

        ax.text((x0 + x1) / 2.0, area_h * 0.52, f"UAV{uid + 1}", color=c, fontsize=12, alpha=0.65, ha="center")

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()  # Positive y points south.
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_title(style.title, fontsize=15, pad=12)
    ax.set_xlabel("x (km, positive to the east)")
    ax.set_ylabel("y (km, positive to the south)")

    # Axis margins keep base and ferry segments fully visible.
    ax.set_xlim(min(base_x, 0.0) - 0.08 * area_w, area_w + 0.10 * area_w)
    ax.set_ylim(min(base_y, 0.0) - 0.10 * area_h, area_h + 0.08 * area_h)

    # Unified legend, including global elements.
    area_handle = Line2D([0], [0], color="black", lw=2.2)
    base_handle = Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markeredgecolor="black", markersize=12, linestyle="None")
    entry_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=7, linestyle="None")

    all_handles = [area_handle, base_handle, entry_handle] + legend_handles
    all_labels = ["Target Particle Motion Range", "Base (Wenzhou Longwan Airport)", "Entry Point"] + legend_labels
    ax.legend(all_handles, all_labels, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9, frameon=True)

    fig.tight_layout()
    output_path = os.path.join(CURRENT_DIR, style.output_name)
    fig.savefig(output_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    out = plot_figure3_total()
    print(f"Saved Figure 3 to: {out}")
