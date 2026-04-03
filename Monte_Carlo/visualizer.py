"""可视化模块：负责热力图刷新、状态面板、视频导出与汇总图保存。"""

import os
import math

import matplotlib
import matplotlib.animation as animation
import numpy as np


class SimVisualizer:
    """仿真可视化器。"""

    def __init__(self, cfg, run_results_dir: str, initial_density: np.ndarray | None = None):
        self.cfg = cfg
        self.run_results_dir = run_results_dir
        self.initial_density = initial_density

        self.fig = None
        self.window = None
        self.im = None
        self.status_text = None
        self.uav_dot = None
        self.uav_path = None
        self.video_writer = None
        self.video_output_path = None
        self.effective_video_dpi = self.cfg.video.dpi

        # 根据运行模式切换后端：实时窗口用 QtAgg，无界面模式用 Agg。
        backend = "QtAgg" if self.cfg.runtime.realtime_visualization else "Agg"
        matplotlib.use(backend)
        import matplotlib.pyplot as plt

        self.plt = plt

        if self.cfg.enable_visual_output:
            self._init_visual_elements()
        else:
            print("Realtime visualization disabled.")

    def _init_visual_elements(self):
        """初始化主图、热力图、轨迹线、状态面板与视频写入器。"""
        d = self.cfg.derived

        if self.cfg.runtime.realtime_visualization:
            self.plt.ion()

        self.fig, self.window = self.plt.subplots(figsize=self.cfg.figure.main_fig_size)
        self.fig.subplots_adjust(left=0.24)

        # 固定 vmax 可减少刷新时色条跳动，便于观察密度变化趋势。
        fixed_vmax = 1.0
        if self.initial_density is not None:
            fixed_vmax = max(1.0, float(np.max(self.initial_density)))

        self.im = self.window.imshow(
            np.zeros((d.n_y_bins, d.n_x_bins), dtype=np.float32),
            extent=(0, self.cfg.environment.area_width_km, 0, self.cfg.environment.area_height_km),
            origin="lower",
            cmap="coolwarm",
            animated=True,
            vmin=0,
            vmax=fixed_vmax,
        )
        self.window.figure.colorbar(self.im, ax=self.window, label="Particle Density (particles/grid)")
        (self.uav_dot,) = self.window.plot([], [], "ro", markersize=3, label="UAV")
        (self.uav_path,) = self.window.plot([], [], color="cyan", linewidth=1.2, alpha=0.9, label="UAV Path")
        self.window.set_title("CUDA Particle Density (Interval Refresh)")
        self.window.set_xlabel("X (km)")
        self.window.set_ylabel("Y (km)")
        self.window.legend()

        self.status_text = self.fig.text(
            self.cfg.figure.debug_text_x,
            self.cfg.figure.debug_text_y,
            "",
            transform=self.fig.transFigure,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="gray"),
        )

        if self.cfg.runtime.export_simulation_video:
            # 帧像素过大时自动降级 DPI，避免导出阶段内存暴涨。
            fig_w_in, fig_h_in = self.cfg.figure.main_fig_size
            frame_pixels = int(fig_w_in * self.cfg.video.dpi) * int(fig_h_in * self.cfg.video.dpi)
            if frame_pixels > self.cfg.video.max_frame_pixels:
                scale = math.sqrt(self.cfg.video.max_frame_pixels / float(frame_pixels))
                self.effective_video_dpi = max(72, int(self.cfg.video.dpi * scale))
                print(
                    "Warning: video frame is too large; auto-reducing DPI "
                    f"from {self.cfg.video.dpi} to {self.effective_video_dpi}."
                )

            video_basename = os.path.basename(self.cfg.video.output_filename)
            if animation.writers.is_available("ffmpeg"):
                self.video_output_path = os.path.join(self.run_results_dir, video_basename)
                self.video_writer = animation.FFMpegWriter(fps=self.cfg.video.fps, bitrate=2400)
            else:
                self.video_output_path = os.path.join(
                    self.run_results_dir,
                    video_basename.rsplit(".", 1)[0] + ".gif",
                )
                self.video_writer = animation.PillowWriter(fps=max(1, min(self.cfg.video.fps, 20)))

            self.video_writer.setup(self.fig, self.video_output_path, dpi=self.effective_video_dpi)
            print(f"Video export enabled: {self.video_output_path}")

    def update(self, particle_system, uav_controller, data_logger, elapsed_h: float, remaining_particles: int) -> None:
        """刷新一帧：密度图、UAV 位置/轨迹、状态文本、视频帧。"""
        if not self.cfg.enable_visual_output:
            return

        assert self.im is not None
        assert self.uav_dot is not None
        assert self.uav_path is not None
        assert self.fig is not None

        # 该调用包含 GPU->CPU 回传，是可视化路径中的主要开销点。
        density_matrix = particle_system.get_counts_in_grids()
        self.im.set_data(density_matrix)

        x_km, y_km = uav_controller.position_km()
        self.uav_dot.set_data([x_km], [y_km])
        self.uav_path.set_data(data_logger.uav_traj_x_km, data_logger.uav_traj_y_km)

        self._update_status_panel(uav_controller, elapsed_h, remaining_particles)
        self.fig.canvas.draw()

        if self.cfg.runtime.realtime_visualization:
            self.plt.pause(0.001)

        if self.video_writer is not None:
            self.video_writer.grab_frame()

    def _update_status_panel(self, uav_controller, elapsed_h: float, remaining_particles: int) -> None:
        """更新左侧状态面板文本。"""
        assert self.status_text is not None
        x_km, y_km = uav_controller.position_km()
        remaining_ratio = (remaining_particles / self.cfg.simulation.n_particles) * 100.0
        self.status_text.set_text(
            f"UAV: ({x_km:.2f}, {y_km:.2f}) km\n"
            f"Angle: {uav_controller.angle_deg():.1f} deg\n"
            f"Time: {elapsed_h:.2f} h\n"
            f"Particles: {remaining_particles}/{self.cfg.simulation.n_particles} ({remaining_ratio:.2f}%)"
        )

    def save_summary_figure(self, history_count: list[int]) -> None:
        """保存收敛曲线图，并在末端标注终点坐标。"""
        self.plt.figure(figsize=self.cfg.figure.summary_fig_size)
        self.plt.plot(history_count)

        if history_count:
            end_x = len(history_count) - 1
            end_y = history_count[-1]
            self.plt.scatter([end_x], [end_y], color="red", s=28, zorder=3)
            self.plt.annotate(
                f"({end_x}, {end_y})",
                xy=(end_x, end_y),
                xytext=(8, 8),
                textcoords="offset points",
                color="black",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="gray"),
            )

        self.plt.title("Remaining Potential Target Particles Over Time")
        self.plt.xlabel("Time Steps")
        self.plt.ylabel("Particle Count")
        self.plt.xlim(left=0)
        self.plt.ylim(bottom=0)
        self.plt.grid(True)

        # 无论实时模式与否都保存图片，保证结果可复查。
        out_file = os.path.join(self.run_results_dir, "remaining_particles.png")
        self.plt.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"Saved summary figure to {out_file}")

        if self.cfg.runtime.realtime_visualization:
            self.plt.show()

    def finalize(self) -> None:
        """结束可视化：关闭视频写入器并处理交互模式收尾。"""
        if self.video_writer is not None:
            self.video_writer.finish()
            print(f"Simulation video exported: {self.video_output_path}")

        if self.cfg.runtime.realtime_visualization:
            self.plt.ioff()
