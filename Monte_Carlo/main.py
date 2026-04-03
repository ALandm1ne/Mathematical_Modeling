"""
主程序入口：负责装配配置、计算引擎、控制器、可视化器与数据记录器。

设计原则：
1. main 只做编排，不承载核心计算细节。
2. GPU 粒子系统与 CPU UAV 控制器解耦。
3. 所有运行参数统一来自 config.py。
"""

import os
import time
from collections import deque

from config import build_default_config
from core.simulation_gpu import ParticleSystem
from core.uav_controller import UAVController
from utils.data_manager import DataLogger
from visualizer import SimVisualizer


def _get_rss_gb() -> float:
    """读取当前进程 RSS（常驻内存），单位 GB。"""
    # Linux: /proc/self/statm second field is resident pages.
    with open("/proc/self/statm", "r", encoding="utf-8") as f:
        parts = f.read().strip().split()
    if len(parts) < 2:
        return 0.0
    rss_pages = int(parts[1])
    page_size = os.sysconf("SC_PAGE_SIZE")
    return (rss_pages * page_size) / (1024.0 ** 3)


def main() -> None:
    """
    运行一次完整仿真。

    流程：
    1. 构建配置并初始化各模块
    2. 进入仿真循环（粒子更新 -> UAV 更新 -> 扫描剔除 -> 记录）
    3. 根据刷新策略执行可视化
    4. 结束后统一导出轨迹、视频与汇总图
    """
    # 以当前脚本目录为根，保证路径在不同启动目录下都一致。
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = build_default_config(script_dir=script_dir)
    assert cfg.device_runtime is not None

    print(f"Using CUDA device: {cfg.device_runtime.gpu_name}")

    # DataLogger 会在初始化时创建带时间戳的运行目录。
    data_logger = DataLogger(cfg)
    print(f"Results directory: {data_logger.run_results_dir}")

    particle_system = ParticleSystem(cfg)
    uav_controller = UAVController(cfg, uav_id=0)
    data_logger.init_uav_trace(uav_controller)

    # 可视化初始化阶段读取一次初始密度，用于固定色条范围。
    initial_density = None
    if cfg.enable_visual_output:
        initial_density = particle_system.get_counts_in_grids()

    visualizer = SimVisualizer(cfg, data_logger.run_results_dir, initial_density)

    history_count: list[int] = []
    time_elapsed_h = 0.0
    sim_step_counter = 0
    # 用滑动窗口检测“持续增长型”内存风险。
    mem_window = deque(maxlen=cfg.memory.leak_warning_window)

    print("Starting CUDA search simulation...")
    sim_start_time = time.perf_counter()

    for step in range(cfg.simulation.max_steps):
        # 无活跃粒子表示搜索空间已被完全排除，可提前结束。
        if particle_system.active_count <= 0:
            break

        # A) Particle motion on GPU.
        particle_system.update_particles()

        # B) UAV motion on CPU.
        if not uav_controller.update(time_elapsed_h):
            break

        # C) Scan and inactivate particles covered by UAV.
        remaining_particles = particle_system.remove_scanned_particles(uav_controller.position_u)

        # D) Bookkeeping.
        history_count.append(remaining_particles)
        time_elapsed_h += cfg.simulation.dt_h
        sim_step_counter += 1

        data_logger.record_uav_step_trace(
            step=sim_step_counter,
            time_h=time_elapsed_h,
            uav_controller=uav_controller,
            remaining_particles=remaining_particles,
        )

        if cfg.enable_visual_output and step % cfg.refresh.steps_to_update == 0:
            visualizer.update(
                particle_system=particle_system,
                uav_controller=uav_controller,
                data_logger=data_logger,
                elapsed_h=time_elapsed_h,
                remaining_particles=remaining_particles,
            )

        if cfg.memory.enable and step % cfg.memory.monitor_every_steps == 0:
            # 该监控为轻量级：每 N 步读取一次 RSS，避免影响主循环吞吐。
            rss_gb = _get_rss_gb()
            mem_window.append(rss_gb)
            print(f"[MEM] step={step} rss={rss_gb:.2f} GB")
            if len(mem_window) == mem_window.maxlen:
                # 仅做趋势告警，不中断仿真，便于后续分析与调参。
                growth = mem_window[-1] - mem_window[0]
                if growth >= cfg.memory.leak_warning_gb:
                    print(
                        "[MEM][WARN] memory increased continuously over window: "
                        f"+{growth:.2f} GB (possible leak or oversized frame buffers)."
                    )

        if remaining_particles <= 0:
            break

    # Final frame to include the true terminal state.
    if cfg.enable_visual_output and history_count:
        visualizer.update(
            particle_system=particle_system,
            uav_controller=uav_controller,
            data_logger=data_logger,
            elapsed_h=time_elapsed_h,
            remaining_particles=history_count[-1],
        )

    sim_cost = time.perf_counter() - sim_start_time
    print(f"Simulation wall time: {sim_cost:.2f}s")

    # 收尾阶段统一导出，避免中途频繁 I/O 干扰计算性能。
    data_logger.export_uav_trace()
    visualizer.finalize()
    visualizer.save_summary_figure(history_count)


if __name__ == "__main__":
    main()
