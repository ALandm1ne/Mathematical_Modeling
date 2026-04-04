"""
主程序入口：负责装配配置、计算引擎、控制器、可视化器与数据记录器。

设计原则：
1. main 只做编排，不承载核心计算细节。
2. GPU 粒子系统与 CPU UAV 控制器解耦。
3. 所有运行参数统一来自 config.py。
"""

import os                       # 拼接路径、检查文件存在性。
import time                     # 统计仿真 wall time。
from collections import deque   # 维护内存监控滑动窗口。

from config import build_default_config           # 构建并初始化全局配置。
from core.simulation_gpu import ParticleSystem    # GPU 端粒子系统。
from core.uav_controller import UAVFleetBuilder   # UAV 机群高层构造器。
from utils.data_manager import DataLogger         # 轨迹与结果导出器。
from visualizer import SimVisualizer              # 仿真过程可视化器。


def _get_rss_gb() -> float:
    """读取当前进程 RSS（常驻内存），单位 GB。"""
    # Linux 的 /proc/self/statm 第二列是 resident pages。
    with open("/proc/self/statm", "r", encoding="utf-8") as f:  # 打开进程内存统计文件。
        parts = f.read().strip().split()                              # 读取并拆成字段列表。
    if len(parts) < 2:                                               # 防御式检查：字段不足则返回 0。
        return 0.0
    rss_pages = int(parts[1])                                        # 第二列是常驻页数。
    page_size = os.sysconf("SC_PAGE_SIZE")                           # 读取系统页大小。
    return (rss_pages * page_size) / (1024.0 ** 3)                   # bytes -> GB。


def _apply_api_demo_mode(cfg, script_dir: str) -> None:
    """
    应用 API 可视化验证模式。

    触发条件：cfg.runtime.api_demo_enable=True
    相关配置：
    - cfg.runtime.api_demo_json_path: 自定义路径 JSON（相对 script_dir）
    - cfg.runtime.api_demo_steps: 仿真最大步数
    - cfg.runtime.api_demo_realtime_visualization: 是否打开实时窗口
    """
    if not cfg.runtime.api_demo_enable:                               # 未开启演示模式则直接返回。
        return

    json_path = cfg.runtime.api_demo_json_path                        # 读取配置中的 JSON 路径。
    if not os.path.isabs(json_path):                                  # 允许配置写相对路径。
        json_path = os.path.join(script_dir, json_path)               # 转成绝对路径。
    if not os.path.exists(json_path):                                 # 防止演示直接失败在文件缺失上。
        raise FileNotFoundError(f"UAV_API_DEMO_JSON not found: {json_path}")

    demo_steps = int(cfg.runtime.api_demo_steps)                     # 演示步数由配置决定。

    cfg.uav_fleet_mode.mode = "custom_paths"                        # 演示模式固定为自定义路径。
    cfg.uav_fleet_mode.custom_paths_json = json_path                 # 指向 JSON 路径。

    # 为验证模式强化可视化可观察性。
    cfg.runtime.realtime_visualization = cfg.runtime.api_demo_realtime_visualization  # 由配置决定是否开窗。
    cfg.runtime.export_simulation_video = True                        # 演示模式固定导出视频。
    cfg.refresh.steps_to_update = 1                                   # 每步刷新，便于看实时变化。
    cfg.simulation.max_steps = max(1, demo_steps)                     # 限制总步数，避免演示过长。

    print("[API-DEMO] enabled")                                      # 打印演示模式标记。
    print(f"[API-DEMO] json={json_path}")                             # 打印所用 JSON 方便追踪。
    print(f"[API-DEMO] max_steps={cfg.simulation.max_steps}")        # 打印步数上限。
    print(
        "[API-DEMO] visual: "                                        # 打印可视化状态摘要。
        f"realtime={cfg.runtime.realtime_visualization}, "
        f"video={cfg.runtime.export_simulation_video}"
    )


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
    script_dir = os.path.dirname(os.path.abspath(__file__))           # 定位 main.py 所在目录。
    cfg = build_default_config(script_dir=script_dir)                 # 构建默认配置与设备信息。
    _apply_api_demo_mode(cfg, script_dir)                             # 若开启演示模式则覆盖配置。
    assert cfg.device_runtime is not None                             # 设备信息必须已经初始化。

    print(f"Using CUDA device: {cfg.device_runtime.gpu_name}")       # 告知当前运行设备。

    # DataLogger 会在初始化时创建带时间戳的运行目录。
    data_logger = DataLogger(cfg)                                     # 负责记录轨迹和导出文件。
    print(f"Results directory: {data_logger.run_results_dir}")        # 打印结果目录，便于快速定位。

    particle_system = ParticleSystem(cfg)                             # 创建 GPU 粒子系统。
    fleet_controller = UAVFleetBuilder.from_default_config(cfg)       # 创建 UAV 机群控制器。
    data_logger.init_uav_trace_fleet(fleet_controller.controllers)    # 初始化轨迹缓存。

    # 可视化初始化阶段读取一次初始密度，用于固定色条范围。
    initial_density = None                                            # 默认不预取密度。
    if cfg.enable_visual_output:                                      # 只有启用可视化才需要初始密度。
        initial_density = particle_system.get_counts_in_grids()       # 供色条 vmax 固定。

    visualizer = SimVisualizer(cfg, data_logger.run_results_dir, initial_density)  # 构造可视化器。

    history_count: list[int] = []                                     # 记录每步剩余粒子数。
    time_elapsed_h = 0.0                                              # 仿真时间（小时）。
    sim_step_counter = 0                                              # 实际记录步数。
    # 用滑动窗口检测“持续增长型”内存风险。
    mem_window = deque(maxlen=cfg.memory.leak_warning_window)         # RSS 滑动窗口。

    print("Starting CUDA search simulation...")                      # 仿真开始提示。
    sim_start_time = time.perf_counter()                              # 记录 wall time 起点。

    for step in range(cfg.simulation.max_steps):                      # 主仿真循环，步数上限受配置控制。
        # 无活跃粒子表示搜索空间已被完全排除，可提前结束。
        if particle_system.active_count <= 0:                         # 粒子已经全部被排除。
            break
        # 无活跃 UAV 时无需继续推进粒子。
        if not fleet_controller.active_positions_u:                   # 没有可用 UAV 就停止。
            break

        # A) Particle motion on GPU.
        particle_system.update_particles()                            # 先移动粒子，再由 UAV 去扫描。

        # B) UAV fleet motion on CPU.
        any_uav_active = fleet_controller.update_all(time_elapsed_h)  # 再推进 UAV 机群。

        # C) Scan and inactivate particles covered by all UAVs moved in this step.
        #    包含“本步刚结束扫描”的 UAV 终点位置，避免终点步漏扫。
        remaining_particles = particle_system.active_count            # 先取当前剩余粒子数。
        for pos_u in fleet_controller.scan_positions_u:               # 对本步所有扫描位置逐个剔除。
            remaining_particles = particle_system.remove_scanned_particles(pos_u)

        # D) Bookkeeping.
        history_count.append(remaining_particles)                     # 记录收敛曲线数据。
        time_elapsed_h += cfg.simulation.dt_h                         # 推进仿真时钟。
        sim_step_counter += 1                                         # 记录逻辑步数。

        data_logger.record_uav_step_trace_fleet(                      # 将当前步的 UAV 状态写入日志。
            step=sim_step_counter,
            time_h=time_elapsed_h,
            controllers=fleet_controller.controllers,
            active_flags=fleet_controller.active_flags,
            remaining_particles=remaining_particles,
        )

        if cfg.enable_visual_output and step % cfg.refresh.steps_to_update == 0:  # 按刷新间隔更新图像。
            visualizer.update(                                                     # 画出当前密度和 UAV 轨迹。
                particle_system=particle_system,
                fleet_controller=fleet_controller,
                data_logger=data_logger,
                elapsed_h=time_elapsed_h,
                remaining_particles=remaining_particles,
            )

        if cfg.memory.enable and step % cfg.memory.monitor_every_steps == 0:      # 低频内存监控。
            # 该监控为轻量级：每 N 步读取一次 RSS，避免影响主循环吞吐。
            rss_gb = _get_rss_gb()                                                # 读取 RSS。
            mem_window.append(rss_gb)                                             # 写入滑动窗口。
            print(f"[MEM] step={step} rss={rss_gb:.2f} GB")                       # 打印当前内存值。
            if len(mem_window) == mem_window.maxlen:                              # 只有窗口满了才做趋势判断。
                # 仅做趋势告警，不中断仿真，便于后续分析与调参。
                growth = mem_window[-1] - mem_window[0]                           # 计算窗口增长量。
                if growth >= cfg.memory.leak_warning_gb:                           # 超阈值则提示风险。
                    print(
                        "[MEM][WARN] memory increased continuously over window: "
                        f"+{growth:.2f} GB (possible leak or oversized frame buffers)."
                    )

        if remaining_particles <= 0:                                          # 粒子耗尽则结束。
            break
        if not any_uav_active:                                                # UAV 全部结束则结束。
            break

    # 仿真循环结束后再刷新一次可视化，确保最终状态被捕获在视频与汇总图中。
    # Final frame to include the true terminal state.
    if cfg.enable_visual_output and history_count:                        # 只有有历史记录时才补最后一帧。
        visualizer.update(                                               # 再画一次终态。
            particle_system=particle_system,
            fleet_controller=fleet_controller,
            data_logger=data_logger,
            elapsed_h=time_elapsed_h,
            remaining_particles=history_count[-1],
        )

    sim_cost = time.perf_counter() - sim_start_time                      # 计算总 wall time。
    print(f"Simulation wall time: {sim_cost:.2f}s")                     # 打印性能结果。

    # 收尾阶段统一导出，避免中途频繁 I/O 干扰计算性能。
    data_logger.export_uav_trace()                                       # 导出轨迹 CSV/Parquet。
    visualizer.finalize()                                                # 关闭视频写入器/交互状态。
    visualizer.save_summary_figure(history_count)                        # 保存收敛曲线。


if __name__ == "__main__":                                              # 仅脚本直接运行时启动主流程。
    main()                                                                # 入口函数。
