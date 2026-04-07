"""
主程序入口：负责装配配置、计算引擎、控制器、可视化器与数据记录器。

设计原则：
1. main 只做编排，不承载核心计算细节。
2. GPU 粒子系统与 CPU UAV 控制器解耦。
3. 所有运行参数统一来自 config.py。
"""

import os                       # 拼接路径、检查文件存在性。
import sys                      # 获取当前 Python 解释器路径。
import subprocess               # 调用图 11 生成脚本。
import time                     # 统计仿真 wall time。
import math                     # 完成阈值计算（向上取整）。
from collections import deque   # 维护内存监控滑动窗口。

from config import build_default_config           # 构建并初始化全局配置。
from core.simulation_gpu import ParticleSystem    # GPU 端粒子系统。
from core.uav_controller import UAVFleetBuilder   # UAV 机群高层构造器。
from core.replanning_engine import ReplanningEngine  # 动态重规划引擎。
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


def _resolve_external_path_source(cfg, script_dir: str) -> str | None:
    """统一解析外置路径来源：收集、冲突检测、绝对路径解析。"""
    sources: list[tuple[str, str]] = []
    cfg_path = (cfg.uav_fleet_mode.custom_paths_json or "").strip()
    if cfg_path:
        sources.append(("uav_fleet_mode.custom_paths_json", cfg_path))

    if cfg.runtime.api_demo_enable:
        demo_path = (cfg.runtime.api_demo_json_path or "").strip()
        if demo_path:
            sources.append(("runtime.api_demo_json_path", demo_path))

    if len(sources) > 1:
        msg = (
            "[PATH][WARN] multiple path sources configured: "
            + ", ".join(f"{name}={value}" for name, value in sources)
            + ". fallback to uav_fleet_mode.custom_paths_json and ignore runtime.api_demo_json_path."
        )
        print(msg)
        sources = [("uav_fleet_mode.custom_paths_json", cfg_path)]

    if not sources:
        if not cfg.uav_fleet_mode.require_external_paths:
            return None
        msg = "[PATH][WARN] no external UAV path file configured."
        if cfg.uav_fleet_mode.missing_path_action == "raise":
            raise ValueError(msg)
        print(msg)
        return None

    _, raw_path = sources[0]
    resolved_path = raw_path
    if cfg.uav_fleet_mode.resolve_relative_to_script_dir and not os.path.isabs(resolved_path):
        resolved_path = os.path.join(script_dir, resolved_path)
    resolved_path = os.path.abspath(resolved_path)

    if not os.path.exists(resolved_path):
        msg = f"[PATH][WARN] external UAV path file not found: {resolved_path}"
        if cfg.uav_fleet_mode.missing_path_action == "raise":
            raise FileNotFoundError(msg)
        print(msg)
        return None

    return resolved_path


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
    bootstrap_mode = (
        cfg.dynamic_replanning.enable
        and cfg.uav_fleet_mode.dynamic_bootstrap_without_external_paths
    )
    resolved_path: str | None = None
    if bootstrap_mode:
        cfg.uav_fleet_mode.require_external_paths = False
        if not cfg.runtime.realtime_visualization:
            cfg.runtime.realtime_visualization = True
            print("[VIS] realtime visualization enabled for dynamic bootstrap inspection")
        print("[PATH] dynamic bootstrap mode enabled: external UAV path JSON will be ignored")
    else:
        resolved_path = _resolve_external_path_source(cfg, script_dir)
        if resolved_path is None:
            print("[PATH][WARN] simulation skipped: unresolved external UAV path source.")
            return
        cfg.uav_fleet_mode.custom_paths_json = resolved_path
    assert cfg.device_runtime is not None                             # 设备信息必须已经初始化。

    print(f"Using CUDA device: {cfg.device_runtime.gpu_name}")       # 告知当前运行设备。
    if resolved_path is not None:
        print(f"[PATH] loaded UAV path file: {resolved_path}")       # 明确提示当前实际使用的模板文件。

    # DataLogger 会在初始化时创建带时间戳的运行目录。
    data_logger = DataLogger(cfg)                                     # 负责记录轨迹和导出文件。
    print(f"Results directory: {data_logger.run_results_dir}")        # 打印结果目录，便于快速定位。

    particle_system = ParticleSystem(cfg)                             # 创建 GPU 粒子系统。
    if bootstrap_mode:
        fleet_controller = UAVFleetBuilder.from_dynamic_bootstrap(cfg)
    else:
        assert resolved_path is not None
        fleet_controller = UAVFleetBuilder.from_custom_json(cfg, resolved_path)  # 创建 UAV 机群控制器。
    print(f"[PATH] loaded UAV count: {len(fleet_controller.controllers)}")  # 明确提示实际加载的 UAV 数量。
    data_logger.init_uav_trace_fleet(fleet_controller.controllers)    # 初始化轨迹缓存。

    # 可视化初始化阶段读取一次初始密度，用于固定色条范围。
    initial_density = None                                            # 默认不预取密度。
    if cfg.enable_visual_output:                                      # 只有启用可视化才需要初始密度。
        initial_density = particle_system.get_counts_in_grids()       # 供色条 vmax 固定。

    visualizer = SimVisualizer(cfg, data_logger.run_results_dir, initial_density)  # 构造可视化器。

    # 初始化动态重规划引擎（如果启用）
    replanning_engine = None
    if cfg.dynamic_replanning.enable:
        replanning_engine = ReplanningEngine(cfg)
        print("[REPLANNING] Dynamic replanning engine initialized (enabled)")
        if bootstrap_mode:
            initial_pending = [(uav.uav_id, "cold_start") for uav in fleet_controller.controllers]
            replanning_engine.process_pending_replans(
                pending_replans=initial_pending,
                fleet_controller=fleet_controller,
                particle_system=particle_system,
                current_step=0,
                elapsed_time_h=0.0,
            )
    else:
        print("[REPLANNING] Dynamic replanning engine disabled")

    history_count: list[int] = []                                     # 记录每步剩余粒子数。
    time_elapsed_h = 0.0                                              # 仿真时间（小时）。
    sim_step_counter = 0                                              # 实际记录步数。
    completion_threshold_particles = max(
        1,
        int(math.ceil(cfg.simulation.n_particles * 0.001)),
    )
    print(
        "[STOP] completion threshold set to "
        f"0.1% of initial particles: {completion_threshold_particles}"
    )
    # 用滑动窗口检测“持续增长型”内存风险。
    mem_window = deque(maxlen=cfg.memory.leak_warning_window)         # RSS 滑动窗口。

    print("Starting CUDA search simulation...")                      # 仿真开始提示。
    sim_start_time = time.perf_counter()                              # 记录 wall time 起点。

    for step in range(cfg.simulation.max_steps):                      # 主仿真循环，步数上限受配置控制。
        # 剩余粒子低于阈值（初始粒子数的 0.1%）即视为搜索完成。
        if particle_system.active_count <= completion_threshold_particles:
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
            remaining_particles = particle_system.remove_scanned_particles(
                pos_u,
                cfg.motion.uav_detection_probability,
            )
        # C.5) Dynamic replanning: after scanning, process pending boundary-triggered replans.
        if replanning_engine and fleet_controller.pending_replans:
            replanning_engine.process_pending_replans(
                pending_replans=fleet_controller.pending_replans,
                fleet_controller=fleet_controller,
                particle_system=particle_system,
                current_step=sim_step_counter,
                elapsed_time_h=time_elapsed_h,
            )
        # D) Bookkeeping（统一时标：本步状态按当前 t_n 记录）。
        history_count.append(remaining_particles)                     # 记录收敛曲线数据。

        data_logger.record_uav_step_trace_fleet(                      # 将当前步的 UAV 状态写入日志。
            step=sim_step_counter,
            time_h=time_elapsed_h,
            controllers=fleet_controller.controllers,
            active_flags=fleet_controller.active_flags,
            remaining_particles=remaining_particles,
        )

        time_elapsed_h += cfg.simulation.dt_h                         # 推进仿真时钟。
        sim_step_counter += 1                                         # 记录逻辑步数。

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

        if remaining_particles <= completion_threshold_particles:              # 达到完成阈值则结束。
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
    
    # 导出动态重规划数据（如果启用）
    if cfg.dynamic_replanning.enable:
        data_logger.export_replanning_events(fleet_controller)           # 导出重规划触发事件。
        data_logger.export_search_strategy(fleet_controller)             # 导出搜索策略对比数据。

    figure11_script = os.path.join(script_dir, "pictures", "11.py")
    figure11_output = os.path.join(script_dir, "pictures", "11.png")
    if os.path.exists(figure11_script):
        try:
            subprocess.run(
                [sys.executable, figure11_script, data_logger.run_results_dir, figure11_output],
                check=True,
            )
        except Exception as exc:
            print(f"[FIG11][WARN] failed to generate figure 11: {exc}")
    else:
        print(f"[FIG11][WARN] figure script not found: {figure11_script}")
    
    final_remaining = history_count[-1] if history_count else particle_system.active_count
    visualizer.save_final_scan_snapshot(                                 # 无论 realtime/video 开关如何都保存终态图。
        particle_system=particle_system,
        fleet_controller=fleet_controller,
        data_logger=data_logger,
        elapsed_h=time_elapsed_h,
        remaining_particles=final_remaining,
    )
    visualizer.finalize()                                                # 关闭视频写入器/交互状态。
    visualizer.save_summary_figure(history_count)                        # 保存收敛曲线。


if __name__ == "__main__":                                              # 仅脚本直接运行时启动主流程。
    main()                                                                # 入口函数。
