"""全局配置中心：定义参数、派生常量、校验规则与设备策略。"""

import math
import os
from dataclasses import dataclass, field
from datetime import datetime

import torch


@dataclass
class SimulationTuningConfig:
    n_particles: int = 10_000_000  # 初始粒子总数（潜在目标样本量）
    dt_h: float = 0.01             # 仿真时间步长，单位小时
    max_steps: int = 4000          # 最大仿真步数上限


@dataclass
class EnvironmentConfig:
    area_width_km: float = 306.0   # 搜索区域宽度x（经向），单位 km
    area_height_km: float = 444.0  # 搜索区域高度y（纬向），单位 km


@dataclass
class MotionConfig:
    target_speed_km_h: float = 30.0    # 目标机动速度，单位 km/h
    uav_speed_km_h: float = 150.0      # 无人机巡航速度，单位 km/h
    uav_scan_radius_km: float = 20.0   # 无人机传感器扫描半径，单位 km
    uav_detection_probability: float = 0.1  # UAV 扫描命中后的剔除概率，范围 [0, 1]


@dataclass
class UAVFleetModeConfig:
    """UAV 机群路径规划模式配置。"""
    custom_paths_json: str | None = None       # 自定义路径 JSON 文件路径（可相对 script_dir）
    strict_path_validation: bool = True        # 是否启用强校验：几何/字段不合法直接报错
    require_external_paths: bool = True        # 是否强制要求提供外置路径文件
    missing_path_action: str = "warn_and_exit"  # 缺失路径时行为：warn_and_exit/raise
    path_source_conflict_action: str = "warn_and_exit"  # API 路径与配置路径冲突时行为
    resolve_relative_to_script_dir: bool = True  # 路径是否按 script_dir 解析相对路径


@dataclass
class RuntimeSwitchesConfig:
    """运行开关与演示模式配置。"""
    realtime_visualization: bool = False                # 是否开启实时交互窗口
    export_simulation_video: bool = True               # 是否导出仿真视频（mp4/gif）
    export_uav_trajectory: bool = False                 # 是否导出 UAV 轨迹文件

    # API 演示模式：仅作为“额外路径来源”，不改变路径模式语义。
    api_demo_enable: bool = True                      # 是否启用演示路径来源
    api_demo_json_path: str = "config_templates/uav_paths.json"  # 演示路径 JSON（相对 script_dir）


@dataclass
class RefreshPolicyConfig:
    steps_to_update: int = 10  # 可视化刷新间隔（每 N 步刷新一次）


@dataclass
class ExportConfig:
    trajectory_export_format: str = "both"            # 轨迹导出格式：csv/parquet/both
    trajectory_output_basename: str = "uav_trajectory"  # 轨迹输出文件名（不含扩展名）
    trajectory_parquet_compression: str = "zstd"      # Parquet 压缩算法
    trajectory_include_extended: bool = True           # 是否包含扩展字段（角度/转向/剩余粒子）


@dataclass
class VideoConfig:
    output_filename: str = "simulation.mp4"  # 视频输出文件名
    fps: int = 10                              # 视频帧率
    dpi: int = 240                             # 视频导出 DPI（会受安全阈值自动降级）
    max_frame_pixels: int = 16_000_000        # 单帧像素安全上限，防止内存爆炸


@dataclass
class MemoryMonitorConfig:
    enable: bool = True               # 是否开启运行期内存监控
    monitor_every_steps: int = 30     # 每 N 步打印一次 RSS
    leak_warning_window: int = 30     # 泄漏趋势判定窗口大小（采样点数量）
    leak_warning_gb: float = 2.0      # 窗口内增长超过该值触发告警（GB）


@dataclass
class FigureConfig:
    main_fig_size: tuple = (12, 12)    # 主仿真图尺寸（英寸）
    summary_fig_size: tuple = (12, 6)  # 收敛曲线图尺寸（英寸）
    debug_text_x: float = 0.02         # 状态文本在画布中的 x 位置（0-1）
    debug_text_y: float = 0.075        # 状态文本在画布中的 y 位置（0-1）


@dataclass
class PathConfig:
    script_dir: str                                # 脚本所在目录（作为输出根路径基准）
    results_root_name: str = "results"            # 结果根目录名称
    run_timestamp_format: str = "%Y%m%d_%H%M%S"  # 结果目录时间戳格式


@dataclass
class NumericCoreConfig:
    scale: int = 1000          # 定点缩放：1 km = 1000 units
    grid_size_km: float = 2.0  # 热力图网格边长，单位 km


@dataclass
class DebugFlagsConfig:
    use_active_index_cache: bool = True  # 是否启用活跃索引缓存（性能优化开关）


@dataclass
class DevicePolicyConfig:
    require_cuda: bool = True  # 是否强制要求 CUDA 可用


@dataclass
class DerivedConfig:
    area_width_u: int = 0       # 区域宽度（整数单位）
    area_height_u: int = 0      # 区域高度（整数单位）
    uav_scan_radius_u: int = 0  # 扫描半径（整数单位）
    uav_scan_radius_u2: int = 0 # 扫描半径平方（距离判断用）
    uav_step_u: int = 0         # 无人机每步位移（整数单位）
    particle_step_u: int = 0    # 目标粒子每步位移（整数单位）
    grid_size_u: int = 0        # 热力图网格大小（整数单位）
    n_x_bins: int = 0           # 热力图 x 方向网格数
    n_y_bins: int = 0           # 热力图 y 方向网格数
    pi: float = math.pi         # 圆周率常量
    two_pi: float = 2.0 * math.pi  # 2π 常量


@dataclass
class DeviceRuntime:
    device: torch.device  # 当前运行设备对象（cuda/cpu）
    gpu_name: str         # 设备名称（显卡型号或 cpu）


@dataclass
class AppConfig:
    simulation: SimulationTuningConfig               # 仿真核心参数
    environment: EnvironmentConfig                   # 区域参数
    motion: MotionConfig                             # 运动参数
    uav_fleet_mode: UAVFleetModeConfig              # UAV 路径规划模式配置
    runtime: RuntimeSwitchesConfig                  # 运行时功能开关
    refresh: RefreshPolicyConfig                    # 刷新策略
    export: ExportConfig                             # 数据导出策略
    video: VideoConfig                               # 视频导出策略
    memory: MemoryMonitorConfig                      # 内存监控策略
    figure: FigureConfig                             # 图形尺寸与布局参数
    paths: PathConfig                                # 路径策略
    numeric: NumericCoreConfig                       # 数值核心参数（定点缩放等）
    debug: DebugFlagsConfig                          # 调试/性能开关
    device_policy: DevicePolicyConfig                # 设备策略
    derived: DerivedConfig = field(default_factory=DerivedConfig)  # 派生参数缓存
    device_runtime: DeviceRuntime | None = None      # 运行时设备信息

    @property
    def enable_visual_output(self) -> bool:
        """只要实时显示或视频导出任一开启，即启用可视化流程。"""
        return self.runtime.realtime_visualization or self.runtime.export_simulation_video

    @property
    def results_root_dir(self) -> str:
        """返回结果根目录绝对路径。"""
        return os.path.join(self.paths.script_dir, self.paths.results_root_name)

    def recompute_derived(self) -> None:
        """根据源参数重算派生常量（统一整数网格体系）。"""
        d = self.derived
        e = self.environment
        m = self.motion
        s = self.simulation
        n = self.numeric

        d.area_width_u = int(round(e.area_width_km * n.scale))
        d.area_height_u = int(round(e.area_height_km * n.scale))
        d.uav_scan_radius_u = int(round(m.uav_scan_radius_km * n.scale))
        d.uav_scan_radius_u2 = d.uav_scan_radius_u * d.uav_scan_radius_u
        d.uav_step_u = int(round(m.uav_speed_km_h * s.dt_h * n.scale))
        d.particle_step_u = int(round(m.target_speed_km_h * s.dt_h * n.scale))
        d.grid_size_u = int(round(n.grid_size_km * n.scale))
        d.n_x_bins = d.area_width_u // d.grid_size_u + 1
        d.n_y_bins = d.area_height_u // d.grid_size_u + 1

    def validate(self) -> None:
        """执行配置合法性校验，尽早暴露参数错误。"""
        if self.simulation.n_particles <= 0:
            raise ValueError("n_particles must be > 0")
        if self.simulation.dt_h <= 0:
            raise ValueError("dt_h must be > 0")
        if self.simulation.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if self.numeric.scale <= 0:
            raise ValueError("scale must be > 0")
        if self.numeric.grid_size_km <= 0:
            raise ValueError("grid_size_km must be > 0")
        if self.motion.uav_scan_radius_km <= 0:
            raise ValueError("uav_scan_radius_km must be > 0")
        if not (0.0 <= self.motion.uav_detection_probability <= 1.0):
            raise ValueError("uav_detection_probability must be in [0, 1]")
        if self.export.trajectory_export_format.lower() not in {"csv", "parquet", "both"}:
            raise ValueError("trajectory_export_format must be one of csv/parquet/both")
        if self.uav_fleet_mode.missing_path_action not in {"warn_and_exit", "raise"}:
            raise ValueError("missing_path_action must be one of warn_and_exit/raise")
        if self.uav_fleet_mode.path_source_conflict_action not in {"warn_and_exit", "raise"}:
            raise ValueError("path_source_conflict_action must be one of warn_and_exit/raise")
        if self.uav_fleet_mode.require_external_paths and self.uav_fleet_mode.missing_path_action not in {
            "warn_and_exit",
            "raise",
        }:
            raise ValueError("require_external_paths requires valid missing_path_action")

    def configure_device(self) -> None:
        """按策略检测并配置运行设备（cuda/cpu）。"""
        if self.device_policy.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available.")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
        else:
            device = torch.device("cpu")
            gpu_name = "cpu"

        self.device_runtime = DeviceRuntime(device=device, gpu_name=gpu_name)


def build_default_config(script_dir: str, require_cuda_override: bool | None = None) -> AppConfig:
    """构建默认配置并完成：派生计算 + 校验 + 设备配置。"""
    cfg = AppConfig(
        simulation=SimulationTuningConfig(),
        environment=EnvironmentConfig(),
        motion=MotionConfig(),
        uav_fleet_mode=UAVFleetModeConfig(),
        runtime=RuntimeSwitchesConfig(),
        refresh=RefreshPolicyConfig(),
        export=ExportConfig(),
        video=VideoConfig(),
        memory=MemoryMonitorConfig(),
        figure=FigureConfig(),
        paths=PathConfig(script_dir=script_dir),
        numeric=NumericCoreConfig(),
        debug=DebugFlagsConfig(),
        device_policy=DevicePolicyConfig(),
    )
    if require_cuda_override is not None:
        cfg.device_policy.require_cuda = bool(require_cuda_override)
    cfg.recompute_derived()
    cfg.validate()
    cfg.configure_device()
    return cfg


def build_run_timestamp(cfg: AppConfig) -> str:
    """按配置格式生成本次运行时间戳。"""
    return datetime.now().strftime(cfg.paths.run_timestamp_format)
