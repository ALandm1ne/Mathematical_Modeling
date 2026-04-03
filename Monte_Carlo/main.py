'''
在现代军事活动中，无人侦察机的应用日益广泛，特别是在军事冲突中，利用无人机追踪敌方机动目标，可以为本方提供预警或者指引本方武器精准打击对方目标。
2025年长春航空展上，国产BZK-005中高空远程无人侦察机BZK-005正式亮相，它能够在18000米的高空持续飞行40小时，搭载合成孔径雷达可以从18公里的高度看清地面车牌号，配备30倍光学变焦吊舱。

假设已知敌方机动目标在海上一定范围内，如何组织本方多架BZK-005无人机进行协同搜索？
    1、请建立数学模型，根据目标大致范围、BZK-005出发点等指标，确定调用多少架无人机、每架无人机的飞行路线。
    2、假设已知目标为一艘长182米、宽24米的舰船，在海上一片矩形区域内以每小时30公里速度机动(不离开此区域)，该区域左上角顶点坐标(124E, 25N)，右下角顶点坐标(127E, 21N)。
从温州龙湾机场起飞2架BZK-005无人机，请利用你的模型计算出最少需要多长时间能完成搜索，并给出每架无人机的搜索路径。如果希望在10小时内找到目标，最少需要多少架BZK-005无人机？

注：题目中未提及的变量、参数，可以根据实际情况自行设置；如果找不到参数，可根据类似事物自行推定。
'''

import math
import time
import csv
import os
from datetime import datetime

import matplotlib
import matplotlib.animation as animation
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

# ------------------------------------------全局变量------------------------------------------

# --- 参数设置 ---
AREA_WIDTH_KM = 306.0  # 模拟区域宽度，单位 km（经度方向跨度）
AREA_HEIGHT_KM = 444.0  # 模拟区域高度，单位 km（纬度方向跨度）
N_PARTICLES = 10000000  # 初始粒子数，表示潜在目标样本数量
TARGET_SPEED_KM = 30.0  # 目标运动速度，单位 km/h
UAV_SPEED_KM = 150.0  # 无人机巡航速度，单位 km/h
UAV_SCAN_RADIUS_KM = 20.0  # 无人机传感器扫描半径，单位 km
DT = 0.01  # 仿真时间步长，单位小时
MAX_STEPS = 4000  # 最大仿真步数

# --- 定点缩放设置 ---
# 用整数网格单位替代浮点坐标：1 km -> 1000 单位（1 单位=1 m）
SCALE = 1000    # 坐标缩放系数
AREA_WIDTH_U = int(round(AREA_WIDTH_KM * SCALE))    # 区域宽度（整数单位）
AREA_HEIGHT_U = int(round(AREA_HEIGHT_KM * SCALE))  # 区域高度（整数单位）
UAV_SCAN_RADIUS_U = int(round(UAV_SCAN_RADIUS_KM * SCALE))  # 扫描半径（整数单位）
UAV_SCAN_RADIUS_U2 = UAV_SCAN_RADIUS_U * UAV_SCAN_RADIUS_U  # 扫描半径的平方（整数单位）

# --- UAV 参数 ---
UAV_STEP_U = int(round(UAV_SPEED_KM * DT * SCALE))  # 无人机每步移动距离（整数单位）
particle_step_u = int(round(TARGET_SPEED_KM * DT * SCALE))  # 目标粒子每步移动距离（整数单位）

# --- 数值常量 ---
TWO_PI = 2.0 * math.pi
PI = math.pi

# --------- 绘制设置 ---------

# --- 设备设置（仅支持 CUDA） ---
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required but not available.')

DEVICE = torch.device('cuda')
GPU_NAME = torch.cuda.get_device_name(0)
print(f'Using CUDA device: {GPU_NAME}')

STEPS_TO_UPDATE = 30  # 每隔多少步更新一次绘图（调整以平衡性能和实时性）
GRID_SIZE_U = 2 * SCALE  # 热力图网格大小（2km x 2km）
N_X_BINS = AREA_WIDTH_U // GRID_SIZE_U + 1  # x 方向网格数
N_Y_BINS = AREA_HEIGHT_U // GRID_SIZE_U + 1  # y 方向网格数

# --- 调试文本位置（相对于 figure 坐标系，0-1 之间） ---
DEBUG_TEXT_X = 0.02  # 文本左侧距离 figure 左边界
DEBUG_TEXT_Y = 0.075  # 文本顶部距离 figure 下边界

# 实时可视化开关：
# True  -> 开启实时窗口
# False -> 关闭实时窗口（适合 SSH/无图形环境）
REALTIME_VISUALIZATION = False

# 仿真视频导出开关（推荐在无界面模式下使用）
EXPORT_SIMULATION_VIDEO = True
VIDEO_OUTPUT_FILENAME = 'simulation.mp4'
VIDEO_FPS = 20
VIDEO_DPI = 120

# 显示窗口尺寸参数（单位：英寸）
MAIN_FIG_SIZE = (12, 12)      # 主仿真窗口尺寸
SUMMARY_FIG_SIZE = (10, 5)   # 收敛曲线窗口尺寸

# UAV 轨迹导出设置
EXPORT_UAV_TRAJECTORY = True
# 支持: 'csv' / 'parquet' / 'both'
UAV_TRAJECTORY_EXPORT_FORMAT = 'both'
UAV_TRAJECTORY_OUTPUT_BASENAME = 'uav_trajectory'
UAV_TRAJECTORY_PARQUET_COMPRESSION = 'zstd'  # 可选: 'zstd'/'snappy'/'gzip'/None
# True: 导出扩展字段（angle_deg/is_turning/remaining_particles）
# False: 仅导出基础字段（step/time_h/x_km/y_km）
UAV_TRAJECTORY_INCLUDE_EXTENDED = True

# 性能优化开关：启用后复用活跃粒子索引缓存，减少每步 nonzero 开销
USE_ACTIVE_INDEX_CACHE = True

ENABLE_VISUAL_OUTPUT = REALTIME_VISUALIZATION or EXPORT_SIMULATION_VIDEO

# 输出路径统一管理：所有产物写入 main.py 同目录/results/<timestamp>/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT_DIR = os.path.join(SCRIPT_DIR, 'results')
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_RESULTS_DIR = os.path.join(RESULTS_ROOT_DIR, RUN_TIMESTAMP)
os.makedirs(RUN_RESULTS_DIR, exist_ok=True)
print(f'Results directory: {RUN_RESULTS_DIR}')

# 根据开关选择后端：关闭实时可视化时使用无界面 Agg 后端
matplotlib.use('QtAgg' if REALTIME_VISUALIZATION else 'Agg')
import matplotlib.pyplot as plt

# ------------------------------------------初始化------------------------------------------

def init_particles_on_cuda(num_particles: int):
    """
    在 GPU 上初始化目标粒子。

    采用分层均匀采样：
    1. 先把区域按网格划分
    2. 每个网格内加随机抖动（jitter）
    3. 坐标缩放到整数单位，避免长期累计浮点误差

    :param num_particles: 粒子总数
    :return:
        p_locs: shape=(N,2) 的 int32 坐标张量（CUDA）
        p_angles: shape=(N,) 的 float32 角度张量（CUDA）
        p_active: shape=(N,) 的 bool 活跃掩码（CUDA）
    """
    # 让网格宽高比与区域宽高比接近，减少采样偏斜
    grid_nx = max(1, int(math.sqrt(num_particles * AREA_WIDTH_U / AREA_HEIGHT_U)))
    grid_ny = int(math.ceil(num_particles / grid_nx))

    # 每个粒子先绑定到一个网格单元
    cell_indices = torch.arange(num_particles, device=DEVICE, dtype=torch.int64)
    cell_x = cell_indices.remainder(grid_nx).to(torch.float32)
    cell_y = torch.div(cell_indices, grid_nx, rounding_mode='floor').to(torch.float32)

    # 在网格内加入随机偏移，使初始密度更均匀且无明显条纹
    jitter_x = torch.rand(num_particles, device=DEVICE)
    jitter_y = torch.rand(num_particles, device=DEVICE)

    # 把归一化坐标映射到整数物理坐标系
    x = torch.round(((cell_x + jitter_x) / float(grid_nx)) * AREA_WIDTH_U)
    y = torch.round(((cell_y + jitter_y) / float(grid_ny)) * AREA_HEIGHT_U)

    x = torch.clamp(x, 0, AREA_WIDTH_U).to(torch.int32)
    y = torch.clamp(y, 0, AREA_HEIGHT_U).to(torch.int32)

    p_locs = torch.stack((x, y), dim=1)
    p_angles = torch.rand(num_particles, device=DEVICE, dtype=torch.float32) * TWO_PI
    p_active = torch.ones(num_particles, device=DEVICE, dtype=torch.bool)
    return p_locs, p_angles, p_active


particle_locations, particle_angles, particle_active_mask = init_particles_on_cuda(N_PARTICLES)

# 活跃粒子索引缓存：默认全活跃，无需首次 nonzero
active_idx_cache = torch.arange(N_PARTICLES, device=DEVICE, dtype=torch.int64)

# --- 初始化无人机（仍保留 CPU 标量逻辑） ---
uav_angle = 0.5 * np.pi  # 无人机当前航向角（弧度），0.5π 表示向上
is_uav_turning = False  # 无人机是否处于转向过程
is_uav_turning_clockwise = False  # 转向方向：True 顺时针，False 逆时针
uav_turning_angle_each = 0.0  # 转向过程中每步变化角
uav_turn_step_remain = 0  # 本次转向剩余步数

uav_pos_u = np.array([UAV_SCAN_RADIUS_U, 0], dtype=np.int64)  # 左下角附近起飞
uav_kpt_from__x_u = 0  # 当前转向段起点 x
uav_kpt_from__y_u = 0  # 当前转向段起点 y
uav_kpt_to__x_u = 0  # 当前转向段终点 x
uav_kpt_to__y_u = 0  # 当前转向段终点 y

history_count = []  # 每步剩余活跃粒子数（用于收敛曲线）
time_elapsed = 0.0  # 累计仿真时间（小时）

uav_traj_x_km = [uav_pos_u[0] / SCALE]  # 无人机轨迹 x（km）
uav_traj_y_km = [uav_pos_u[1] / SCALE]  # 无人机轨迹 y（km）

# 每步真实轨迹记录（与可视化刷新频率解耦）
uav_step_trace = {
    'step': [],
    'time_h': [],
    'x_km': [],
    'y_km': [],
    'angle_deg': [],
    'is_turning': [],
    'remaining_particles': [],
}
sim_step_counter = 0


def record_uav_step_trace(step: int, remaining_particles: int):
    """
    记录无人机单步真实轨迹点。

    说明：
    1. 该记录与可视化刷新频率无关，每个仿真步都会追加
    2. 同时维护用于绘图的轨迹列表，确保几何轨迹来自真实步进
    """
    uav_x_km = uav_pos_u[0] / SCALE
    uav_y_km = uav_pos_u[1] / SCALE
    uav_angle_deg = (uav_angle * 180.0 / np.pi) % 360.0

    uav_step_trace['step'].append(step)
    uav_step_trace['time_h'].append(time_elapsed)
    uav_step_trace['x_km'].append(uav_x_km)
    uav_step_trace['y_km'].append(uav_y_km)
    uav_step_trace['angle_deg'].append(uav_angle_deg)
    uav_step_trace['is_turning'].append(bool(is_uav_turning))
    uav_step_trace['remaining_particles'].append(int(remaining_particles))

    # 轨迹绘图数据按“每步真实点”累计
    uav_traj_x_km.append(uav_x_km)
    uav_traj_y_km.append(uav_y_km)


def _get_trajectory_fieldnames(include_extended: bool):
    base_fields = ['step', 'time_h', 'x_km', 'y_km']
    if not include_extended:
        return base_fields
    return base_fields + ['angle_deg', 'is_turning', 'remaining_particles']


def export_uav_trace(base_name: str, export_format: str, include_extended: bool):
    """
    导出 UAV 真实轨迹。

    export_format 支持：
    - 'csv': 仅导出 CSV
    - 'parquet': 仅导出 Parquet
    - 'both': 同时导出 Parquet 和 CSV
    """
    if len(uav_step_trace['step']) == 0:
        print('UAV trajectory export skipped: no recorded steps.')
        return

    fieldnames = _get_trajectory_fieldnames(include_extended)

    def write_csv(path: str):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            n_rows = len(uav_step_trace['step'])
            for i in range(n_rows):
                row = {
                    'step': uav_step_trace['step'][i],
                    'time_h': uav_step_trace['time_h'][i],
                    'x_km': uav_step_trace['x_km'][i],
                    'y_km': uav_step_trace['y_km'][i],
                    'angle_deg': uav_step_trace['angle_deg'][i],
                    'is_turning': uav_step_trace['is_turning'][i],
                    'remaining_particles': uav_step_trace['remaining_particles'][i],
                }
                writer.writerow({k: row[k] for k in fieldnames})

        file_size = os.path.getsize(path)
        print(f'UAV trajectory exported: {path} ({n_rows} rows, {file_size} bytes)')

    def write_parquet(path: str):
        table_data = {k: uav_step_trace[k] for k in fieldnames}
        table = pa.table(table_data)
        pq.write_table(table, path, compression=UAV_TRAJECTORY_PARQUET_COMPRESSION)

        n_rows = len(uav_step_trace['step'])
        file_size = os.path.getsize(path)
        print(f'UAV trajectory exported: {path} ({n_rows} rows, {file_size} bytes)')

    fmt = export_format.strip().lower()
    csv_path = os.path.join(RUN_RESULTS_DIR, f'{base_name}.csv')
    parquet_path = os.path.join(RUN_RESULTS_DIR, f'{base_name}.parquet')

    if fmt == 'csv':
        write_csv(csv_path)
        return

    if fmt == 'parquet':
        write_parquet(parquet_path)
        return

    if fmt == 'both':
        write_parquet(parquet_path)
        write_csv(csv_path)
        return

    raise ValueError(
        f'Unknown UAV_TRAJECTORY_EXPORT_FORMAT={export_format}. '
        "Use one of: 'csv', 'parquet', 'both'."
    )

# ------------------------------------------移动控制相关函数------------------------------------------

def update_particles(p_locs, p_angles, active_idx, step_u):
    """
    按当前速度和方向更新活跃粒子，并在边界处做反弹处理。

    说明：
    1. 每个 step 都为活跃粒子重置随机方向
    2. 用向量化方式在 GPU 上计算位移
    3. 越界时按边界法线反射角度
    4. 最后将坐标裁剪回合法区域
    """
    if active_idx.numel() == 0:
        return

    # 每个活跃粒子本步随机朝向（均匀分布在 [0, 2π)）
    new_angles = torch.rand(active_idx.numel(), device=DEVICE, dtype=torch.float32) * TWO_PI
    # new_angles = p_angles

    # 位移投影（先用浮点算投影，再四舍五入到整数网格）
    dx = torch.round(step_u * torch.cos(new_angles)).to(torch.int32)
    dy = torch.round(step_u * torch.sin(new_angles)).to(torch.int32)

    # 仅提取活跃粒子进行更新，避免对失活粒子做无效计算
    p_active = p_locs[active_idx]
    p_active[:, 0] += dx
    p_active[:, 1] += dy

    # 边界检查：左右边界
    mask_x = (p_active[:, 0] < 0) | (p_active[:, 0] > AREA_WIDTH_U)
    # 边界检查：上下边界
    mask_y = (p_active[:, 1] < 0) | (p_active[:, 1] > AREA_HEIGHT_U)

    # 边界反射：左右边界对应 x 法线反射
    if mask_x.any():
        new_angles[mask_x] = (PI - new_angles[mask_x]).remainder(TWO_PI)
    # 边界反射：上下边界对应 y 法线反射
    if mask_y.any():
        new_angles[mask_y] = (-new_angles[mask_y]).remainder(TWO_PI)

    # 坐标裁剪回合法区域，避免粒子停留在边界外
    p_active[:, 0] = torch.clamp(p_active[:, 0], 0, AREA_WIDTH_U)
    p_active[:, 1] = torch.clamp(p_active[:, 1], 0, AREA_HEIGHT_U)

    # 写回主张量
    p_locs[active_idx] = p_active
    p_angles[active_idx] = new_angles


def is_uav_up() -> bool:
    """当前航向是否属于“向上扫描”半平面。"""
    return (uav_angle > 0) and (uav_angle < np.pi)


def is_uav_down() -> bool:
    """当前航向是否属于“向下扫描”半平面。"""
    return (uav_angle > np.pi) and (uav_angle < 2 * np.pi)


def is_uav_outside_top_edge() -> bool:
    """是否触及/越过上边界。"""
    return uav_pos_u[1] >= AREA_HEIGHT_U


def is_uav_outside_bottom_edge() -> bool:
    """是否触及/越过下边界。"""
    return uav_pos_u[1] <= 0


def get_turn_angle(angle_from, angle_to, clockwise):
    """
    计算在指定旋转方向下，从 angle_from 到 angle_to 的转向角。

    注意：这里返回的是“带符号的最短可行转角（受 clockwise 约束）”。
    例如目标角相同但 clockwise=True 时，返回 0；否则不会强制绕满一圈。

    :return:
        逆时针：返回 [0, 2π)
        顺时针：返回 (-2π, 0]
    """
    diff = (angle_to - angle_from) % TWO_PI
    if clockwise:
        if diff == 0:
            return 0.0
        return diff - TWO_PI
    return diff


def uav_turn_start(start_point: np.ndarray, end_point: np.ndarray, start_angle: float, end_angle: float, is_clockwise: bool):
    """
    无人机转向初始化。

    主要步骤：
    1. 记录转向起点/终点
    2. 根据起始和目标航向计算总转角
    3. 用弦长-圆心角关系估计转弯半径
    4. 用弧长推算转弯步数并得到每步转角
    """
    global uav_turning_angle_each, is_uav_turning, is_uav_turning_clockwise
    global uav_kpt_from__x_u, uav_kpt_from__y_u, uav_kpt_to__x_u, uav_kpt_to__y_u, uav_turn_step_remain

    uav_kpt_from__x_u = start_point[0]
    uav_kpt_from__y_u = start_point[1]
    uav_kpt_to__x_u = end_point[0]
    uav_kpt_to__y_u = end_point[1]

    total_angle = get_turn_angle(start_angle, end_angle, is_clockwise)  # 总转角（含方向）

    dx = float(uav_kpt_to__x_u - uav_kpt_from__x_u)
    dy = float(uav_kpt_to__y_u - uav_kpt_from__y_u)
    chord_u = math.hypot(dx, dy)
    theta = abs(total_angle)

    # 退化情形：几乎无需转弯，直接退出转向状态
    if theta < 1e-9 or chord_u < 1e-9:
        uav_turn_step_remain = 0
        uav_turning_angle_each = 0.0
        is_uav_turning = False
        is_uav_turning_clockwise = is_clockwise
        return

    den = 2.0 * math.sin(theta * 0.5)  # 弦长公式中的分母项
    # 小角度时 sin(theta/2) 可能接近 0，此处提供稳定兜底半径
    if abs(den) < 1e-9:
        turn_radius_u = float(UAV_SCAN_RADIUS_U)
    else:
        turn_radius_u = chord_u / abs(den)

    arc_length_u = abs(total_angle) * turn_radius_u  # 弧长 = 半径 * 角度
    uav_turn_step_remain = max(1, int(round(arc_length_u / UAV_STEP_U)))
    uav_turning_angle_each = total_angle / uav_turn_step_remain

    is_uav_turning = True
    is_uav_turning_clockwise = is_clockwise


def is_uav_at_end_corner() -> bool:
    """
    判定无人机是否完成条带扫描。

    完成条件（与原逻辑一致）：
    1. 到达右上角附近且朝上
    2. 到达右下角附近且朝下
    """
    if is_uav_outside_top_edge() and (uav_pos_u[0] + UAV_SCAN_RADIUS_U) >= AREA_WIDTH_U and is_uav_up():
        return True
    if is_uav_outside_bottom_edge() and (uav_pos_u[0] + UAV_SCAN_RADIUS_U) >= AREA_WIDTH_U and is_uav_down():
        return True
    return False


def update_uav() -> bool:
    """
    无人机移动状态机。

    流程：
    1. 若正在转向，更新航向角和剩余步数
    2. 按当前航向前进一步
    3. 判定是否达到扫描终点
    4. 若不在转向，检查是否触发顶边/底边掉头
    """
    global uav_pos_u, uav_angle, uav_turn_step_remain, uav_turning_angle_each, is_uav_turning

    # 1) 若处于转向阶段：每步累加固定转角，直到步数耗尽
    if uav_turn_step_remain > 0:
        uav_angle += uav_turning_angle_each
        uav_turn_step_remain -= 1
        if uav_turn_step_remain == 0:
            uav_turning_angle_each = 0.0
            is_uav_turning = False

    # 2) 角度归一化到 [0, 2π)，避免数值无限增长
    uav_angle = uav_angle % TWO_PI

    # 3) 按当前航向推进一个离散时间步
    uav_pos_u[0] += int(round(UAV_STEP_U * np.cos(uav_angle)))
    uav_pos_u[1] += int(round(UAV_STEP_U * np.sin(uav_angle)))

    if is_uav_at_end_corner():
        print(f'UAV scan completed! Elapsed time: {time_elapsed:.2f} h')
        return False

    # 4) 航路切换逻辑（条带扫描状态机）
    if not is_uav_turning:
        # A. 向上飞行到达顶边 -> 顺时针掉头到下一条条带
        if is_uav_up() and is_uav_outside_top_edge():
            uav_turn_start(
                uav_pos_u.copy(),
                np.array([uav_pos_u[0] + 2 * UAV_SCAN_RADIUS_U, uav_pos_u[1]]),
                uav_angle,
                1.5 * np.pi,
                is_clockwise=True,
            )
        # B. 向下飞行到达底边 -> 逆时针掉头到下一条条带
        elif is_uav_down() and is_uav_outside_bottom_edge():
            uav_turn_start(
                uav_pos_u.copy(),
                np.array([uav_pos_u[0] + 2 * UAV_SCAN_RADIUS_U, uav_pos_u[1]]),
                uav_angle,
                0.5 * np.pi,
                is_clockwise=False,
            )

    return True


def remove_scanned_particles(p_locs, active_mask, active_idx):
    """
    按无人机扫描半径剔除粒子。

    为减少显存重分配开销，这里不做物理删除，
    仅将命中粒子在 active_mask 中标记为 False。
    """
    if active_idx.numel() == 0:
        return active_idx, 0

    uav_x = int(uav_pos_u[0])
    uav_y = int(uav_pos_u[1])

    # int32 坐标在平方时可能溢出，转 int64 后再算距离平方
    x = p_locs[active_idx, 0].to(torch.int64)
    y = p_locs[active_idx, 1].to(torch.int64)
    dx = x - uav_x
    dy = y - uav_y
    dist2 = dx * dx + dy * dy
    hit_mask = dist2 <= UAV_SCAN_RADIUS_U2

    hit_count = int(torch.count_nonzero(hit_mask).item())
    if hit_count > 0:
        active_mask[active_idx[hit_mask]] = False

    # 直接返回“幸存索引”，供下一步复用，避免重新 nonzero
    next_active_idx = active_idx[~hit_mask]
    return next_active_idx, hit_count


def get_counts_in_grids(p_locs, active_mask, active_idx=None):
    """
    统计所有网格的活跃粒子数量。

    实现：
    1. 在 GPU 上把 (x, y) 映射到网格索引
    2. 线性化为 1D 索引后用 bincount 聚合
    3. 最终 reshape 成 2D 矩阵并在绘图阶段回传 CPU
    """
    # 优先使用外部传入的活跃索引缓存，避免重复执行 nonzero
    if active_idx is None:
        active_idx = torch.nonzero(active_mask, as_tuple=True)[0]
    if active_idx.numel() == 0:
        return np.zeros((N_Y_BINS, N_X_BINS), dtype=np.float32)

    x = p_locs[active_idx, 0].to(torch.int64)
    y = p_locs[active_idx, 1].to(torch.int64)

    x_bin = torch.div(x, GRID_SIZE_U, rounding_mode='floor')
    y_bin = torch.div(y, GRID_SIZE_U, rounding_mode='floor')
    x_bin = torch.clamp(x_bin, 0, N_X_BINS - 1)
    y_bin = torch.clamp(y_bin, 0, N_Y_BINS - 1)

    linear = y_bin * N_X_BINS + x_bin  # 2D -> 1D
    counts_1d = torch.bincount(linear, minlength=N_X_BINS * N_Y_BINS)
    counts_2d = counts_1d.reshape(N_Y_BINS, N_X_BINS).to(torch.float32)

    # 同步后回传，确保可视化拿到当前步完整数据
    torch.cuda.synchronize()
    return counts_2d.cpu().numpy()


def run_one_step():
    """
    执行一个仿真步；返回 False 表示应终止仿真。

    A. 目标移动（GPU）
    B. 无人机移动（CPU）
    C. 扫描剔除（GPU）
    D. 记录剩余粒子与时间
    """
    global time_elapsed, sim_step_counter, active_idx_cache

    # 当前仍“可能为目标”的粒子索引（优先使用缓存，避免每步 nonzero）
    if USE_ACTIVE_INDEX_CACHE:
        active_idx = active_idx_cache
    else:
        active_idx = torch.nonzero(particle_active_mask, as_tuple=True)[0]
    
    if active_idx.numel() == 0: # 没有活跃粒子了，提前结束仿真
        return False

    # A) 粒子随机机动一步
    update_particles(particle_locations, particle_angles, active_idx, particle_step_u)

    # B) 无人机条带扫描推进一步
    if not update_uav():
        return False

    # C) 将当前扫描圆覆盖到的粒子标记为失活
    next_active_idx, _ = remove_scanned_particles(
        particle_locations,
        particle_active_mask,
        active_idx,
    )

    # D) 统计剩余不确定粒子并推进仿真时钟
    if USE_ACTIVE_INDEX_CACHE:
        active_idx_cache = next_active_idx
        remaining_particles = int(active_idx_cache.numel())
    else:
        remaining_particles = int(torch.count_nonzero(particle_active_mask).item())
    history_count.append(remaining_particles)
    time_elapsed += DT
    sim_step_counter += 1

    # 每步追加真实轨迹记录（与可视化解耦）
    record_uav_step_trace(sim_step_counter, remaining_particles)

    return (not is_uav_at_end_corner()) and (remaining_particles > 0)


# ------------------------------------------可视化------------------------------------------

if ENABLE_VISUAL_OUTPUT:
    # 实时模式下开启交互刷新；仅导出视频时不弹窗
    if REALTIME_VISUALIZATION:
        plt.ion()

    fig, window = plt.subplots(figsize=MAIN_FIG_SIZE)
    # 为左侧调试面板预留空白区域，避免与主图重叠
    fig.subplots_adjust(left=0.24)

    # 初始密度用来固定 colorbar 范围，避免刷新时色标跳动
    initial_density_matrix = get_counts_in_grids(
        particle_locations,
        particle_active_mask,
        active_idx=active_idx_cache if USE_ACTIVE_INDEX_CACHE else None,
    )
    fixed_vmax = max(1.0, float(np.max(initial_density_matrix)))

    # 热力图：显示每个网格中的活跃粒子数（概率密度代理）
    im = window.imshow(
        np.zeros((N_Y_BINS, N_X_BINS), dtype=np.float32),
        extent=(0, AREA_WIDTH_KM, 0, AREA_HEIGHT_KM),
        origin='lower',
        cmap='coolwarm',
        animated=True,
        vmin=0,
        vmax=fixed_vmax,
    )
    window.figure.colorbar(im, ax=window, label='Particle Density (particles/grid)')
    uav_dot, = window.plot([], [], 'ro', markersize=3, label='UAV')
    uav_path, = window.plot([], [], color='cyan', linewidth=1.2, alpha=0.9, label='UAV Path')
    window.set_title('CUDA Particle Density (Interval Refresh)')
    window.set_xlabel('X (km)')
    window.set_ylabel('Y (km)')
    window.legend()

    # 调试信息放在 figure 坐标系下的左侧空白区（图外），避免覆盖热力图
    status_text = fig.text(
        DEBUG_TEXT_X,
        DEBUG_TEXT_Y,
        '',
        transform=fig.transFigure,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray'),
    )
    video_writer = None
    video_output_path = None
    if EXPORT_SIMULATION_VIDEO:
        video_basename = os.path.basename(VIDEO_OUTPUT_FILENAME)
        # 优先使用 ffmpeg 导出 mp4；不可用时回退到 gif
        if animation.writers.is_available('ffmpeg'):
            video_output_path = os.path.join(RUN_RESULTS_DIR, video_basename)
            video_writer = animation.FFMpegWriter(fps=VIDEO_FPS, bitrate=2400)
        else:
            video_output_path = os.path.join(
                RUN_RESULTS_DIR,
                video_basename.rsplit('.', 1)[0] + '.gif',
            )
            video_writer = animation.PillowWriter(fps=max(1, min(VIDEO_FPS, 20)))

        video_writer.setup(fig, video_output_path, dpi=VIDEO_DPI)
        print(f'Video export enabled: {video_output_path}')
else:
    fig = None
    window = None
    im = None
    status_text = None
    video_writer = None
    video_output_path = None
    print('Realtime visualization disabled.')


def update_status_panel(remaining_particles):
    """刷新左侧状态面板文本。"""
    if not ENABLE_VISUAL_OUTPUT:
        return

    assert status_text is not None

    uav_x_km = uav_pos_u[0] / SCALE
    uav_y_km = uav_pos_u[1] / SCALE
    uav_angle_deg = (uav_angle * 180.0 / np.pi) % 360.0
    # 剩余比例可理解为“当前仍未被排除的目标概率质量”
    remaining_ratio = (remaining_particles / N_PARTICLES) * 100.0
    status_text.set_text(
        f'UAV: ({uav_x_km:.2f}, {uav_y_km:.2f}) km\n'
        f'Angle: {uav_angle_deg:.1f} deg\n'
        f'Time: {time_elapsed:.2f} h\n'
        f'Particles: {remaining_particles}/{N_PARTICLES} ({remaining_ratio:.2f}%)'
    )


def update_visualization(remaining_particles):
    """更新热力图、无人机位置、轨迹和状态文本。"""
    if not ENABLE_VISUAL_OUTPUT:
        return

    assert im is not None
    assert fig is not None
    assert uav_dot is not None
    assert uav_path is not None

    # 每次刷新都会触发一次 GPU->CPU 数据回传，频率由 STEPS_TO_UPDATE 控制
    density_matrix = get_counts_in_grids(
        particle_locations,
        particle_active_mask,
        active_idx=active_idx_cache if USE_ACTIVE_INDEX_CACHE else None,
    )
    im.set_data(density_matrix)

    # 无人机当前位置（红点）
    uav_dot.set_data([uav_pos_u[0] / SCALE], [uav_pos_u[1] / SCALE])

    # 轨迹来源于每步真实记录，这里仅负责显示
    uav_path.set_data(uav_traj_x_km, uav_traj_y_km)

    update_status_panel(remaining_particles)
    fig.canvas.draw()

    if REALTIME_VISUALIZATION:
        plt.pause(0.001)

    if video_writer is not None:
        video_writer.grab_frame()


# ------------------------------------------模拟循环------------------------------------------

print('Starting CUDA search simulation...')
sim_start_time = time.perf_counter()

for step in range(MAX_STEPS):
    # run_one_step 返回 False 的两类情况：
    # 1) 所有粒子已失活（搜索完成）
    # 2) 无人机完成整块条带覆盖（到达终止角）
    if not run_one_step():
        break

    # 间隔绘图，避免每步都进行昂贵的数据回传与渲染
    if ENABLE_VISUAL_OUTPUT and step % STEPS_TO_UPDATE == 0:
        remaining_particles = history_count[-1]
        update_visualization(remaining_particles)

torch.cuda.synchronize()
sim_cost = time.perf_counter() - sim_start_time
print(f'Simulation wall time: {sim_cost:.2f}s')

if EXPORT_UAV_TRAJECTORY:
    export_uav_trace(
        base_name=UAV_TRAJECTORY_OUTPUT_BASENAME,
        export_format=UAV_TRAJECTORY_EXPORT_FORMAT,
        include_extended=UAV_TRAJECTORY_INCLUDE_EXTENDED,
    )

if video_writer is not None:
    video_writer.finish()
    print(f'Simulation video exported: {video_output_path}')

if REALTIME_VISUALIZATION:
    plt.ioff()
    plt.show()

# 收敛曲线：剩余粒子数随时间步变化
# 若曲线快速下降，说明扫描路径覆盖效率较高。
plt.figure(figsize=SUMMARY_FIG_SIZE)
plt.plot(history_count)
plt.title('Remaining Potential Target Particles Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Particle Count')
plt.grid(True)

if REALTIME_VISUALIZATION:
    plt.show()
else:
    out_file = os.path.join(RUN_RESULTS_DIR, 'remaining_particles.png')
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f'Headless mode: saved summary figure to {out_file}')
