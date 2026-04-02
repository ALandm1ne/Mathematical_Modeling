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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use('QtAgg')

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
STEPS_TO_UPDATE = 1  # 每隔多少步更新一次绘图（调整以平衡性能和实时性）

# --- 设备设置（仅支持 CUDA） ---
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required but not available.')

DEVICE = torch.device('cuda')
GPU_NAME = torch.cuda.get_device_name(0)
print(f'Using CUDA device: {GPU_NAME}')

# --- 定点缩放设置 ---
# 用整数网格单位替代浮点坐标：1 km -> 1000 单位（1 单位=1 m）
SCALE = 1000    # 坐标缩放系数
AREA_WIDTH_U = int(round(AREA_WIDTH_KM * SCALE))    # 区域宽度（整数单位）
AREA_HEIGHT_U = int(round(AREA_HEIGHT_KM * SCALE))  # 区域高度（整数单位）
UAV_SCAN_RADIUS_U = int(round(UAV_SCAN_RADIUS_KM * SCALE))  # 扫描半径（整数单位）
UAV_SCAN_RADIUS_U2 = UAV_SCAN_RADIUS_U * UAV_SCAN_RADIUS_U  # 扫描半径的平方（整数单位）

# --- UAV 参数 ---
UAV_STEP_U = int(round(UAV_SPEED_KM * DT * SCALE))
particle_step_u = int(round(TARGET_SPEED_KM * DT * SCALE))

# --- 数值常量 ---
TWO_PI = 2.0 * math.pi
PI = math.pi

# --- 绘制设置 ---
GRID_SIZE_U = 2 * SCALE
N_X_BINS = AREA_WIDTH_U // GRID_SIZE_U + 1
N_Y_BINS = AREA_HEIGHT_U // GRID_SIZE_U + 1

# ------------------------------------------初始化------------------------------------------

def init_particles_on_cuda(num_particles: int):
    """在 GPU 上完成分层均匀采样初始化。"""
    grid_nx = max(1, int(math.sqrt(num_particles * AREA_WIDTH_U / AREA_HEIGHT_U)))
    grid_ny = int(math.ceil(num_particles / grid_nx))

    cell_indices = torch.arange(num_particles, device=DEVICE, dtype=torch.int64)
    cell_x = cell_indices.remainder(grid_nx).to(torch.float32)
    cell_y = torch.div(cell_indices, grid_nx, rounding_mode='floor').to(torch.float32)

    jitter_x = torch.rand(num_particles, device=DEVICE)
    jitter_y = torch.rand(num_particles, device=DEVICE)

    x = torch.round(((cell_x + jitter_x) / float(grid_nx)) * AREA_WIDTH_U)
    y = torch.round(((cell_y + jitter_y) / float(grid_ny)) * AREA_HEIGHT_U)

    x = torch.clamp(x, 0, AREA_WIDTH_U).to(torch.int32)
    y = torch.clamp(y, 0, AREA_HEIGHT_U).to(torch.int32)

    p_locs = torch.stack((x, y), dim=1)
    p_angles = torch.rand(num_particles, device=DEVICE, dtype=torch.float32) * TWO_PI
    p_active = torch.ones(num_particles, device=DEVICE, dtype=torch.bool)
    return p_locs, p_angles, p_active


particle_locations, particle_angles, particle_active_mask = init_particles_on_cuda(N_PARTICLES)

# --- 初始化无人机（仍保留 CPU 标量逻辑） ---
uav_angle = 0.5 * np.pi
is_uav_turning = False
is_uav_turning_clockwise = False
uav_turning_angle_each = 0.0
uav_turn_step_remain = 0

uav_pos_u = np.array([UAV_SCAN_RADIUS_U, 0], dtype=np.int64)
uav_kpt_from__x_u = 0
uav_kpt_from__y_u = 0
uav_kpt_to__x_u = 0
uav_kpt_to__y_u = 0

history_count = []
time_elapsed = 0.0

uav_traj_x_km = [uav_pos_u[0] / SCALE]
uav_traj_y_km = [uav_pos_u[1] / SCALE]

# ------------------------------------------移动控制相关函数------------------------------------------

def update_particles(p_locs, p_angles, active_idx, step_u):
    """按当前速度更新活跃粒子，并在边界处做反弹处理。"""
    if active_idx.numel() == 0:
        return

    new_angles = torch.rand(active_idx.numel(), device=DEVICE, dtype=torch.float32) * TWO_PI

    dx = torch.round(step_u * torch.cos(new_angles)).to(torch.int32)
    dy = torch.round(step_u * torch.sin(new_angles)).to(torch.int32)

    p_active = p_locs[active_idx]
    p_active[:, 0] += dx
    p_active[:, 1] += dy

    mask_x = (p_active[:, 0] < 0) | (p_active[:, 0] > AREA_WIDTH_U)
    mask_y = (p_active[:, 1] < 0) | (p_active[:, 1] > AREA_HEIGHT_U)

    if mask_x.any():
        new_angles[mask_x] = (PI - new_angles[mask_x]).remainder(TWO_PI)
    if mask_y.any():
        new_angles[mask_y] = (-new_angles[mask_y]).remainder(TWO_PI)

    p_active[:, 0] = torch.clamp(p_active[:, 0], 0, AREA_WIDTH_U)
    p_active[:, 1] = torch.clamp(p_active[:, 1], 0, AREA_HEIGHT_U)

    p_locs[active_idx] = p_active
    p_angles[active_idx] = new_angles


def is_uav_up() -> bool:
    return (uav_angle > 0) and (uav_angle < np.pi)


def is_uav_down() -> bool:
    return (uav_angle > np.pi) and (uav_angle < 2 * np.pi)


def is_uav_outside_top_edge() -> bool:
    return uav_pos_u[1] >= AREA_HEIGHT_U


def is_uav_outside_bottom_edge() -> bool:
    return uav_pos_u[1] <= 0


def get_turn_angle(angle_from, angle_to, clockwise):
    diff = (angle_to - angle_from) % TWO_PI
    if clockwise:
        if diff == 0:
            return 0.0
        return diff - TWO_PI
    return diff


def uav_turn_start(start_point: np.ndarray, end_point: np.ndarray, start_angle: float, end_angle: float, is_clockwise: bool):
    global uav_turning_angle_each, is_uav_turning, is_uav_turning_clockwise
    global uav_kpt_from__x_u, uav_kpt_from__y_u, uav_kpt_to__x_u, uav_kpt_to__y_u, uav_turn_step_remain

    uav_kpt_from__x_u = start_point[0]
    uav_kpt_from__y_u = start_point[1]
    uav_kpt_to__x_u = end_point[0]
    uav_kpt_to__y_u = end_point[1]

    total_angle = get_turn_angle(start_angle, end_angle, is_clockwise)

    dx = float(uav_kpt_to__x_u - uav_kpt_from__x_u)
    dy = float(uav_kpt_to__y_u - uav_kpt_from__y_u)
    chord_u = math.hypot(dx, dy)
    theta = abs(total_angle)

    if theta < 1e-9 or chord_u < 1e-9:
        uav_turn_step_remain = 0
        uav_turning_angle_each = 0.0
        is_uav_turning = False
        is_uav_turning_clockwise = is_clockwise
        return

    den = 2.0 * math.sin(theta * 0.5)
    if abs(den) < 1e-9:
        turn_radius_u = float(UAV_SCAN_RADIUS_U)
    else:
        turn_radius_u = chord_u / abs(den)

    arc_length_u = abs(total_angle) * turn_radius_u
    uav_turn_step_remain = max(1, int(round(arc_length_u / UAV_STEP_U)))
    uav_turning_angle_each = total_angle / uav_turn_step_remain

    is_uav_turning = True
    is_uav_turning_clockwise = is_clockwise


def is_uav_at_end_corner() -> bool:
    if is_uav_outside_top_edge() and (uav_pos_u[0] + UAV_SCAN_RADIUS_U) >= AREA_WIDTH_U and is_uav_up():
        return True
    if is_uav_outside_bottom_edge() and (uav_pos_u[0] + UAV_SCAN_RADIUS_U) >= AREA_WIDTH_U and is_uav_down():
        return True
    return False


def update_uav() -> bool:
    global uav_pos_u, uav_angle, uav_turn_step_remain, uav_turning_angle_each, is_uav_turning

    if uav_turn_step_remain > 0:
        uav_angle += uav_turning_angle_each
        uav_turn_step_remain -= 1
        if uav_turn_step_remain == 0:
            uav_turning_angle_each = 0.0
            is_uav_turning = False

    uav_angle = uav_angle % TWO_PI

    uav_pos_u[0] += int(round(UAV_STEP_U * np.cos(uav_angle)))
    uav_pos_u[1] += int(round(UAV_STEP_U * np.sin(uav_angle)))

    if is_uav_at_end_corner():
        print(f'UAV scan completed! Elapsed time: {time_elapsed:.2f} h')
        return False

    if not is_uav_turning:
        if is_uav_up() and is_uav_outside_top_edge():
            uav_turn_start(
                uav_pos_u.copy(),
                np.array([uav_pos_u[0] + 2 * UAV_SCAN_RADIUS_U, uav_pos_u[1]]),
                uav_angle,
                1.5 * np.pi,
                is_clockwise=True,
            )
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
    """基于当前 UAV 位置将扫描到的粒子标记为非活跃。"""
    if active_idx.numel() == 0:
        return

    uav_x = int(uav_pos_u[0])
    uav_y = int(uav_pos_u[1])

    x = p_locs[active_idx, 0].to(torch.int64)
    y = p_locs[active_idx, 1].to(torch.int64)
    dx = x - uav_x
    dy = y - uav_y
    dist2 = dx * dx + dy * dy
    hit_mask = dist2 <= UAV_SCAN_RADIUS_U2

    if hit_mask.any():
        active_mask[active_idx[hit_mask]] = False


def get_counts_in_grids(p_locs, active_mask):
    """在 GPU 上做二维分箱统计，绘图时再拷回 CPU。"""
    active_idx = torch.nonzero(active_mask, as_tuple=True)[0]
    if active_idx.numel() == 0:
        return np.zeros((N_Y_BINS, N_X_BINS), dtype=np.float32)

    x = p_locs[active_idx, 0].to(torch.int64)
    y = p_locs[active_idx, 1].to(torch.int64)

    x_bin = torch.div(x, GRID_SIZE_U, rounding_mode='floor')
    y_bin = torch.div(y, GRID_SIZE_U, rounding_mode='floor')
    x_bin = torch.clamp(x_bin, 0, N_X_BINS - 1)
    y_bin = torch.clamp(y_bin, 0, N_Y_BINS - 1)

    linear = y_bin * N_X_BINS + x_bin
    counts_1d = torch.bincount(linear, minlength=N_X_BINS * N_Y_BINS)
    counts_2d = counts_1d.reshape(N_Y_BINS, N_X_BINS).to(torch.float32)

    torch.cuda.synchronize()
    return counts_2d.cpu().numpy()


def run_one_step():
    """执行一个仿真步；返回 False 表示应终止仿真。"""
    global time_elapsed

    active_idx = torch.nonzero(particle_active_mask, as_tuple=True)[0]
    if active_idx.numel() == 0:
        return False

    update_particles(particle_locations, particle_angles, active_idx, particle_step_u)

    if not update_uav():
        return False

    remove_scanned_particles(particle_locations, particle_active_mask, active_idx)

    remaining_particles = int(torch.count_nonzero(particle_active_mask).item())
    history_count.append(remaining_particles)
    time_elapsed += DT

    return (not is_uav_at_end_corner()) and (remaining_particles > 0)


# ------------------------------------------可视化------------------------------------------

plt.ion()
fig, window = plt.subplots(figsize=(8, 6))

x_bins = np.arange(0, AREA_WIDTH_U + GRID_SIZE_U, GRID_SIZE_U)
y_bins = np.arange(0, AREA_HEIGHT_U + GRID_SIZE_U, GRID_SIZE_U)

initial_density_matrix = get_counts_in_grids(particle_locations, particle_active_mask)
fixed_vmax = max(1.0, float(np.max(initial_density_matrix)))

im = window.imshow(
    np.zeros((N_Y_BINS, N_X_BINS), dtype=np.float32),
    extent=(0, AREA_WIDTH_KM, 0, AREA_HEIGHT_KM),
    origin='lower',
    cmap='coolwarm',
    animated=True,
    vmin=0,
    vmax=fixed_vmax,
)
fig.colorbar(im, ax=window, label='Particle Density (particles/grid)')
uav_dot, = window.plot([], [], 'ro', markersize=3, label='UAV')
uav_path, = window.plot([], [], color='cyan', linewidth=1.2, alpha=0.9, label='UAV Path')
window.set_title('CUDA Particle Density (Interval Refresh)')
window.set_xlabel('X (km)')
window.set_ylabel('Y (km)')
window.legend()

status_text = window.text(
    -0.7,
    0.98,
    '',
    transform=window.transAxes,
    va='top',
    ha='left',
    fontsize=10,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray'),
)


def update_status_panel(remaining_particles):
    uav_x_km = uav_pos_u[0] / SCALE
    uav_y_km = uav_pos_u[1] / SCALE
    uav_angle_deg = (uav_angle * 180.0 / np.pi) % 360.0
    remaining_ratio = (remaining_particles / N_PARTICLES) * 100.0
    status_text.set_text(
        f'UAV: ({uav_x_km:.2f}, {uav_y_km:.2f}) km\n'
        f'Angle: {uav_angle_deg:.1f} deg\n'
        f'Time: {time_elapsed:.2f} h\n'
        f'Particles: {remaining_particles}/{N_PARTICLES} ({remaining_ratio:.2f}%)'
    )


def update_visualization(remaining_particles):
    density_matrix = get_counts_in_grids(particle_locations, particle_active_mask)
    im.set_data(density_matrix)

    uav_dot.set_data([uav_pos_u[0] / SCALE], [uav_pos_u[1] / SCALE])

    uav_traj_x_km.append(uav_pos_u[0] / SCALE)
    uav_traj_y_km.append(uav_pos_u[1] / SCALE)
    uav_path.set_data(uav_traj_x_km, uav_traj_y_km)

    update_status_panel(remaining_particles)
    plt.draw()
    plt.pause(0.001)


# ------------------------------------------模拟循环------------------------------------------

print('Starting CUDA search simulation...')
sim_start_time = time.perf_counter()

for step in range(MAX_STEPS):
    if not run_one_step():
        break

    if step % STEPS_TO_UPDATE == 0:
        remaining_particles = history_count[-1]
        update_visualization(remaining_particles)

torch.cuda.synchronize()
sim_cost = time.perf_counter() - sim_start_time
print(f'Simulation wall time: {sim_cost:.2f}s')

plt.ioff()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history_count)
plt.title('Remaining Potential Target Particles Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Particle Count')
plt.grid(True)
plt.show()
