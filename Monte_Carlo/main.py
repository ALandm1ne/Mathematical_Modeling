'''
在现代军事活动中，无人侦察机的应用日益广泛，特别是在军事冲突中，利用无人机追踪敌方机动目标，可以为本方提供预警或者指引本方武器精准打击对方目标。
2025年长春航空展上，国产BZK-005中高空远程无人侦察机BZK-005正式亮相，它能够在18000米的高空持续飞行40小时，搭载合成孔径雷达可以从18公里的高度看清地面车牌号，配备30倍光学变焦吊舱。

假设已知敌方机动目标在海上一定范围内，如何组织本方多架BZK-005无人机进行协同搜索？
    1、请建立数学模型，根据目标大致范围、BZK-005出发点等指标，确定调用多少架无人机、每架无人机的飞行路线。
    2、假设已知目标为一艘长182米、宽24米的舰船，在海上一片矩形区域内以每小时30公里速度机动(不离开此区域)，该区域左上角顶点坐标(124E, 25N)，右下角顶点坐标(127E, 21N)。
从温州龙湾机场起飞2架BZK-005无人机，请利用你的模型计算出最少需要多长时间能完成搜索，并给出每架无人机的搜索路径。如果希望在10小时内找到目标，最少需要多少架BZK-005无人机？

注：题目中未提及的变量、参数，可以根据实际情况自行设置；如果找不到参数，可根据类似事物自行推定。
'''



import math  # 数学库，用于角度计算和反弹处理
import numpy as np  # 数值计算库，用于向量化更新大量粒子
import matplotlib.pyplot as plt  # 绘图库，用于画剩余粒子数量曲线
from matplotlib.markers import MarkerStyle  # 用于自定义无人机标记样式

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

# --- 定点缩放设置 ---
# 用整数网格单位替代浮点坐标：1 km -> 1000 单位（1 单位=1 m）
SCALE = 1000  # 坐标缩放系数
AREA_WIDTH_U = int(round(AREA_WIDTH_KM * SCALE))  # 区域宽度（整数单位）
AREA_HEIGHT_U = int(round(AREA_HEIGHT_KM * SCALE))  # 区域高度（整数单位）
UAV_SCAN_RADIUS_U = int(round(UAV_SCAN_RADIUS_KM * SCALE))  # 扫描半径（整数单位）

# --- 初始化粒子 (目标) ---
# 粒子位置数组 shape=(N,2)，每行是 [x, y]，使用 int64 避免累计浮点误差
# 使用分层均匀采样：将区域划分成规则网格，每个网格内随机投放 1 个粒子，使初始密度更均匀
particle_locations = np.empty((N_PARTICLES, 2), dtype=np.int64)  # 初始化粒子位置数组: N行2列，整数类型
grid_nx = max(1, int(math.sqrt(N_PARTICLES * AREA_WIDTH_U / AREA_HEIGHT_U)))
grid_ny = int(math.ceil(N_PARTICLES / grid_nx))
cell_indices = np.arange(N_PARTICLES, dtype=np.int64)
cell_x = cell_indices % grid_nx
cell_y = cell_indices // grid_nx
jitter_x = np.random.rand(N_PARTICLES)
jitter_y = np.random.rand(N_PARTICLES)
particle_locations[:, 0] = np.minimum(
    AREA_WIDTH_U,
    np.rint(((cell_x + jitter_x) / grid_nx) * AREA_WIDTH_U).astype(np.int64),
)
particle_locations[:, 1] = np.minimum(
    AREA_HEIGHT_U,
    np.rint(((cell_y + jitter_y) / grid_ny) * AREA_HEIGHT_U).astype(np.int64),
)
particle_angles = np.random.rand(N_PARTICLES) * 2 * np.pi  # 每个粒子的运动方向角（弧度）[0, 2π)均匀随机

# --- 初始化无人机 ---

UAV_STEP_U = int(round(UAV_SPEED_KM * DT * SCALE))  # 无人机每步移动的距离（整数单位）
uav_angle = 0.5 * np.pi  # 无人机当前移动方向角（弧度），初始向右（0度）
is_uav_turning = False  # 无人机是否正在转向
is_uav_turning_clockwise = False  # 无人机转向方向（True -> 顺时针 || False -> 逆时针）
uav_turning_angle_each = 0  # 无人机每步的转向角（弧度），初始为0，后续根据边界情况调整
uav_turn_step_remain = 0  # 无人机当前转向还剩多少步，初始为0，后续根据边界情况调整

uav_pos_u = np.array([UAV_SCAN_RADIUS_U, 0], dtype=np.int64)  # 无人机从左下角出发，单位为缩放后的整数坐标
uav_kpt_from__x_u = 0  # 无人机上次kpt的 x 坐标 (整数单位)
uav_kpt_from__y_u = 0  # 无人机上次kpt的 y 坐标 (整数单位)
uav_kpt_to__x_u = 0  # 无人机下次kpt的 x 坐标 (整数单位)
uav_kpt_to__y_u = 0  # 无人机下次kpt的 y 坐标 (整数单位)

# --- 模拟循环变量 ---
history_count = []  # 记录每一步剩余粒子数，用于最终绘图
time_elapsed = 0  # 累计仿真时间（小时）

# UAV 轨迹缓存（单位 km，用于实时绘图）
uav_traj_x_km = [uav_pos_u[0] / SCALE]
uav_traj_y_km = [uav_pos_u[1] / SCALE]

particle_step_u = int(round(TARGET_SPEED_KM * DT * SCALE))  # 目标每步移动的距离 (整数单位)

# --- 绘制设置 ---
GRID_SIZE_U = 2 * SCALE  # 调整网格大小以平衡性能和清晰度

# ------------------------------------------移动控制相关函数------------------------------------------

# --- 粒子位置更新函数 ---

def update_particles(p, ang, step_u):
    
    """
    按当前速度和方向更新粒子位置，并在边界处做反弹处理。
    
    :param p: 粒子位置数组 shape=(N,2)，每行是 [x, y]，整数类型
    :param ang: 粒子运动方向角数组 shape=(N,)，单位为弧度
    :param step_u: 每步移动的距离（整数单位）
    :return: 更新后的粒子位置数组和方向角数组
    """
    
    # 每个 step 为每个粒子重置随机方向
    ang[:] = np.random.rand(ang.shape[0]) * 2 * np.pi

    dx = np.rint(step_u * np.cos(ang)).astype(np.int64)  # x 方向位移（按角度投影后取整）
    dy = np.rint(step_u * np.sin(ang)).astype(np.int64)  # y 方向位移（按角度投影后取整）
    p[:, 0] += dx  # 批量更新所有粒子的 x 坐标
    p[:, 1] += dy  # 批量更新所有粒子的 y 坐标

    # 边界检查：超出左右边界时，方向角按 x 轴法线反射
    mask_x = (p[:, 0] < 0) | (p[:, 0] > AREA_WIDTH_U)

    # 边界检查：超出上下边界时，方向角按 y 轴法线反射
    mask_y = (p[:, 1] < 0) | (p[:, 1] > AREA_HEIGHT_U)

    # 边界反射：只更新越界粒子的角度
    ang[mask_x] = (np.pi - ang[mask_x]) % (2 * np.pi)
    ang[mask_y] = (-ang[mask_y]) % (2 * np.pi)

    # 坐标裁剪回合法区域，避免粒子停留在边界外
    p[:, 0] = np.clip(p[:, 0], 0, AREA_WIDTH_U )  # 裁剪 x 坐标到 [0, AREA_WIDTH_U]
    p[:, 1] = np.clip(p[:, 1], 0, AREA_HEIGHT_U)  # 裁剪 y 坐标到 [0, AREA_HEIGHT_U]

    return p, ang  # 返回更新后的粒子坐标和方向角

# --- UAV移动控制相关函数 ---

def is_uav_up() -> bool:
    return (uav_angle > 0) and (uav_angle < np.pi)
def is_uav_down() -> bool:
    return (uav_angle > np.pi) and (uav_angle < 2 * np.pi)
def is_uav_right() -> bool:
    return ((uav_angle >= 0) and (uav_angle < 0.5 * np.pi)) or (uav_angle > 1.5 * np.pi)
def is_uav_left() -> bool:
    return (uav_angle > 0.5 * np.pi) and (uav_angle < 1.5 * np.pi)

def is_uav_outside_top_edge() -> bool:
    return (uav_pos_u[1]) >= AREA_HEIGHT_U
def is_uav_outside_bottom_edge() -> bool:
    return (uav_pos_u[1]) <= 0
def is_uav_outside_right_edge() -> bool:
    return (uav_pos_u[0]) >= AREA_WIDTH_U
def is_uav_outside_left_edge() -> bool:
    return (uav_pos_u[0]) <= 0

def is_uav_2_scan_radius_from_kpt() -> bool:
    return (uav_pos_u[0] - uav_kpt_from__x_u)**2 + (uav_pos_u[1] - uav_kpt_from__y_u)**2 >= (2 * UAV_SCAN_RADIUS_U)**2 



def get_turn_angle(angle_from, angle_to, clockwise):
    
    """
    计算给定方向下的旋转角度
    
    :param angle_from: 起始角度 (弧度)
    :param angle_to: 目标角度 (弧度)
    :param clockwise: 是否为顺时针 (True -> 顺时针 || False  -> 逆时针)
    :return: 旋转角度。逆时针返回 (0, 2pi], 顺时针返回 [-2pi, 0)
    """
    
    two_pi = 2 * math.pi
    
    # 计算逆时针方向的基础差值，并映射到 [0, 2pi)
    # Python 的 % 运算符会自动处理负数，例如 -0.1 % 6.28 会得到 6.18
    diff = (angle_to - angle_from) % two_pi
    
    if clockwise:
        # 如果 diff 为 0，表示目标就在原地。
        # 如果需要强制转一圈，可以处理 diff == 0 的情况。
        if diff == 0:
            return 0.0
        # 顺时针旋转：将正向跨度减去 2pi 得到对应的负向跨度
        return diff - two_pi
    else:
        # 逆时针旋转：直接返回 [0, 2pi) 范围内的差值
        # 如果 diff 为 0 且需要表示“转一圈”，可以根据需求改为 two_pi
        return diff



def uav_turn_start(start_point: np.ndarray, end_point: np.ndarray, start_angle: float, end_angle: float, is_clockwise: bool):
    '''
    无人机转向开始时的初始化:
    1. 记录转向起点和终点的坐标（整数单位）
    2. 计算总转向角度（根据当前角度、目标角度和转向方向）
    3. 根据转向半径和总转向角度计算转向弧长，进而计算总转向步数
    4. 计算每步的转向角度
    5. 设置转向状态和转向方向
    
    :param start_point: 转向起点坐标 [x, y]，整数单位
    :param end_point: 转向终点坐标 [x, y]，整数单位
    :param start_angle: 转向起始角度（弧度）
    :param end_angle: 转向目标角度（弧度）
    :param is_clockwise: 转向方向（True -> 顺时针 || False -> 逆时针）
    '''
    global uav_turning_angle_each, is_uav_turning, is_uav_turning_clockwise, uav_kpt_from__x_u, uav_kpt_from__y_u, uav_kpt_to__x_u, uav_kpt_to__y_u, uav_turn_step_remain
    
    uav_kpt_from__x_u = start_point[0]  # 记录kpt的 x 坐标
    uav_kpt_from__y_u = start_point[1]  # 记录kpt的 y 坐标
    
    uav_kpt_to__x_u = end_point[0]  # 记录kpt的 x 坐标
    uav_kpt_to__y_u = end_point[1]  # 记录kpt的 y 坐标

    total_angle = get_turn_angle(start_angle, end_angle, is_clockwise)

    # 用弦长-圆心角关系计算半径，避免 cos/sin 接近 0 时数值爆炸
    dx = float(uav_kpt_to__x_u - uav_kpt_from__x_u)
    dy = float(uav_kpt_to__y_u - uav_kpt_from__y_u)
    chord_u = math.hypot(dx, dy)  # 弦长(整数单位)
    theta = abs(total_angle)  # 转向的总角度（弧度）

    if theta < 1e-9 or chord_u < 1e-9:
        uav_turn_step_remain = 0
        uav_turning_angle_each = 0.0
        is_uav_turning = False
        is_uav_turning_clockwise = is_clockwise
        return

    den = 2.0 * math.sin(theta * 0.5)  # 弦长/半径
    if abs(den) < 1e-9:
        turn_radius_u = float(UAV_SCAN_RADIUS_U)
    else:
        turn_radius_u = chord_u / abs(den)
    
    # 弧长 = 角度(弧度) * 半径
    arc_length_u = abs(total_angle) * turn_radius_u
    
    # 总步数 = 弧长 / 每步移动距离
    uav_turn_step_remain = max(1, int(round(arc_length_u / UAV_STEP_U)))
    
    # 每步分摊的角度
    uav_turning_angle_each = total_angle / uav_turn_step_remain
    
    is_uav_turning = True
    is_uav_turning_clockwise = is_clockwise



def is_uav_at_end_corner() -> bool:
    '''
    无人机是否到达右上角/右下角并且用速度方向判定是否完成扫描的判定:
    1. 到达右上角并且向上方飞行
    2. 到达右下角并且向下方飞行
    '''
    ret = False
    if(is_uav_outside_top_edge() and (uav_pos_u[0] + UAV_SCAN_RADIUS_U) >= AREA_WIDTH_U and is_uav_up()):
        ret = True
    if(is_uav_outside_bottom_edge() and (uav_pos_u[0] + UAV_SCAN_RADIUS_U) >= AREA_WIDTH_U and is_uav_down()):
        ret = True
    
    return ret



def update_uav() -> bool:
    '''
    无人机移动逻辑
    1. 更新无人机飞行状态（转向中则更新角度和剩余转向步数）
    2. 更新无人机位置（无论直线还是曲线，每步移动 UAV_STEP_U）
    3. 判定扫描完成（到达右上角/右下角并且用速度方向判定是否完成扫描）
    4. 航路切换逻辑 (状态机)
        A. 向上飞行到达顶点 -> 开始顺时针转 90 度向右
        B. 向下飞行到达底点 -> 开始逆时针转 90 度向右
    
    :return: 是否继续仿真（False -> 无人机完成扫描，True -> 继续仿真）
    '''
    global uav_pos_u, uav_angle, uav_turn_step_remain, uav_turning_angle_each, is_uav_turning, time_elapsed
    
    # --- 1. 更新无人机飞行状态 ---
    if uav_turn_step_remain > 0:
        # 转向中：更新角度
        uav_angle += uav_turning_angle_each
        uav_turn_step_remain -= 1
        if uav_turn_step_remain == 0:
            uav_turning_angle_each = 0
            is_uav_turning = False
    
    # 保持角度在 [0, 2π) 范围内
    uav_angle = uav_angle % (2 * np.pi)

    # 更新位置 (无论直线还是曲线，每步移动 UAV_STEP_U)
    uav_pos_u[0] += int(round(UAV_STEP_U * np.cos(uav_angle)))
    uav_pos_u[1] += int(round(UAV_STEP_U * np.sin(uav_angle)))

    # --- 2. 判定扫描完成 ---
    if is_uav_at_end_corner():
        print(f"UAV scan completed! Elapsed time: {time_elapsed:.2f} h")
        return False

    # --- 3. 航路切换逻辑 (状态机) ---
    if is_uav_turning == False:
        # A. 向上飞行到达顶点 -> 开始顺时针调头扫描下一条条带
        if is_uav_up() and is_uav_outside_top_edge():
            # 目标位置 ([x+r], [y])，目标航向 1.5π (向下)
            uav_turn_start(uav_pos_u.copy(), 
                           np.array([uav_pos_u[0] + 2 * UAV_SCAN_RADIUS_U, uav_pos_u[1]]),
                           uav_angle,
                           1.5 * np.pi,  # 目标航向 1.5π (向下)
                           is_clockwise=True)

        # B. 向下飞行到达底点 -> 开始逆时针调头扫描下一条条带
        elif is_uav_down() and is_uav_outside_bottom_edge():
            # 目标位置 ([x+r], [y])，目标航向 0.5π (向上)
            uav_turn_start(uav_pos_u.copy(), 
                           np.array([uav_pos_u[0] + 2 * UAV_SCAN_RADIUS_U, uav_pos_u[1]]),
                           uav_angle,
                           0.5 * np.pi,  # 目标航向 0.5π (向上)
                           is_clockwise=False)
            
    return True
    
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------

# --- 4. 模拟循环 ---

# --- 实时可视化准备 ---
plt.ion()  # 开启交互模式
fig, window = plt.subplots(figsize=(8, 6))

# 初始化热力图网格
x_bins = np.arange(0, AREA_WIDTH_U + GRID_SIZE_U, GRID_SIZE_U)
y_bins = np.arange(0, AREA_HEIGHT_U + GRID_SIZE_U, GRID_SIZE_U)

# 初始密度矩阵（用初始粒子位置）
initial_density_matrix, _, _ = np.histogram2d(
    particle_locations[:, 1],
    particle_locations[:, 0],
    bins=[y_bins, x_bins]
)
fixed_vmax = max(1, float(np.max(initial_density_matrix)))

# 创建初始空图层
im = window.imshow(
               np.zeros((len(y_bins)-1, len(x_bins)-1)),    # 注意这里的维度是 (y_bins-1, x_bins-1)，因为 histogram2d 的输出是这样的
            
               extent=(0, AREA_WIDTH_KM, 0, AREA_HEIGHT_KM),    # 设置坐标轴范围为实际的 km 单位
               origin='lower',                                  # 设置 origin='lower' 左下角为原点
            #    cmap='YlOrRd',                                   # 使用 YlOrRd 颜色映射
                cmap='coolwarm',                                 # 使用 coolwarm 颜色映射
            #    cmap='magma',                                    # 使用 magma 颜色映射
            #    cmap='inferno',                                  # 使用 inferno 颜色映射
            #    cmap='viridis',                                  # 使用 viridis 颜色映射
            #    cmap='hot',                                      # 使用 hot 颜色映射
               animated=True ,                                   # 设置为动画模式以提高更新效率
               vmin = 0 ,
               vmax = fixed_vmax
            )    
cbar = fig.colorbar(im, ax=window, label='Particle Density (particles/km²)')
uav_dot, = window.plot([], [], 'ro', markersize=3 , label='UAV') # 绘制无人机位置点
uav_path, = window.plot([], [], color='cyan', linewidth=1.2, alpha=0.9, label='UAV Path')
window.set_title('Real-time Particle Density')
window.set_xlabel('X (km)')
window.set_ylabel('Y (km)')
window.legend() # 显示图例

# 实时信息面板：显示无人机坐标、角度和剩余粒子比例
status_text = window.text(
    -0.7,
    0.98,
    '',
    transform=window.transAxes,
    va='top',
    ha='left',
    fontsize=10,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray')
)
status_text.set_text(
    f"UAV: ({uav_pos_u[0] / SCALE:.2f}, {uav_pos_u[1] / SCALE:.2f}) km\n"
    f"Angle: {(uav_angle * 180.0 / np.pi) % 360.0:.1f} deg\n"
    f"Time: {time_elapsed:.2f} h\n"
    f"Particles: {len(particle_locations)}/{N_PARTICLES} ({len(particle_locations) / N_PARTICLES * 100:.2f}%)"
)

def get_counts_in_grids(p_locs):
    """计算当前所有格子的粒子数量矩阵"""
    # 使用 numpy 的直方图函数，比双重循环快得多
    counts, _, _ = np.histogram2d(
        p_locs[:, 1],
        p_locs[:, 0], 
        bins=[y_bins, x_bins]
    )
    # area_km2 = (grid_size / SCALE)**2
    # return counts / area_km2
    return counts  # 返回每个格子内的粒子数量矩阵


def update_status_panel(remaining_particles):
    """刷新左侧状态面板文本。"""
    uav_x_km = uav_pos_u[0] / SCALE
    uav_y_km = uav_pos_u[1] / SCALE
    uav_angle_deg = (uav_angle * 180.0 / np.pi) % 360.0
    remaining_ratio = (remaining_particles / N_PARTICLES) * 100.0
    status_text.set_text(
        f"UAV: ({uav_x_km:.2f}, {uav_y_km:.2f}) km\n"
        f"Angle: {uav_angle_deg:.1f} deg\n"
        f"Time: {time_elapsed:.2f} h\n"
        f"Particles: {remaining_particles}/{N_PARTICLES} ({remaining_ratio:.2f}%)"
    )


def remove_scanned_particles(p_locs, p_angles):
    """按无人机扫描半径剔除粒子。"""
    delta = p_locs - uav_pos_u
    keep_mask = np.sum(delta**2, axis=1) > (UAV_SCAN_RADIUS_U**2)
    return p_locs[keep_mask], p_angles[keep_mask]


def run_one_step():
    """执行一个仿真步；返回 False 表示应终止仿真。"""
    global particle_locations, particle_angles, time_elapsed

    # A. 目标移动
    particle_locations, particle_angles = update_particles(
        particle_locations,
        particle_angles,
        particle_step_u,
    )

    # B. 无人机移动
    if not update_uav():
        return False

    # C. 扫描并剔除被覆盖粒子
    particle_locations, particle_angles = remove_scanned_particles(
        particle_locations,
        particle_angles,
    )

    history_count.append(len(particle_locations))
    time_elapsed += DT

    return (not is_uav_at_end_corner()) and (len(particle_locations) > 0)


def update_visualization():
    """按刷新频率更新热力图、无人机标记和状态面板。"""

    density_matrix = get_counts_in_grids(particle_locations)
    im.set_data(density_matrix)

    # 更新无人机位置标记
    uav_dot.set_data([uav_pos_u[0] / SCALE], [uav_pos_u[1] / SCALE])

    # 更新无人机轨迹
    uav_traj_x_km.append(uav_pos_u[0] / SCALE)
    uav_traj_y_km.append(uav_pos_u[1] / SCALE)
    uav_path.set_data(uav_traj_x_km, uav_traj_y_km)

    update_status_panel(len(particle_locations))

    plt.draw()
    plt.pause(0.001)  # 暂停微小时间以刷新画布

# ------------------------------------------

# --- 5. 模拟循环 ---
print("Starting search simulation...")
for step in range(MAX_STEPS):
    if not run_one_step():
        break
    
    if step % STEPS_TO_UPDATE == 0:
        update_visualization()



plt.ioff() # 关闭交互模式
plt.show() # 保持最后结果显示

# --- 6. 结果可视化 ---
plt.figure(figsize=(10, 5))  # 创建画布
plt.plot(history_count)  # 绘制剩余粒子数量曲线
plt.title("Remaining Potential Target Particles Over Time")  # 图标题
plt.xlabel("Time Steps")  # x 轴标签
plt.ylabel("Particle Count")  # y 轴标签
plt.grid(True)  # 显示网格线
plt.show()  # 展示图像窗口
