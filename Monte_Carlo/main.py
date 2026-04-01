import math  # 数学库，用于角度计算和反弹处理
import numpy as np  # 数值计算库，用于向量化更新大量粒子
import matplotlib.pyplot as plt  # 绘图库，用于画剩余粒子数量曲线
from matplotlib.markers import MarkerStyle  # 用于自定义无人机标记样式

# ------------------------------------------全局变量------------------------------------------

# --- 参数设置 ---
AREA_WIDTH_KM = 306.0  # 模拟区域宽度，单位 km（经度方向跨度）
AREA_HEIGHT_KM = 444.0  # 模拟区域高度，单位 km（纬度方向跨度）
N_PARTICLES = 1000000  # 初始粒子数，表示潜在目标样本数量
TARGET_SPEED_KM = 30.0  # 目标运动速度，单位 km/h
UAV_SPEED_KM = 150.0  # 无人机巡航速度，单位 km/h
UAV_SCAN_RADIUS_KM = 15.0  # 无人机传感器扫描半径，单位 km
DT = 0.01  # 仿真时间步长，单位小时
MAX_STEPS = 1000  # 最大仿真步数
STEPS_TO_UPDATE = 5  # 每隔多少步更新一次绘图（调整以平衡性能和实时性）

# --- 定点缩放设置 ---
# 用整数网格单位替代浮点坐标：1 km -> 1000 单位（1 单位=1 m）
SCALE = 1000  # 坐标缩放系数
AREA_WIDTH_U = int(round(AREA_WIDTH_KM * SCALE))  # 区域宽度（整数单位）
AREA_HEIGHT_U = int(round(AREA_HEIGHT_KM * SCALE))  # 区域高度（整数单位）
UAV_SCAN_RADIUS_U = int(round(UAV_SCAN_RADIUS_KM * SCALE))  # 扫描半径（整数单位）

# --- 初始化粒子 (目标) ---
# 粒子位置数组 shape=(N,2)，每行是 [x, y]，使用 int64 避免累计浮点误差
particle_locations = np.empty((N_PARTICLES, 2), dtype=np.int64)  # 初始化粒子位置数组: N行2列，整数类型
particle_locations[:, 0] = (np.random.rand(N_PARTICLES) * AREA_WIDTH_U ).astype(np.int64)  # x 坐标均匀随机
particle_locations[:, 1] = (np.random.rand(N_PARTICLES) * AREA_HEIGHT_U).astype(np.int64)  # y 坐标均匀随机
particle_angles = np.random.rand(N_PARTICLES) * 2 * np.pi  # 每个粒子的运动方向角（弧度）[0, 2π)均匀随机

# --- 初始化无人机 ---

UAV_STEP_U = int(round(UAV_SPEED_KM * DT * SCALE))  # 无人机每步移动的距离（整数单位）
uav_angle = 0  # 无人机当前移动方向角（弧度），初始向右（0度）
uav_turning_angle = 0  # 无人机每步的转向角（弧度），初始为0，后续根据边界情况调整

uav_pos_u = np.array([0, 0], dtype=np.int64)  # 无人机从左下角出发，单位为缩放后的整数坐标
uav_kpt_from_x_u = 0  # 无人机上次kpt的 x 坐标 (整数单位)
uav_kpt_from_y_u = 0  # 无人机上次kpt的 y 坐标 (整数单位)
uav_kpt__to__x_u = 0  # 无人机下次kpt的 x 坐标 (整数单位)
uav_kpt__to__y_u = 0  # 无人机下次kpt的 y 坐标 (整数单位)

# --- 模拟循环变量 ---
history_count = []  # 记录每一步剩余粒子数，用于最终绘图
time_elapsed = 0  # 累计仿真时间（小时）

particle_step_u = int(round(TARGET_SPEED_KM * DT * SCALE))  # 目标每步移动的距离 (整数单位)

# ------------------------------------------移动控制相关函数------------------------------------------

# --- 粒子位置更新函数 ---

def update_particles(p, ang, step_u):
    """按当前速度和方向更新粒子位置，并在边界处做反弹处理。"""
    dx = np.rint(step_u * np.cos(ang)).astype(np.int64)  # x 方向位移（按角度投影后取整）
    dy = np.rint(step_u * np.sin(ang)).astype(np.int64)  # y 方向位移（按角度投影后取整）
    p[:, 0] += dx  # 批量更新所有粒子的 x 坐标
    p[:, 1] += dy  # 批量更新所有粒子的 y 坐标

    # 边界检查：超出左右边界时，方向角按 x 轴法线反射
    mask_x = (p[:, 0] < 0) | (p[:, 0] > AREA_WIDTH_U)

    # 边界检查：超出上下边界时，方向角按 y 轴法线反射
    mask_y = (p[:, 1] < 0) | (p[:, 1] > AREA_HEIGHT_U)

    ang[mask_x] = np.where(mask_x, (np.pi - ang[mask_x]) % (2 * np.pi), ang)
    ang[mask_y] = np.where(mask_y, (-ang) % (2 * np.pi), ang)

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

def is_uav_at_top_edge() -> bool:
    return (uav_pos_u[1]) >= AREA_HEIGHT_U
def is_uav_at_bottom_edge() -> bool:
    return (uav_pos_u[1]) <= 0
def is_uav_at_right_edge() -> bool:
    return (uav_pos_u[0]) >= AREA_WIDTH_U
def is_uav_at_left_edge() -> bool:
    return (uav_pos_u[0]) <= 0

def is_uav_2_scan_radius_from_kpt() -> bool:
    return (uav_pos_u[0] - uav_kpt_from_x_u)**2 + (uav_pos_u[1] - uav_kpt_from_y_u)**2 >= (2 * UAV_SCAN_RADIUS_U)**2 

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

# 无人机是否到达右上角/右下角并且用速度方向判定是否完成扫描的判定
def is_uav_at_end_corner() -> bool:
    ret = False
    if(is_uav_at_top_edge() and (uav_pos_u[0] + UAV_SCAN_RADIUS_U) >= AREA_WIDTH_U and is_uav_up()):
        ret = True
    if(is_uav_at_bottom_edge() and (uav_pos_u[0] + UAV_SCAN_RADIUS_U) >= AREA_WIDTH_U and is_uav_down()):
        ret = True
    
    return ret

# uav是否完成扫描的判定
def is_uav_complete_scan() -> bool:
    return is_uav_at_end_corner()

# 无人机移动逻辑
def update_uav() -> bool:
    global uav_pos_u, uav_angle, uav_kpt_from_x_u, uav_kpt_from_y_u, time_elapsed
    
    # 更新无人机方向
    uav_angle += uav_turning_angle  # 这里的角度更新逻辑在后续的边界判断中进行调整
    uav_angle = uav_angle % (2 * np.pi)  # 确保角度在 [0, 2π) 范围内

    # 更新无人机位置
    uav_pos_u[0] += UAV_STEP_U * np.cos(uav_angle)
    uav_pos_u[1] += UAV_STEP_U * np.sin(uav_angle)

    # 边界判断:变向
    # 到达右上角/右下角后终止
    if (is_uav_complete_scan()):
        print(f"无人机完成扫描！耗时: {time_elapsed:.2f} 小时")
        return False

    # 向上到达顶边后向右转
    if (is_uav_up() and is_uav_at_top_edge()):  
        # uav_pos_u[1] = AREA_HEIGHT_U - UAV_SCAN_RADIUS_U  # 修正 y 坐标
        uav_angle = 0  # 向右转
        uav_kpt_from_x_u = uav_pos_u[0]  # 记录kpt的 x 坐标
        uav_kpt_from_y_u = uav_pos_u[1]  # 记录kpt的 y 坐标
        
    # 向下到达底边后向右转
    if(is_uav_down() and is_uav_at_bottom_edge()):  
        # uav_pos_u[1] = UAV_SCAN_RADIUS_U  # 修正 y 坐标
        uav_angle = 0  # 向右转
        uav_kpt_from_x_u = uav_pos_u[0]  # 记录kpt的 x 坐标
        uav_kpt_from_y_u = uav_pos_u[1]  # 记录kpt的 y 坐标
    
    # 即将向右到距离kpt两个扫描半径处后向上/向下转
    if (is_uav_right() and is_uav_2_scan_radius_from_kpt()):  
        if(uav_pos_u[1] < AREA_HEIGHT_U / 2):  # 如果在区域下半部分，向上转
            uav_angle = 0.5 * np.pi
        else:  # 如果在区域上半部分，向下转
            uav_angle = 1.5 * np.pi
        uav_kpt_from_x_u = uav_pos_u[0]  # 记录kpt的 x 坐标
        uav_kpt_from_y_u = uav_pos_u[1]  # 记录kpt的 y 坐标

    return True
    
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------

# --- 4. 模拟循环 ---

# --- 实时可视化准备 ---
plt.ion()  # 开启交互模式
fig, window = plt.subplots(figsize=(8, 6))

# 初始化热力图网格
grid_size = 1 * SCALE  # 调整网格大小以平衡性能和清晰度
x_bins = np.arange(0, AREA_WIDTH_U + grid_size, grid_size)
y_bins = np.arange(0, AREA_HEIGHT_U + grid_size, grid_size)

# 创建初始空图层
im = window.imshow(np.zeros((len(y_bins)-1, len(x_bins)-1)),    # 注意这里的维度是 (y_bins-1, x_bins-1)，因为 histogram2d 的输出是这样的
               extent=(0, AREA_WIDTH_KM, 0, AREA_HEIGHT_KM),    # 设置坐标轴范围为实际的 km 单位
               origin='lower',                                  # 设置 origin='lower' 左下角为原点
               cmap='YlOrRd',                                   # 使用 YlOrRd 颜色映射
            #    cmap='coolwarm',                                 # 使用 coolwarm 颜色映射
            #    cmap='magma',                                    # 使用 magma 颜色映射
            #    cmap='inferno',                                  # 使用 inferno 颜色映射
            #    cmap='viridis',                                  # 使用 viridis 颜色映射
            #    cmap='hot',                                      # 使用 hot 颜色映射
               animated=True                                    # 设置为动画模式以提高更新效率
            )    
cbar = fig.colorbar(im, ax=window, label='粒子密度 (粒子数/km²) - Particle Density (particles/km²)')
# uav_dot, = window.plot([], [], 'ro', markersize=3, label='UAV') # 绘制无人机位置点
uav_dot, = window.plot([], [], 'ro', markersize=3) # 绘制无人机位置点
window.set_title('实时粒子密度 (Real-time Particle Density)')
window.set_xlabel('X (km)')
window.set_ylabel('Y (km)')
window.legend()

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

# ------------------------------------------

# --- 5. 模拟循环 ---
print("开始搜索模拟...")
for step in range(MAX_STEPS):
    # A. 目标移动
    particle_locations, particle_angles = update_particles(particle_locations, particle_angles, particle_step_u)

    # B. 无人机移动
    


    # C. 判定捕获
    delta = particle_locations - uav_pos_u
    keep_mask = np.sum(delta**2, axis=1) > (UAV_SCAN_RADIUS_U**2)
    particle_locations = particle_locations[keep_mask]
    particle_angles = particle_angles[keep_mask]
    
    history_count.append(len(particle_locations))  # 记录当前剩余粒子数
    time_elapsed += DT  # 更新时间累计

    # D. 实时更新绘图 (建议每隔 N 个步长更新一次以提高运行速度)
    if step % STEPS_TO_UPDATE == 0: 
        density_matrix = get_counts_in_grids(particle_locations)
        im.set_data(density_matrix)
        im.set_clim(vmin=0, vmax=np.max(density_matrix) if len(particle_locations)>0 else 1)
        
        # 更新无人机位置标记
        uav_dot.set_data([uav_pos_u[0]/SCALE], [uav_pos_u[1]/SCALE])
        
        plt.draw()
        plt.pause(0.001) # 暂停微小时间以刷新画布

    # 终止条件
    if is_uav_complete_scan() or len(particle_locations) == 0:
        break

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
