import numpy as np  # 数值计算库，用于向量化更新大量粒子
import matplotlib.pyplot as plt  # 绘图库，用于画剩余粒子数量曲线
from matplotlib.markers import MarkerStyle  # 用于自定义无人机标记样式

# ----------------------------------------

# --- 1. 参数设置 ---
AREA_WIDTH_KM = 306.0  # 模拟区域宽度，单位 km（经度方向跨度）
AREA_HEIGHT_KM = 444.0  # 模拟区域高度，单位 km（纬度方向跨度）
N_PARTICLES = 10000  # 初始粒子数，表示潜在目标样本数量
TARGET_SPEED_KM = 30.0  # 目标运动速度，单位 km/h
UAV_SPEED_KM = 150.0  # 无人机巡航速度，单位 km/h
UAV_SCAN_RADIUS_KM = 15.0  # 无人机传感器扫描半径，单位 km
DT = 0.01  # 仿真时间步长，单位小时
MAX_STEPS = 10000  # 最大仿真步数

# --- 定点缩放设置 ---
# 用整数网格单位替代浮点坐标：1 km -> 1000 单位（1 单位=1 m）
SCALE = 1000  # 坐标缩放系数
AREA_WIDTH_U = int(round(AREA_WIDTH_KM * SCALE))  # 区域宽度（整数单位）
AREA_HEIGHT_U = int(round(AREA_HEIGHT_KM * SCALE))  # 区域高度（整数单位）
UAV_SCAN_RADIUS_U = int(round(UAV_SCAN_RADIUS_KM * SCALE))  # 扫描半径（整数单位）

# ----------------------------------------

# --- 2. 初始化粒子 (目标) ---
# 粒子位置数组 shape=(N,2)，每行是 [x, y]，使用 int64 避免累计浮点误差
particle_locations = np.empty((N_PARTICLES, 2), dtype=np.int64)  # 初始化粒子位置数组: N行2列，整数类型
particle_locations[:, 0] = (np.random.rand(N_PARTICLES) * AREA_WIDTH_U ).astype(np.int64)  # x 坐标均匀随机
particle_locations[:, 1] = (np.random.rand(N_PARTICLES) * AREA_HEIGHT_U).astype(np.int64)  # y 坐标均匀随机
particle_angles = np.random.rand(N_PARTICLES) * 2 * np.pi  # 每个粒子的运动方向角（弧度）

# ----------------------------------------

# --- 3. 初始化无人机 ---
uav_pos = np.array([0, 0], dtype=np.int64)  # 无人机从左下角出发，单位为缩放后的整数坐标


def update_particles(p, ang, step_u):
    """按当前速度和方向更新粒子位置，并在边界处做反弹处理。"""
    dx = np.rint(step_u * np.cos(ang)).astype(np.int64)  # x 方向位移（按角度投影后取整）
    dy = np.rint(step_u * np.sin(ang)).astype(np.int64)  # y 方向位移（按角度投影后取整）
    p[:, 0] += dx  # 批量更新所有粒子的 x 坐标
    p[:, 1] += dy  # 批量更新所有粒子的 y 坐标

    # 边界检查：超出左右边界时，方向角按 x 轴法线反射
    mask_x = (p[:, 0] < 0) | (p[:, 0] > AREA_WIDTH_U)
    ang[mask_x] = np.pi - ang[mask_x]

    # 边界检查：超出上下边界时，方向角按 y 轴法线反射
    mask_y = (p[:, 1] < 0) | (p[:, 1] > AREA_HEIGHT_U)
    ang[mask_y] = -ang[mask_y]

    # 坐标裁剪回合法区域，避免粒子停留在边界外
    p[:, 0] = np.clip(p[:, 0], 0, AREA_WIDTH_U)
    p[:, 1] = np.clip(p[:, 1], 0, AREA_HEIGHT_U)
    return p, ang  # 返回更新后的粒子坐标和方向角

# ------------------------------------------
# ------------------------------------------
# ------------------------------------------

# --- 4. 模拟循环 ---

# 前置变量
history_count = []  # 记录每一步剩余粒子数，用于最终绘图
time_elapsed = 0  # 累计仿真时间（小时）

target_step_u = int(round(TARGET_SPEED_KM * DT * SCALE))  # 目标每步移动的距离（整数单位）

# ------------------------------------------

# UAV移动控制相关变量和函数

UAV_STEP_U = int(round(UAV_SPEED_KM * DT * SCALE))  # 无人机每步移动的距离（整数单位）
uav_direction_x = 0  # 无人机当前移动方向：1=向右，-1=向左（初始向右）
uav_direction_y = 1  # 无人机当前移动方向：1=向上，-1=向下（初始向上）
uav_last_kpt_x = 0  # 无人机上次kpt扫描的 x 坐标（整数单位）
uav_last_kpt_y = 0  # 无人机上次kpt扫描的 y 坐标（整数单位）

def uav_up():
    global uav_direction_x, uav_direction_y
    uav_direction_x = 0
    uav_direction_y = 1
def uav_down():
    global uav_direction_x, uav_direction_y
    uav_direction_x = 0
    uav_direction_y = -1
def uav_right():
    global uav_direction_x, uav_direction_y
    uav_direction_x = 1
    uav_direction_y = 0
def uav_left():
    global uav_direction_x, uav_direction_y
    uav_direction_x = -1
    uav_direction_y = 0

def is_uav_up():
    return (uav_direction_x == 0) and (uav_direction_y == 1)
def is_uav_down():
    return (uav_direction_x == 0) and (uav_direction_y == -1)
def is_uav_right():
    return (uav_direction_x == 1) and (uav_direction_y == 0)
def is_uav_left():
    return (uav_direction_x == -1) and (uav_direction_y == 0)

def is_uav_at_top_edge():
    return (uav_pos[1] + UAV_SCAN_RADIUS_U) > AREA_HEIGHT_U
def is_uav_at_bottom_edge():
    return (uav_pos[1] - UAV_SCAN_RADIUS_U) < 0
def is_uav_at_right_edge():
    return (uav_pos[0] + UAV_SCAN_RADIUS_U) > AREA_WIDTH_U
def is_uav_at_left_edge():
    return (uav_pos[0] - UAV_SCAN_RADIUS_U) < 0

def is_uav_2_scan_radius_from_kpt():
    return ((uav_pos[0] + UAV_STEP_U) > (uav_last_kpt_x + 2 * UAV_SCAN_RADIUS_U)) or ((uav_pos[0] - UAV_STEP_U) < (uav_last_kpt_x - 2 * UAV_SCAN_RADIUS_U)) or \
           ((uav_pos[1] + UAV_STEP_U) > (uav_last_kpt_y + 2 * UAV_SCAN_RADIUS_U)) or ((uav_pos[1] - UAV_STEP_U) < (uav_last_kpt_y - 2 * UAV_SCAN_RADIUS_U))

# uav是否完成扫描的判定：当无人机在顶边且向上，或在底边且向下时，认为完成扫描
def is_uav_complete_scan():
    return (is_uav_at_top_edge() and is_uav_up()) or (is_uav_at_bottom_edge() and is_uav_down())

# ------------------------------------------
 
### 可视化

# 计算区域 (x1,y1) 到 (x2,y2) 内粒子的数量，返回密度值。
def density(x1,y1,x2,y2):
    count = np.sum((particle_locations[:, 0] >= x1) & (particle_locations[:, 0] < x2) &
                   (particle_locations[:, 1] >= y1) & (particle_locations[:, 1] < y2))
    area = (x2 - x1) * (y2 - y1) / (SCALE * SCALE)  # 区域面积，单位 km^2
    return count / area if area > 0 else 0

# 绘制粒子密度热图，显示区域内粒子分布情况。
def plot_density():
    grid_size = 10 * SCALE  # 10 km 网格
    x_bins = np.arange(0, AREA_WIDTH_U + grid_size, grid_size)
    y_bins = np.arange(0, AREA_HEIGHT_U + grid_size, grid_size)
    density_grid = np.zeros((len(y_bins)-1, len(x_bins)-1))
    
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            density_grid[j, i] = density(x_bins[i], y_bins[j], x_bins[i+1], y_bins[j+1])
    
    plt.imshow(density_grid, extent=(0, AREA_WIDTH_KM, 0, AREA_HEIGHT_KM), origin='lower', cmap='viridis')
    plt.colorbar(label='Particle Density (particles/km²)')
    plt.title('Particle Density Heatmap')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.show()

# ------------------------------------------

# 循环开始
print("开始搜索模拟...")  # 启动提示
for step in range(MAX_STEPS):  # 最大迭代 MAX_STEPS 个时间步
    # A. 目标移动
    particle_locations, particle_angles = update_particles(particle_locations, particle_angles, target_step_u)

    # B. 无人机沿简化的 Z 字路径移动（示例策略）
    uav_pos[0] += UAV_STEP_U * uav_direction_x  # 无人机 x 坐标更新
    uav_pos[1] += UAV_STEP_U * uav_direction_y  # 无人机 y 坐标更新
    
    ### 边界判断:变向
    # 向上到达顶边后向右转
    if (is_uav_up() and is_uav_at_top_edge()):  
        uav_pos[1] = AREA_HEIGHT_U - UAV_SCAN_RADIUS_U  # 修正 y 坐标
        uav_right()  # 向右转
        uav_last_kpt_x = uav_pos[0]  # 记录kpt的 x 坐标
        uav_last_kpt_y = uav_pos[1]  # 记录kpt的 y 坐标
    
    # 即将向右到距离kpt两个扫描半径处后向上/向下转
    if (is_uav_right() and is_uav_2_scan_radius_from_kpt()):  
        if(uav_pos[1] < AREA_HEIGHT_U / 2):  # 如果在区域下半部分，向上转
            uav_up()
        else:  # 如果在区域上半部分，向下转
            uav_down()
        uav_last_kpt_x = uav_pos[0]  # 记录kpt的 x 坐标
        uav_last_kpt_y = uav_pos[1]  # 记录kpt的 y 坐标
    
    # 向下到达底边后向右转
    if(is_uav_down() and is_uav_at_bottom_edge()):  
        uav_pos[1] = UAV_SCAN_RADIUS_U  # 修正 y 坐标
        uav_right()  # 向右转
        uav_last_kpt_x = uav_pos[0]  # 记录kpt的 x 坐标
        uav_last_kpt_y = uav_pos[1]  # 记录kpt的 y 坐标

    # 到达右上角/右下角后终止
    if (is_uav_complete_scan()):
        print(f"无人机完成扫描！耗时: {time_elapsed:.2f} 小时")
        break

    # C. 判定捕获
    delta = particle_locations - uav_pos  # 每个粒子到无人机位置的坐标差
    d2 = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]  # 粒子到无人机的距离平方
    r2 = UAV_SCAN_RADIUS_U * UAV_SCAN_RADIUS_U  # 扫描半径平方

    # 仅保留未被扫描到的粒子
    keep_mask = d2 > r2  # 只有距离大于扫描半径的粒子才保留
    particle_locations = particle_locations[keep_mask]  # 更新粒子位置数组，仅保留未被捕获的粒子
    particle_angles = particle_angles[keep_mask]

    history_count.append(len(particle_locations))  # 记录当前剩余粒子数
    time_elapsed += DT  # 更新时间累计
    
    plot_density()  # 可视化当前粒子密度分布（可选，调试时使用）

    if len(particle_locations) == 0:  # 若已无粒子，则提前结束仿真
        print(f"目标已全部锁定！耗时: {time_elapsed:.2f} 小时")
        break

# --- 5. 结果可视化 ---
plt.figure(figsize=(10, 5))  # 创建画布
plt.plot(history_count)  # 绘制剩余粒子数量曲线
plt.title("Remaining Potential Target Particles Over Time")  # 图标题
plt.xlabel("Time Steps")  # x 轴标签
plt.ylabel("Particle Count")  # y 轴标签
plt.grid(True)  # 显示网格线
plt.show()  # 展示图像窗口
