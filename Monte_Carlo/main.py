import numpy as np  # 数值计算库，用于向量化更新大量粒子
import matplotlib.pyplot as plt  # 绘图库，用于画剩余粒子数量曲线

# ----------------------------------------

# --- 1. 参数设置 ---
AREA_WIDTH = 306.0  # 模拟区域宽度，单位 km（经度方向跨度）
AREA_HEIGHT = 444.0  # 模拟区域高度，单位 km（纬度方向跨度）
N_PARTICLES = 1000000  # 初始粒子数，表示潜在目标样本数量
TARGET_SPEED = 30.0  # 目标运动速度，单位 km/h
UAV_SPEED = 150.0  # 无人机巡航速度，单位 km/h
UAV_SCAN_RADIUS = 15.0  # 无人机传感器扫描半径，单位 km
DT = 0.01  # 仿真时间步长，单位小时

# --- 定点缩放设置 ---
# 用整数网格单位替代浮点坐标：1 km -> 1000 单位（1 单位=1 m）
SCALE = 1000  # 坐标缩放系数
AREA_WIDTH_U = int(round(AREA_WIDTH * SCALE))  # 区域宽度（整数单位）
AREA_HEIGHT_U = int(round(AREA_HEIGHT * SCALE))  # 区域高度（整数单位）
UAV_SCAN_RADIUS_U = int(round(UAV_SCAN_RADIUS * SCALE))  # 扫描半径（整数单位）

# ----------------------------------------

# --- 2. 初始化粒子 (目标) ---
# 粒子位置数组 shape=(N,2)，每行是 [x, y]，使用 int64 避免累计浮点误差
particles = np.empty((N_PARTICLES, 2), dtype=np.int64)
particles[:, 0] = (np.random.rand(N_PARTICLES) * AREA_WIDTH_U).astype(np.int64)  # x 坐标均匀随机
particles[:, 1] = (np.random.rand(N_PARTICLES) * AREA_HEIGHT_U).astype(np.int64)  # y 坐标均匀随机
angles = np.random.rand(N_PARTICLES) * 2 * np.pi  # 每个粒子的运动方向角（弧度）

# ----------------------------------------

# --- 3. 初始化无人机 ---
uav_pos = np.array([0, 0], dtype=np.int64)  # 无人机从左下角出发，单位为缩放后的整数坐标


def update_particles(p, ang, v, dt):
    """按当前速度和方向更新粒子位置，并在边界处做反弹处理。"""
    step_u = int(round(v * dt * SCALE))  # 单步位移长度（整数单位）
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


# --- 4. 模拟循环 ---
history_count = []  # 记录每一步剩余粒子数，用于最终绘图
time_elapsed = 0  # 累计仿真时间（小时）

print("开始搜索模拟...")  # 启动提示
for step in range(100):  # 固定迭代 100 个时间步
    # A. 目标移动
    particles, angles = update_particles(particles, angles, TARGET_SPEED, DT)

    # B. 无人机沿简化的 Z 字路径移动（示例策略）
    uav_pos[1] += int(round(UAV_SPEED * DT * SCALE))  # 先沿 y 方向推进
    if uav_pos[1] > AREA_HEIGHT_U:  # 到达顶边后换列扫描
        uav_pos[1] = 0  # y 回到底部
        uav_pos[0] += 2 * UAV_SCAN_RADIUS_U  # x 向右移动一个扫描直径

    # C. 判定捕获：使用平方距离比较，避免开方操作
    delta = particles - uav_pos  # 每个粒子到无人机位置的坐标差
    d2 = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]  # 粒子到无人机的距离平方
    r2 = UAV_SCAN_RADIUS_U * UAV_SCAN_RADIUS_U  # 扫描半径平方

    # 仅保留未被扫描到的粒子
    keep_mask = d2 > r2
    particles = particles[keep_mask]
    angles = angles[keep_mask]

    history_count.append(len(particles))  # 记录当前剩余粒子数
    time_elapsed += DT  # 更新时间累计

    if len(particles) == 0:  # 若已无粒子，则提前结束仿真
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
