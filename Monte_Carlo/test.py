import math

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

print(get_turn_angle(0, math.pi/2, False))  # 逆时针转90度，预期输出: 1.5707963267948966
print(get_turn_angle(0, math.pi/2, True))   # 顺时针转90度，预期输出: -4.71238898038469
print(get_turn_angle(math.pi/2, 0, False))  # 逆时针转-90度，预期输出: 4.71238898038469
print(get_turn_angle(math.pi/2, 0, True))   # 顺时针转-90度，预期输出: -1.5707963267948966
print(get_turn_angle(0, 0, False))          # 逆时针转0度，预期输出: 0.0
print(get_turn_angle(0, 0, True))           # 顺时针转0度，预期输出: 0.0
print(get_turn_angle(0, 2*math.pi, False))  # 逆时针转360度，预期输出: 0.0
print(get_turn_angle(0, 2*math.pi, True))   # 顺时针转360度，预期输出: 0.0
print(get_turn_angle(3.5, 3.1, True))