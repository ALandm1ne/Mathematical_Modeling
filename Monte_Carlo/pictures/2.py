import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

# 设置学术字体
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif']

def draw_partition_subplots():
    # 模拟数据：由于基地在西北，左侧(UAV1)航渡近，负责区域更宽；右侧(UAVn)航渡远，宽度更窄
    # 这种非均匀性是“等时间原则”的核心体现
    case_data = {
        2: [0.55, 0.45],          # N=2 时宽度占比
        3: [0.40, 0.33, 0.27],    # N=3 时宽度占比
        4: [0.32, 0.28, 0.22, 0.18] # N=4 时宽度占比
    }
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F1C40F']
    total_W = 306.0 # 124E-127E 跨度约 306km
    total_L = 444.0 # 21N-25N 跨度约 444km
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=300)
    
    for i, (N, ratios) in enumerate(case_data.items()):
        ax = axes[i]
        current_x = 0
        for j in range(N):
            sub_w = total_W * ratios[j]
            # 绘制子区域
            rect = Rectangle((current_x, 0), sub_w, total_L, 
                             facecolor=colors[j], alpha=0.3, edgecolor='black', lw=1)
            ax.add_patch(rect)
            
            # 标注子区域编号
            ax.text(current_x + sub_w/2, total_L/2, f"UAV{j+1}", 
                    ha='center', va='center', fontweight='bold', fontsize=10)
            
            # 标注宽度 W_i (Unicode 避免 LaTeX 报错)
            ax.text(current_x + sub_w/2, -20, f"W{j+1}", ha='center', fontsize=9)
            
            # 标注入口点 (Entry Point)
            ax.plot(current_x + sub_w/2, 0, 'rv', markersize=6) 
            if j == 0:
                ax.text(current_x + sub_w/2, -45, "Entry P.", ha='center', color='red', fontsize=8)
            
            current_x += sub_w
            
        # 子图格式化
        ax.set_xlim(-20, total_W + 20)
        ax.set_ylim(-60, total_L + 20)
        ax.set_title(f"Case: N = {N}", fontsize=12, pad=15)
        ax.set_aspect('equal')
        ax.invert_yaxis() # 保持南向为正
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Fig. 2 Schematic of Optimal Sub-area Partitioning (Equal-Time Principle)", 
                 fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig("2.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_partition_subplots()