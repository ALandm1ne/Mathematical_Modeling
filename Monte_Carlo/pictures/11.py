"""
图11 动态路径规划前后搜索顺序对比图
=========================================

关键内容：
- 左图：原始静态搜索顺序（初始规划）
- 右图：动态重规划后的搜索顺序（根据实时概率场调整）

显示元素：
- 条带编号与搜索先后顺序  
- 高概率区域视觉强调（色阶变化）
- 搜索策略说明与图例

论证目标：
这张图明确证明系统存在"边界触发 -> 概率评估 -> 条带重选"的完整 pipeline，
而非仅仅"条带扫描 + 马尔科夫预测"的静态模式。
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# 添加父路径以导入config与utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import NumericCoreConfig

# ============================================================================
# 常量定义
# ============================================================================

# 搜索区域
AREA_WIDTH_KM = 306.0
AREA_HEIGHT_KM = 444.0

# 条带参数
STRIP_WIDTH_KM = 40  # 每条带的宽度（km）

# 绘图参数
FIG_SIZE = (16.0, 7.0)  # (宽, 高) 英寸
DPI = 150

# 颜色方案
CMAP_STATIC = "Blues"  # 静态顺序用蓝色系
CMAP_DYNAMIC = "RdYlBu_r"  # 动态顺序用热力图

# ============================================================================
# 辅助函数
# ============================================================================


def load_search_strategy_data(results_dir: str) -> pd.DataFrame | None:
    """加载搜索策略数据（动态重规划记录）"""
    csv_path = os.path.join(results_dir, "search_strategy_dynamic.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Using default static strategy.")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded search strategy data: {len(df)} records")
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def get_strip_grid(width_km: float, height_km: float, strip_width_km: float) -> tuple:
    """
    根据区域尺寸和条带宽度生成网格。
    返回：(n_strips, strip_centers_km, strip_boundaries_km)
    """
    n_strips = int(np.ceil(width_km / strip_width_km))
    # strip_centers_km[i] 是第i条带的中心x坐标（km）
    strip_centers_km = np.array([
        i * strip_width_km + strip_width_km / 2
        for i in range(n_strips)
    ])
    # strip_boundaries_km[i] 和 strip_boundaries_km[i+1] 是第i条带的左右边界
    strip_boundaries_km = np.arange(0, width_km + strip_width_km, strip_width_km)
    
    return n_strips, strip_centers_km, strip_boundaries_km


def build_static_order(n_strips: int) -> list:
    """
    构造静态搜索顺序（从左到右，按初始条带编号）。
    返回：[(strip_id, rank), ...]
    """
    return [(i, i + 1) for i in range(n_strips)]


def build_dynamic_order(search_df: pd.DataFrame | None) -> dict:
    """
    从搜索策略数据构造动态搜索顺序。
    返回：{strip_id: (rank, time_triggered_h, is_high_priority)}
    """
    if search_df is None or search_df.empty:
        return {}
    
    dynamic_order = {}
    for rank, (_, row) in enumerate(search_df.iterrows(), start=1):
        strip_id = int(row["selected_strip_id"])
        time_h = float(row["time_h"])
        is_high_priority = bool(row.get("is_high_priority", False))
        dynamic_order[strip_id] = (rank, time_h, is_high_priority)
    
    return dynamic_order


def visualize_search_order(
    ax,
    n_strips: int,
    strip_boundaries_km: np.ndarray,
    strip_centers_km: np.ndarray,
    height_km: float,
    search_order: dict | list,
    title: str,
    is_dynamic: bool = False,
    cmap_name: str = "Blues",
) -> None:
    """
    绘制搜索顺序可视化。
    
    参数：
    - ax: matplotlib 轴对象
    - n_strips: 条带总数
    - strip_boundaries_km, strip_centers_km: 条带网格信息
    - height_km: 搜索区域高度
    - search_order: 搜索顺序信息
      - 若是静态：[(strip_id, rank), ...]
      - 若是动态：{strip_id: (rank, time_h, is_high_priority)}
    - title: 子图标题
    - is_dynamic: 是否为动态顺序（影响色彩方案）
    - cmap_name: 颜色图名称
    """
    
    ax.set_xlim(0, np.max(strip_boundaries_km))
    ax.set_ylim(0, height_km)
    ax.set_aspect("auto")
    
    # 背景色（浅灰）
    ax.add_patch(mpatches.Rectangle(
        (0, 0), np.max(strip_boundaries_km), height_km,
        facecolor="lightgray", alpha=0.1, zorder=0
    ))
    
    # 获取颜色映射
    cmap = plt.get_cmap(cmap_name)
    
    # 遍历每条带并绘制
    if isinstance(search_order, list):
        # 静态顺序
        for strip_id, rank in search_order:
            if strip_id >= len(strip_boundaries_km) - 1:
                continue
            
            x_left = strip_boundaries_km[strip_id]
            x_right = strip_boundaries_km[strip_id + 1]
            x_center = strip_centers_km[strip_id]
            
            # 条带颜色：按顺序从浅到深
            norm_rank = rank / len(search_order)  # 0 ~ 1
            color = cmap(norm_rank)
            
            # 绘制条带矩形
            ax.add_patch(mpatches.Rectangle(
                (x_left, 0), x_right - x_left, height_km,
                facecolor=color, edgecolor="black", linewidth=0.5,
                alpha=0.6, zorder=1
            ))
            
            # 条带编号
            ax.text(
                x_center, height_km / 2, str(strip_id),
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="black" if norm_rank < 0.7 else "white",
                zorder=3
            )
            
            # 搜索顺序标号（顶部）
            ax.text(
                x_center, height_km - 10,
                f"#{rank}",
                ha="center", va="top",
                fontsize=9, fontweight="bold",
                color="darkblue", zorder=3
            )
    
    else:
        # 动态顺序（字典）
        max_rank = max([v[0] for v in search_order.values()]) if search_order else 1
        
        for strip_id, (rank, time_h, is_high_priority) in search_order.items():
            if strip_id >= len(strip_boundaries_km) - 1:
                continue
            
            x_left = strip_boundaries_km[strip_id]
            x_right = strip_boundaries_km[strip_id + 1]
            x_center = strip_centers_km[strip_id]
            
            # 条带颜色：高优先级用红色，低优先级用蓝色
            if is_high_priority:
                color = (1.0, 0.3, 0.3, 0.7)  # 红色
            else:
                norm_rank = rank / max(max_rank, 1)
                color = cmap(norm_rank)
            
            # 绘制条带矩形
            ax.add_patch(mpatches.Rectangle(
                (x_left, 0), x_right - x_left, height_km,
                facecolor=color, edgecolor="black", linewidth=1.0,
                alpha=0.7 if is_high_priority else 0.5, zorder=1
            ))
            
            # 条带编号
            ax.text(
                x_center, height_km / 2, str(strip_id),
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if is_high_priority else "black",
                zorder=3
            )
            
            # 搜索顺序标号（顶部）
            order_label = f"#{rank}"
            if is_high_priority:
                order_label += "★"  # 星标表示高优先级
            
            ax.text(
                x_center, height_km - 10,
                order_label,
                ha="center", va="top",
                fontsize=9, fontweight="bold",
                color="red" if is_high_priority else "darkblue",
                zorder=3
            )
    
    # 绘制网格线（条带边界）
    for x in strip_boundaries_km:
        ax.axvline(x, color="black", linestyle="--", linewidth=0.5, alpha=0.3, zorder=0)
    
    # 标题与标签
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("X Position (km)", fontsize=11)
    ax.set_ylabel("Y Position (km)", fontsize=11)
    ax.grid(True, alpha=0.2, zorder=0)


def create_figure_11(results_dir: str | None = None) -> str:
    """
    生成图11：动态路径规划前后搜索顺序对比。
    
    参数：
    - results_dir: 结果目录（若有搜索策略数据）
    
    返回：
    - 输出图片路径
    """
    
    # 初始化
    n_strips, strip_centers_km, strip_boundaries_km = get_strip_grid(
        AREA_WIDTH_KM, AREA_HEIGHT_KM, STRIP_WIDTH_KM
    )
    
    # 静态顺序
    static_order = build_static_order(n_strips)
    
    # 动态顺序（若可用）
    search_df = None
    if results_dir:
        search_df = load_search_strategy_data(results_dir)
    
    dynamic_order = build_dynamic_order(search_df)
    
    # 创建图表
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=FIG_SIZE, dpi=DPI, sharey=True
    )
    
    # 左图：静态顺序
    visualize_search_order(
        ax_left,
        n_strips, strip_boundaries_km, strip_centers_km,
        AREA_HEIGHT_KM,
        static_order,
        "Left: Original Static Search Order",
        is_dynamic=False,
        cmap_name=CMAP_STATIC,
    )
    
    # 右图：动态顺序
    if dynamic_order:
        visualize_search_order(
            ax_right,
            n_strips, strip_boundaries_km, strip_centers_km,
            AREA_HEIGHT_KM,
            dynamic_order,
            "Right: Dynamic Reordered by Probability Field",
            is_dynamic=True,
            cmap_name=CMAP_DYNAMIC,
        )
        
        # 添加图例说明（右图）
        legend_elements = [
            mpatches.Patch(facecolor=(1.0, 0.3, 0.3, 0.7), edgecolor="black",
                          label="High-probability priority (★)"),
            mpatches.Patch(facecolor=(0.3, 0.3, 1.0, 0.5), edgecolor="black",
                          label="Lower-priority strips"),
        ]
        ax_right.legend(handles=legend_elements, loc="upper right", fontsize=10)
    else:
        # 若无动态数据，右图显示同样的静态顺序
        visualize_search_order(
            ax_right,
            n_strips, strip_boundaries_km, strip_centers_km,
            AREA_HEIGHT_KM,
            static_order,
            "Right: No Dynamic Reordering Detected\n(Reordering disabled or no events)",
            is_dynamic=False,
            cmap_name=CMAP_STATIC,
        )
        ax_right.text(
            0.5, 0.5, "No Dynamic Replanning Data",
            transform=ax_right.transAxes,
            ha="center", va="center",
            fontsize=16, fontweight="bold",
            color="red", alpha=0.5,
            zorder=100
        )
    
    # 整体标题
    fig.suptitle(
        "Figure 11: Dynamic Path Replanning - Search Order Comparison\n"
        "Left: Initial static plan | Right: Real-time probability-driven reordering",
        fontsize=15, fontweight="bold", y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    output_dir = Path(__file__).parent
    output_path = output_dir / "11.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"✓ Figure 11 saved: {output_path}")
    
    plt.close()
    
    return str(output_path)


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # 检查是否有结果目录作为命令行参数
    results_dir = None
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        if not os.path.isdir(results_dir):
            print(f"Warning: Results directory not found: {results_dir}")
            results_dir = None
    
    # 生成图表
    output_file = create_figure_11(results_dir)
    print(f"Figure 11 generation complete: {output_file}")
