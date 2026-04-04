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
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman"]
plt.rcParams["axes.unicode_minus"] = False

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
SCALE_U_PER_KM = NumericCoreConfig().scale

# 绘图参数
FIG_SIZE = (16.0, 7.0)  # (宽, 高) 英寸
DPI = 150

# 颜色方案
CMAP_STATIC = "Blues"  # 静态顺序用蓝色系
CMAP_DYNAMIC = "RdYlBu_r"  # 动态顺序用热力图

# ============================================================================
# 辅助函数
# ============================================================================


def load_search_strategy_data(results_dir: str) -> list[dict[str, str]] | None:
    """加载搜索策略数据（动态重规划记录）"""
    csv_path = os.path.join(results_dir, "search_strategy_dynamic.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Using default static strategy.")
        return None
    
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"Loaded search strategy data: {len(rows)} records")
        return rows
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


def build_dynamic_order(search_rows: list[dict[str, str]] | None) -> dict:
    """
        从搜索策略数据构造动态搜索顺序。
        返回：
            {
                "summary": {strip_id: {"first_rank": int, "is_high_priority": bool, "visit_count": int}},
                "events": [(rank, strip_id, is_high_priority), ...],
            }
    """
    if not search_rows:
                return {"summary": {}, "events": []}

    search_rows = sorted(
        search_rows,
        key=lambda row: (float(row.get("time_h", 0.0)), int(row.get("step_idx", 0))),
    )
    
    dynamic_summary: dict[int, dict[str, int | bool]] = {}
    dynamic_events: list[tuple[int, int, bool]] = []
    for rank, row in enumerate(search_rows, start=1):
        # CSV 中 selected_strip_id 是条带左边界坐标（u），需换算为条带索引。
        strip_left_u = int(row["selected_strip_id"])
        strip_left_km = float(strip_left_u) / float(SCALE_U_PER_KM)
        strip_id = int(np.floor(strip_left_km / STRIP_WIDTH_KM))
        is_high_priority = str(row.get("is_high_priority", "")).lower() in {"1", "true", "yes"}
        dynamic_events.append((rank, strip_id, is_high_priority))

        if strip_id not in dynamic_summary:
            dynamic_summary[strip_id] = {
                "first_rank": rank,
                "is_high_priority": is_high_priority,
                "visit_count": 1,
            }
        else:
            dynamic_summary[strip_id]["visit_count"] = int(dynamic_summary[strip_id]["visit_count"]) + 1
            dynamic_summary[strip_id]["is_high_priority"] = bool(dynamic_summary[strip_id]["is_high_priority"]) or is_high_priority

    return {"summary": dynamic_summary, "events": dynamic_events}


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
    static_reference: list[int] | None = None,
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
    
    # 中心线轨迹绘制（不再使用扫过区域填色）。
    def _shade_by_time(base_rgb: tuple[float, float, float], norm_t: float) -> tuple[float, float, float]:
        """按时间深浅着色：早期更浅，晚期更深。"""
        norm_t = max(0.0, min(1.0, float(norm_t)))
        # 与白色线性混合，0->浅色，1->基色
        return (
            (1.0 - norm_t) * 0.75 + norm_t * base_rgb[0],
            (1.0 - norm_t) * 0.75 + norm_t * base_rgb[1],
            (1.0 - norm_t) * 0.75 + norm_t * base_rgb[2],
        )

    if isinstance(search_order, list):
        decisions = [int(strip_id) for strip_id, _ in search_order]
        high_flags = [False] * len(decisions)
    else:
        events = search_order.get("events", []) if isinstance(search_order, dict) else []
        decisions = [int(strip_id) for _, strip_id, _ in events]
        high_flags = [bool(is_high_priority) for _, _, is_high_priority in events]

    visit_index_by_strip: dict[int, int] = {}
    traj_x: list[float] = []
    traj_y: list[float] = []
    n_events = max(len(decisions), 1)
    for i, strip_id in enumerate(decisions, start=1):
        if strip_id < 0 or strip_id >= len(strip_centers_km):
            continue

        visit_idx = visit_index_by_strip.get(strip_id, 0)
        visit_index_by_strip[strip_id] = visit_idx + 1

        x_center = float(strip_centers_km[strip_id]) + (visit_idx % 5 - 2) * 0.8
        norm_time = (i - 1) / max(len(decisions) - 1, 1)

        if is_dynamic:
            same_as_static = bool(static_reference is not None and i - 1 < len(static_reference) and strip_id == static_reference[i - 1])
            base_rgb = (65 / 255.0, 105 / 255.0, 225 / 255.0) if same_as_static else (220 / 255.0, 20 / 255.0, 60 / 255.0)
            line_color = _shade_by_time(base_rgb, norm_time)
        else:
            same_as_static = True
            base_rgb = (65 / 255.0, 105 / 255.0, 225 / 255.0)
            line_color = _shade_by_time(base_rgb, norm_time)

        # 每个事件使用独立的时间分层区间，避免整幅竖线重合。
        y0 = height_km * float(i - 1) / float(n_events)
        y1 = height_km * float(i) / float(n_events)
        y_mid = 0.5 * (y0 + y1)

        ax.plot(
            [x_center, x_center],
            [y0, y1],
            color=line_color,
            linewidth=2.0,
            alpha=0.95,
            zorder=2,
        )
        traj_x.append(x_center)
        traj_y.append(y_mid)

        label = f"#{i}"
        if is_dynamic and high_flags[i - 1]:
            label += "[HIGH]"

        if is_dynamic:
            # 右图编号严格按时间从下到上递增：#1 在底部，后续逐步上移。
            y_label = 4.0 + (float(i - 1) / max(float(n_events - 1), 1.0)) * (height_km - 8.0)
            label_va = "bottom"
        else:
            y_label = y1 - 2.0
            label_va = "top"

        ax.text(
            x_center,
            y_label,
            label,
            ha="center",
            va=label_va,
            fontsize=7,
            fontweight="bold",
            color=line_color,
            zorder=4,
        )

        if visit_idx == 0:
            ax.text(
                x_center,
                y_mid,
                str(strip_id),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
                zorder=4,
            )

    # 连续轨迹线：显示决策随时间在 x 方向的迁移趋势。
    if len(traj_x) >= 2:
        ax.plot(traj_x, traj_y, color="gray", linewidth=1.0, alpha=0.45, linestyle="--", zorder=1.5)

    if is_dynamic:
        same_count = 0
        if static_reference is not None:
            same_count = sum(1 for idx, sid in enumerate(decisions) if idx < len(static_reference) and sid == static_reference[idx])
        ax.text(
            0.01,
            0.99,
            f"Dynamic events shown: {len(decisions)} | Same as static: {same_count}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="gray"),
            zorder=5,
        )
    
    # 绘制网格线（条带边界）
    for x in strip_boundaries_km:
        ax.axvline(x, color="black", linestyle="--", linewidth=0.5, alpha=0.3, zorder=0)
    
    # 标题与标签
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("X Position (km)", fontsize=11)
    ax.set_ylabel("Y Position (km)", fontsize=11)
    ax.grid(True, alpha=0.2, zorder=0)


def create_figure_11(results_dir: str | None = None, output_path: str | None = None) -> str:
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
    search_rows = None
    if results_dir:
        search_rows = load_search_strategy_data(results_dir)
    
    dynamic_order = build_dynamic_order(search_rows)
    static_reference = [strip_id for strip_id, _ in static_order]
    
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
        "Original Static Strip Order",
        is_dynamic=False,
        cmap_name=CMAP_STATIC,
        static_reference=static_reference,
    )
    
    # 右图：动态顺序
    if dynamic_order:
        visualize_search_order(
            ax_right,
            n_strips, strip_boundaries_km, strip_centers_km,
            AREA_HEIGHT_KM,
            dynamic_order,
            "Dynamic Reordered Search Order",
            is_dynamic=True,
            cmap_name=CMAP_DYNAMIC,
            static_reference=static_reference,
        )
        
        # 添加图例说明（右图）
        legend_elements = [
            mpatches.Patch(facecolor="royalblue", edgecolor="royalblue", label="Same decision as static (blue centerline)"),
            mpatches.Patch(facecolor="crimson", edgecolor="crimson", label="Different decision from static (red centerline)"),
            mpatches.Patch(facecolor="white", edgecolor="gray", label="[HIGH]: high-probability-priority event"),
            mpatches.Patch(facecolor="lightgray", edgecolor="gray", label="Color depth encodes time: lighter=earlier, darker=later"),
        ]
        ax_right.legend(handles=legend_elements, loc="upper right", fontsize=10)
    else:
        # 若无动态数据，右图显示同样的静态顺序
        visualize_search_order(
            ax_right,
            n_strips, strip_boundaries_km, strip_centers_km,
            AREA_HEIGHT_KM,
            static_order,
            "No Dynamic Reordering Detected",
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
        "Figure 11 Dynamic Path Replanning Search-Order Comparison\n"
        "Left: Static strip order | Right: Probability-field-driven dynamic order",
        fontsize=15, fontweight="bold", y=0.98
    )
    
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    
    # 保存图片
    if output_path is None:
        save_path = Path(__file__).parent / "11.png"
    else:
        save_path = Path(output_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"✓ Figure 11 saved: {save_path}")
    
    plt.close()
    
    return str(save_path)


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # 检查是否有结果目录作为命令行参数
    results_dir = None
    output_path = None
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        if not os.path.isdir(results_dir):
            print(f"Warning: Results directory not found: {results_dir}")
            results_dir = None
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    # 生成图表
    output_file = create_figure_11(results_dir, output_path)
    print(f"Figure 11 generation complete: {output_file}")
