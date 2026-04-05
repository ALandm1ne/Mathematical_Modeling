"""
Figure 13: Success probability vs. number of UAVs within 10 hours.
Helps answer "minimum number of UAVs needed to find target within 10h".
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


SCRIPT_DIR = Path(__file__).parent.parent
RESULTS_DIR = SCRIPT_DIR / "results"
BASE_X_KM = -314.0
BASE_Y_KM = -323.0
UAV_SPEED_KM_H = 150.0
UNIT_SCALE = 1000.0


@dataclass(frozen=True)
class Figure13Params:
    time_limit_h: float = 10.0
    uav_count_range: tuple[int, int] = (1, 15)  # Test 1-15 UAVs
    runs_per_count: int = 10  # 10 runs per UAV count
    figsize: tuple[float, float] = (12.0, 8.0)
    dpi: int = 200


def generate_uav_path_config(n_uavs: int) -> list[dict]:
    """
    生成 n_uavs 架无人机的路径配置。
    每架无人机负责不同的 x 条带。
    """
    config = []
    
    area_width_u = 306000  # 306 km = 306000 u
    area_height_u = 444000  # 444 km = 444000 u
    scan_radius_u = 20000  # 20 km scan radius
    
    effective_uavs = max(1, n_uavs)
    
    for uav_id in range(effective_uavs):
        # 按 UAV 数量划分责任区，起飞点取责任区左下角向东一个扫描半径。
        x_region_min = (uav_id * area_width_u) // effective_uavs
        x_region_max = ((uav_id + 1) * area_width_u) // effective_uavs
        x_start_u = min(int(x_region_max), int(x_region_min + scan_radius_u))
        start_pos = [x_start_u, 0]
        end_pos = [x_start_u, area_height_u]

        x_start_km = x_start_u / UNIT_SCALE
        y_start_km = 0.0
        ferry_dist_km = math.hypot(x_start_km - BASE_X_KM, y_start_km - BASE_Y_KM)
        start_time_h = ferry_dist_km / UAV_SPEED_KM_H
        
        uav_spec = {
            "_comment": (
                f"UAV#{uav_id}: region=[{x_region_min/1000:.1f},{x_region_max/1000:.1f}]km, "
                f"x_start={x_start_km:.1f}km, start={start_time_h:.3f}h"
            ),
            "uav_id": uav_id,
            "start_time_h": float(start_time_h),
            "start_pos_u": start_pos,
            "segments": [
                {
                    "segment_type": "line",
                    "end_point_u": end_pos
                },
            ]
        }
        config.append(uav_spec)
    
    return config


def run_simulation_with_config(
    config: list[dict],
    timeout_s: int = 600,
    time_limit_h: float = 10.0,
) -> dict | None:
    """
    使用给定的 UAV 配置运行一次模拟，返回成功概率信息。
    """
    venv_python = SCRIPT_DIR / ".venv" / "bin" / "python"
    main_py = SCRIPT_DIR / "main.py"
    
    # 使用默认配置文件（config_templates/uav_paths_n1.json）
    standard_config = SCRIPT_DIR / "config_templates" / "uav_paths_n1.json"
    backup_path = str(standard_config) + ".backup_fig13"
    
    try:
        # 备份原始配置
        if standard_config.exists():
            with open(standard_config, 'r', encoding='utf-8') as f:
                original_config = f.read()
        else:
            original_config = None
        
        # 写入新配置
        with open(standard_config, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # 运行模拟
        result = subprocess.run(
            [str(venv_python), str(main_py)],
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            timeout=timeout_s,
            text=True
        )
        
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            if err:
                print(f"[ERROR] main.py failed: {err.splitlines()[-1]}")
            return None
        
        # 从最新的结果目录读取轨迹数据
        if not RESULTS_DIR.exists():
            return None
        
        latest_dir = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()], reverse=True)
        if not latest_dir:
            return None
        
        latest_result = latest_dir[0]
        trajectory_file = latest_result / "uav_trajectory.csv"
        
        if not trajectory_file.exists():
            return None
        
        # 读取轨迹数据，判断是否在 10 小时内找到目标
        with open(trajectory_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return None
        
        # 找到第一个时间 >= time_limit_h 的行，并计算粒子消减率
        # 成功率定义：1 - (remaining / initial)
        initial_particles = int(rows[0]["remaining_particles"])
        final_remaining = None
        final_time_h = None
        
        for row in rows:
            time_h = float(row['time_h'])
            remaining = int(row['remaining_particles'])
            
            if time_h >= time_limit_h:
                final_remaining = remaining
                final_time_h = time_h
                break

        # 若轨迹未覆盖到 time_limit_h，退化为最后一个时刻数据。
        if final_remaining is None:
            final_remaining = int(rows[-1]["remaining_particles"])
            final_time_h = float(rows[-1]["time_h"])

        success_rate_10h = 1.0 - (float(final_remaining) / float(initial_particles))
        success_rate_10h = max(0.0, min(1.0, success_rate_10h))
        
        return {
            "success_rate_10h": success_rate_10h,
            "initial_particles": initial_particles,
            "final_remaining": final_remaining,
            "final_time_h": final_time_h
        }
    
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"[ERROR] 模拟异常: {e}")
        return None
    finally:
        # 恢复原始配置
        if original_config is not None:
            with open(standard_config, 'w', encoding='utf-8') as f:
                f.write(original_config)


def benchmark_runtime_seconds(params: Figure13Params, sample_runs: int = 3) -> float:
    """实测若干轮 wall time，返回每轮平均秒数，用于总时长估算。"""
    print(f"[基准测速] 开始 {sample_runs} 轮测速（真实运行模式）...")
    samples: list[float] = []
    probe_config = generate_uav_path_config(params.uav_count_range[0])

    for i in range(sample_runs):
        t0 = time.perf_counter()
        result = run_simulation_with_config(
            probe_config,
            timeout_s=600,
            time_limit_h=params.time_limit_h,
        )
        dt = time.perf_counter() - t0
        if result is None:
            print(f"  - 第 {i + 1} 轮失败，耗时 {dt:.2f}s")
            continue
        samples.append(dt)
        print(f"  - 第 {i + 1} 轮: {dt:.2f}s")

    if not samples:
        fallback = 12.0
        print(f"[基准测速] 未获得有效样本，回退到 {fallback:.1f}s/轮")
        return fallback

    avg_s = float(np.mean(samples))
    p90_s = float(np.percentile(samples, 90))
    print(f"[基准测速] 平均 {avg_s:.2f}s/轮, P90 {p90_s:.2f}s/轮")
    return avg_s


def collect_data(
    params: Figure13Params,
    demo_mode: bool = True,
    benchmark_runs: int = 3,
) -> dict[int, list[float]]:
    """
    收集不同无人机数量下的成功率数据。
    返回 {uav_count: [mean_success_rate(0-1), std_success_rate(0-1), uA_success_rate(0-1), run_count]}
    
    demo_mode=True: 快速演示（使用模拟数据）
    demo_mode=False: 真实运行（会需要很长时间）
    """
    min_uavs, max_uavs = params.uav_count_range
    results_by_count = {}
    
    print("=" * 80)
    print("  图13 数据收集: 10 小时成功概率 vs 无人机数量")
    print("=" * 80)
    mode_text = "演示模式（模拟数据）" if demo_mode else "真实运行"
    print(f"模式: {mode_text}")
    print(f"无人机数量范围: {min_uavs}-{max_uavs}")
    print(f"每个数量运行次数: {params.runs_per_count}\n")
    
    if demo_mode:
        # 演示模式：使用模拟的成功概率（典型的 S 形曲线）
        # 基于假设：UAV 数量越多，成功概率越高
        print("[演示模式] 使用模拟数据生成示例曲线...")
        
        for n_uavs in range(min_uavs, max_uavs + 1):
            # 简单的 logistic 增长模型：0-100% 成功率
            # P(success) = 1 / (1 + exp(-k*(n - n50)))
            n50 = 8  # 50% 成功率的 UAV 数量
            k = 0.8  # 斜率
            
            prob = 1.0 / (1.0 + math.exp(-k * (n_uavs - n50)))
            mean_rate = min(1.0, max(0.0, prob))
            std_rate = 0.0
            u_a_rate = 0.0
            results_by_count[n_uavs] = [mean_rate, std_rate, u_a_rate, float(params.runs_per_count)]

            success_rate = mean_rate * 100.0
            filled_blocks = int(round(success_rate / 10.0))
            filled_blocks = max(0, min(10, filled_blocks))
            bar = "█" * filled_blocks + "░" * (10 - filled_blocks)
            print(f"  [{n_uavs:2d} UAVs] {bar} avg={success_rate:8.3f}%")
    
    else:
        # 真实运行：实际执行模拟（耗时）
        print("[真实运行] 开始批量模拟（这会很耗时）...\n")
        
        total = (max_uavs - min_uavs + 1) * params.runs_per_count
        if benchmark_runs > 0:
            est_per_run_s = benchmark_runtime_seconds(params, sample_runs=benchmark_runs)
            est_total_s = total * est_per_run_s
            print(
                f"[真实运行] 预计总时长: {est_total_s / 60:.1f} 分钟 "
                f"（{est_per_run_s:.2f}s/轮 × {total} 轮）\n"
            )
        else:
            print("[真实运行] 跳过基准测速（bench-runs=0）\n")
        completed = 0
        start_time = time.time()
        
        for n_uavs in range(min_uavs, max_uavs + 1):
            run_rates: list[float] = []
            
            print(f"  [{n_uavs:2d} UAVs] ", end="", flush=True)
            
            for run_idx in range(params.runs_per_count):
                completed += 1
                
                # 生成配置
                config = generate_uav_path_config(n_uavs)
                
                # 运行模拟
                result = run_simulation_with_config(
                    config,
                    timeout_s=600,
                    time_limit_h=params.time_limit_h,
                )
                
                if result is None:
                    print("✗", end="", flush=True)
                    continue

                run_rates.append(float(result["success_rate_10h"]))
                print("✓", end="", flush=True)
            
            if run_rates:
                mean_rate = float(np.mean(run_rates))
                if len(run_rates) > 1:
                    std_rate = float(np.std(run_rates, ddof=1))
                    u_a_rate = std_rate / math.sqrt(len(run_rates))
                else:
                    std_rate = 0.0
                    u_a_rate = 0.0
            else:
                mean_rate = 0.0
                std_rate = 0.0
                u_a_rate = 0.0

            results_by_count[n_uavs] = [mean_rate, std_rate, u_a_rate, float(len(run_rates))]
            success_rate = mean_rate * 100.0
            
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            remaining_time = avg_time * (total - completed)
            
            print(
                f"  avg={success_rate:.3f}% std={std_rate * 100.0:.3f}% uA={u_a_rate * 100.0:.3f}% "
                f"valid={len(run_rates)}/{params.runs_per_count} [ETA: {remaining_time/60:.0f}min]"
            )
    
    print("\n" + "=" * 80)
    print(f"✓ 数据收集完成")
    print("=" * 80)
    
    return results_by_count


def plot_figure_13(results: dict[int, list[float]], params: Figure13Params) -> Path:
    """
    绘制图13：成功概率 vs 无人机数量。
    """
    
    # 准备数据
    uav_counts = sorted(results.keys())
    success_rates = [results[n][0] * 100.0 for n in uav_counts]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=params.figsize, dpi=params.dpi)
    
    # 绘制曲线
    ax.plot(uav_counts, success_rates, "o-", linewidth=2.5, markersize=8, 
            label="Success Rate", color="#1f77b4")
    
    # 添加阈值线（0.90 和 0.95）
    ax.axhline(y=90, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="90% threshold")
    ax.axhline(y=95, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="95% threshold")
    
    # 标记达到 90% 和 95% 的点
    for threshold, color, label_suffix in [(0.90, "red", "90%"), (0.95, "green", "95%")]:
        for i, rate in enumerate(success_rates):
            if rate >= threshold * 100:
                ax.scatter([uav_counts[i]], [rate], s=150, color=color, marker="*", 
                          zorder=5, edgecolor="black", linewidth=1.5)
                ax.text(uav_counts[i], rate + 3, f"N={uav_counts[i]}", 
                       ha="center", fontsize=10, fontweight="bold", color=color)
                break
    
    # 配置坐标轴
    ax.set_xlabel("The Number of UAVs (N)", fontsize=14, fontweight="bold")
    time_limit_text = (
        f"{int(params.time_limit_h)}" if float(params.time_limit_h).is_integer() else f"{params.time_limit_h:.1f}"
    )
    ax.set_ylabel(f"Success Rate within {time_limit_text}h (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Discovery Probability vs. The Number of UAVs ({time_limit_text}-hour constraint)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    
    ax.set_xlim(min(uav_counts) - 0.5, max(uav_counts) + 0.5)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="lower right")
    
    # 设置刻度
    ax.set_xticks(uav_counts)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    
    plt.tight_layout()
    
    # 保存图形
    output_path = Path(__file__).parent / "13.png"
    fig.savefig(output_path, dpi=params.dpi, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


def save_data_csv(results: dict[int, list[float]]) -> Path:
    """保存成功概率数据到 CSV."""
    csv_path = Path(__file__).parent / "13_data.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["num_uavs", "avg_success_rate_%", "std_success_rate_%", "uA_success_rate_%", "valid_runs"])
        
        for n_uavs in sorted(results.keys()):
            mean_rate, std_rate, u_a_rate, valid_runs = results[n_uavs]
            writer.writerow([
                n_uavs,
                f"{mean_rate * 100.0:.3f}",
                f"{std_rate * 100.0:.3f}",
                f"{u_a_rate * 100.0:.3f}",
                int(valid_runs),
            ])
    
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 13 (success probability vs UAV count)")
    parser.add_argument("--real", action="store_true", help="Run real Monte Carlo simulations instead of demo data")
    parser.add_argument("--n-min", type=int, default=1, help="Minimum UAV count")
    parser.add_argument("--n-max", type=int, default=15, help="Maximum UAV count")
    parser.add_argument("--runs", type=int, default=3, help="Runs per UAV count")
    parser.add_argument("--bench-runs", type=int, default=3, help="Benchmark runs before real mode estimate (0 to skip)")
    parser.add_argument("--time-limit", type=float, default=10.0, help="Success deadline in hours")
    return parser.parse_args()


def main() -> None:
    """主函数：绘制图13。"""

    args = parse_args()
    if args.n_min <= 0 or args.n_max <= 0 or args.runs <= 0 or args.bench_runs < 0:
        raise ValueError("n-min, n-max, runs must be positive and bench-runs must be >= 0")
    if args.n_min > args.n_max:
        raise ValueError("n-min must be <= n-max")
    if args.time_limit <= 0:
        raise ValueError("time-limit must be > 0")

    params = Figure13Params(
        time_limit_h=float(args.time_limit),
        uav_count_range=(int(args.n_min), int(args.n_max)),
        runs_per_count=int(args.runs),
    )
    
    # 收集数据
    results = collect_data(params, demo_mode=not args.real, benchmark_runs=int(args.bench_runs))
    
    # 保存 CSV
    csv_path = save_data_csv(results)
    print(f"✓ 数据已保存: {csv_path}")
    
    # 绘制图形
    output_path = plot_figure_13(results, params)
    print(f"✓ 图13已保存: {output_path}")


if __name__ == "__main__":
    main()
