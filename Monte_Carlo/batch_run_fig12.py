#!/usr/bin/env python3
"""
批量运行蒙特卡洛模拟，为图 12 生成多策略样本。
流程：
  1. 修改 config.py 中的策略参数
  2. 运行 main.py 
  3. 记录生成的结果目录
  4. 更新策略映射文件 pictures/12_strategy_map.csv
"""

import os
import sys
import subprocess
import time
import csv
from pathlib import Path
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, "config.py")

def make_strategy_config_edit(dynamic_replan_enable: bool, target_speed: float) -> None:
    """临时修改 config.py 中的策略参数。"""
    with open(config_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到并修改 DynamicReplanningConfig 中的 enable 字段
    for i, line in enumerate(lines):
        if line.strip().startswith('enable: bool =') and i > 100:  # 避免修改注释
            lines[i] = f"    enable: bool = {str(dynamic_replan_enable)}\n"
        elif line.strip().startswith('target_speed_km_h: float ='):
            lines[i] = f"    target_speed_km_h: float = {target_speed}\n"
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def restore_strategy_config() -> None:
    """恢复 config.py 到默认状态。"""
    with open(config_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.strip().startswith('enable: bool =') and i > 100:
            lines[i] = "    enable: bool = True\n"
        elif line.strip().startswith('target_speed_km_h: float ='):
            lines[i] = "    target_speed_km_h: float = 30.0\n"
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def get_latest_result_dir() -> str | None:
    """获取 results/ 中最新的运行目录。"""
    results_dir = os.path.join(script_dir, "results")
    if not os.path.exists(results_dir):
        return None
    
    subdirs = sorted(
        [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))],
        reverse=True
    )
    
    return subdirs[0] if subdirs else None

def run_simulation() -> str | None:
    """运行一次完整仿真，返回结果目录名。"""
    main_py = os.path.join(script_dir, "main.py")
    venv_python = os.path.join(script_dir, ".venv", "bin", "python")
    
    try:
        result = subprocess.run(
            [venv_python, main_py],
            cwd=script_dir,
            capture_output=True,
            timeout=600,  # 10 分钟超时
            text=True
        )
        
        if result.returncode == 0:
            return get_latest_result_dir()
        else:
            print(f"[ERROR] 仿真失败: {result.stderr[:200]}")
            return None
    
    except subprocess.TimeoutExpired:
        print("[ERROR] 仿真超时（>10min）")
        return None
    except Exception as e:
        print(f"[ERROR] 仿真异常: {e}")
        return None

def main() -> None:
    """主批量运行循环。"""
    
    # 定义三种策略
    strategies = [
        {
            "name": "strip_markov_dynamic",
            "print_name": "Strip+Markov with Dynamic Replanning",
            "dynamic_replan": True,
            "target_speed": 30.0,
            "target_runs": 3  # 演示用 3 次
        },
        {
            "name": "strip_markov",
            "print_name": "Strip+Markov Baseline (no dynamic)",
            "dynamic_replan": False,
            "target_speed": 30.0,
            "target_runs": 3
        },
        {
            "name": "static_strip",
            "print_name": "Static Strip (target speed = 0)",
            "dynamic_replan": False,
            "target_speed": 0.0,
            "target_runs": 3
        }
    ]
    
    run_mapping = [["run_dir", "strategy"]]
    total_runs = sum(s["target_runs"] for s in strategies)
    completed_runs = 0
    start_time = time.time()
    
    print("=" * 80)
    print("  批量蒙特卡洛模拟 - 图 12 数据生成")
    print("=" * 80)
    print(f"总计: {total_runs} 次运行 (3 种策略)")
    est_per_run_s = 12
    est_total_s = total_runs * est_per_run_s
    print(f"预计时间: ~{est_total_s / 60:.1f} 分钟（按 {est_per_run_s}s/轮估算）\n")
    
    try:
        for strategy in strategies:
            strategy_name = strategy["name"]
            print(f"\n[策略] {strategy['print_name']}")
            print(f"  配置: dynamic_replan={strategy['dynamic_replan']}, target_speed={strategy['target_speed']} km/h")
            print(f"  计划运行: {strategy['target_runs']} 次\n")
            
            for run_idx in range(strategy["target_runs"]):
                run_num = run_idx + 1
                elapsed = time.time() - start_time
                
                print(f"  [{completed_runs + 1:2d}/{total_runs}] 运行 {run_num}/{strategy['target_runs']} ... ", end="", flush=True)
                
                # 修改配置
                make_strategy_config_edit(strategy["dynamic_replan"], strategy["target_speed"])
                
                # 运行仿真
                result_dir = run_simulation()
                
                if result_dir:
                    run_mapping.append([result_dir, strategy_name])
                    elapsed_sim = time.time() - start_time - elapsed
                    print(f"✓ ({elapsed_sim:.1f}s) {result_dir}")
                else:
                    print("✗ 失败")
                
                completed_runs += 1
        
        # 恢复默认配置
        restore_strategy_config()
        
        # 保存策略映射
        mapping_file = os.path.join(script_dir, "pictures", "12_strategy_map.csv")
        with open(mapping_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(run_mapping)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"✓ 批量运行完成！")
        print(f"  成功: {len(run_mapping) - 1} / {total_runs}")
        print(f"  总耗时: {total_time / 60:.1f} 分钟")
        print(f"  策略映射已保存到: {mapping_file}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n[用户中断]")
        restore_strategy_config()
        sys.exit(1)
    except Exception as e:
        print(f"\n[异常] {e}")
        restore_strategy_config()
        sys.exit(1)

if __name__ == "__main__":
    main()
