"""
测试脚本：验证新的 UAV 路径规划接口。

演示：
1. 默认条带扫描模式
2. 自定义路径加载
3. 编程式路径规格构造
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(__file__))

from config import build_default_config
from core.uav_controller import (
    UAVFleetBuilder,
    UAVPathSpec,
    ArcTurnSpec,
)


def test_strip_scan_mode():
    """测试默认条带扫描模式。"""
    print("=" * 60)
    print("Test 1: Strip Scan Mode (Default)")
    print("=" * 60)

    cfg = build_default_config(os.path.dirname(__file__))
    fleet = UAVFleetBuilder.from_default_config(cfg)

    print(f"Fleet initialized with {len(fleet.controllers)} UAVs")
    for i, uav in enumerate(fleet.controllers):
        print(f"  UAV #{uav.uav_id}:")
        print(f"    - Current position (u): {uav.position_u}")
        print(f"    - Start time (h): {uav.start_time_h}")
        print(f"    - Is started: {uav.is_started}")
        print(f"    - Auto gen type: {uav.auto_gen_type}")
        print(f"    - Segments: {len(uav.segments)}")

    print()


def test_custom_paths_json():
    """测试从 JSON 加载自定义路径。"""
    print("=" * 60)
    print("Test 2: Custom Paths from JSON")
    print("=" * 60)

    cfg = build_default_config(os.path.dirname(__file__))
    json_file = os.path.join(os.path.dirname(__file__), "config_templates", "multi_uav_paths.json")

    try:
        fleet = UAVFleetBuilder.from_custom_json(cfg, json_file)
        print(f"Fleet loaded from JSON: {len(fleet.controllers)} UAVs")
        for uav in fleet.controllers:
            print(f"  UAV #{uav.uav_id}:")
            print(f"    - Current position (u): {uav.position_u}")
            print(f"    - Start time (h): {uav.start_time_h}")
            print(f"    - Is started: {uav.is_started}")
            print(f"    - Segments: {len(uav.segments)}")
            for j, seg in enumerate(uav.segments):
                print(f"      Segment {j}:")
                print(f"        - End point: {seg['end_point_u']}")
                if seg["arc_turn"] is not None:
                    arc = seg["arc_turn"]
                    print(f"        - Arc turn: radius={arc.radius_u}, clockwise={arc.is_clockwise}")
                else:
                    print(f"        - No arc turn")
    except FileNotFoundError:
        print(f"JSON file not found: {json_file}")

    print()


def test_programmatic_construction():
    """测试编程式路径规格构造。"""
    print("=" * 60)
    print("Test 3: Programmatic Path Specification")
    print("=" * 60)

    cfg = build_default_config(os.path.dirname(__file__))

    # 构造两个 UAV 的路径规格
    path_specs = [
        UAVPathSpec(
            uav_id=0,
            start_time_h=0.0,
            start_pos_u=(30000, 0),
            segments=[
                {
                    "end_point_u": (100000, 100000),
                    "arc_turn": ArcTurnSpec(
                        radius_u=15000,
                        start_point_u=(100000, 100000),
                        end_point_u=(150000, 100000),
                        center_u=None,
                        is_clockwise=True,
                    ),
                },
                {
                    "end_point_u": (200000, 200000),
                    "arc_turn": None,
                },
            ],
            auto_gen_type="test",
        ),
        UAVPathSpec(
            uav_id=1,
            start_time_h=1.5,
            start_pos_u=(60000, 0),
            segments=[
                {
                    "end_point_u": (120000, 120000),
                    "arc_turn": None,
                },
            ],
            auto_gen_type="test",
        ),
    ]

    fleet = UAVFleetBuilder.from_path_specs(cfg, path_specs)
    print(f"Fleet constructed programmatically: {len(fleet.controllers)} UAVs")
    for uav in fleet.controllers:
        print(f"  UAV #{uav.uav_id}:")
        print(f"    - Current position (u): {uav.position_u}")
        print(f"    - Start time (h): {uav.start_time_h}")
        print(f"    - Is started: {uav.is_started}")
        print(f"    - Segments: {len(uav.segments)}")

    print()


if __name__ == "__main__":
    test_strip_scan_mode()
    test_custom_paths_json()
    test_programmatic_construction()
    print("=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)
