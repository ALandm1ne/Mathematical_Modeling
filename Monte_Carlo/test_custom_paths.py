"""测试脚本：验证外置路径驱动、新圆弧 schema 与 API 兼容行为。"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import build_default_config
from core.uav_controller import ArcTurnSpec, UAVFleetBuilder, UAVPathGenerator, UAVPathSpec
from main import _resolve_external_path_source


def _build_cfg():
    return build_default_config(os.path.dirname(__file__), require_cuda_override=False)


def test_removed_legacy_apis():
    print("=" * 60)
    print("Test 1: Legacy APIs Removed")
    print("=" * 60)

    cfg = _build_cfg()

    try:
        UAVFleetBuilder.from_default_config(cfg)
        raise AssertionError("from_default_config should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"  from_default_config blocked as expected: {e}")

    try:
        UAVFleetBuilder.from_strip_scan(cfg)
        raise AssertionError("from_strip_scan should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"  from_strip_scan blocked as expected: {e}")

    print()


def test_custom_paths_json():
    print("=" * 60)
    print("Test 2: Custom Paths from JSON")
    print("=" * 60)

    cfg = _build_cfg()
    json_file = os.path.join(os.path.dirname(__file__), "config_templates", "uav_paths.json")
    fleet = UAVFleetBuilder.from_custom_json(cfg, json_file)

    print(f"Fleet loaded from JSON: {len(fleet.controllers)} UAVs")
    for uav in fleet.controllers:
        print(f"  UAV #{uav.uav_id}:")
        print(f"    - Current position (u): {uav.position_u}")
        print(f"    - Start time (h): {uav.start_time_h}")
        print(f"    - Is started: {uav.is_started}")
        print(f"    - Segments: {len(uav.segments)}")
        assert all(seg.get("segment_type") in ("line", "arc") for seg in uav.segments)

    print()


def test_programmatic_construction():
    print("=" * 60)
    print("Test 3: Programmatic Path Specification")
    print("=" * 60)

    cfg = _build_cfg()

    path_specs = [
        UAVPathSpec(
            uav_id=0,
            start_time_h=0.0,
            start_pos_u=(30000, 0),
            segments=[
                {
                    "segment_type": "line",
                    "end_point_u": (100000, 100000),
                },
                {
                    "segment_type": "arc",
                    "arc": ArcTurnSpec(
                        start_point_u=(100000, 100000),
                        radius_u=15000,
                        is_clockwise=True,
                        rotation_angle_deg=90,
                    ),
                },
                {
                    "segment_type": "line",
                    "end_point_u": (200000, 200000),
                },
            ],
            auto_gen_type="test",
        )
    ]

    fleet = UAVFleetBuilder.from_path_specs(cfg, path_specs)
    print(f"Fleet constructed programmatically: {len(fleet.controllers)} UAVs")
    for uav in fleet.controllers:
        print(f"  UAV #{uav.uav_id}: pos={uav.position_u}, segments={len(uav.segments)}")

    print()


def test_strict_validation_on_invalid_specs():
    print("=" * 60)
    print("Test 4: Strict Validation")
    print("=" * 60)

    cfg = _build_cfg()
    bad_specs = [
        UAVPathSpec(
            uav_id=0,
            start_time_h=0.0,
            start_pos_u=(0, 0),
            segments=[
                {
                    "segment_type": "arc",
                    "arc": ArcTurnSpec(
                        start_point_u=(0.0, 0.0),
                        radius_u=-1.0,
                        is_clockwise=True,
                        rotation_angle_deg=30,
                    ),
                }
            ],
        )
    ]

    try:
        UAVFleetBuilder.from_path_specs(cfg, bad_specs)
        raise AssertionError("strict validation should fail on negative radius")
    except ValueError as e:
        print(f"  strict validation blocked invalid spec: {e}")

    bad_angle_specs = [
        UAVPathSpec(
            uav_id=1,
            start_time_h=0.0,
            start_pos_u=(0, 0),
            segments=[
                {
                    "segment_type": "arc",
                    "arc": ArcTurnSpec(
                        start_point_u=(0.0, 0.0),
                        radius_u=10.0,
                        is_clockwise=True,
                        rotation_angle_deg=-10,
                    ),
                }
            ],
        )
    ]

    try:
        UAVFleetBuilder.from_path_specs(cfg, bad_angle_specs)
        raise AssertionError("strict validation should fail on negative angle")
    except ValueError as e:
        print(f"  strict validation blocked invalid angle: {e}")

    print()


def test_main_path_source_policy():
    print("=" * 60)
    print("Test 5: Main Path Source Policy")
    print("=" * 60)

    script_dir = os.path.dirname(__file__)

    cfg_conflict = _build_cfg()
    cfg_conflict.uav_fleet_mode.custom_paths_json = "config_templates/uav_paths.json"
    cfg_conflict.runtime.api_demo_enable = True
    cfg_conflict.runtime.api_demo_json_path = "config_templates/uav_paths.json"
    resolved = _resolve_external_path_source(cfg_conflict, script_dir)
    print(f"  conflict policy resolved path: {resolved}")
    expected = os.path.abspath(os.path.join(script_dir, "config_templates/uav_paths.json"))
    assert resolved == expected

    cfg_missing = _build_cfg()
    cfg_missing.uav_fleet_mode.custom_paths_json = None
    cfg_missing.runtime.api_demo_enable = False
    resolved = _resolve_external_path_source(cfg_missing, script_dir)
    print(f"  missing policy resolved path: {resolved}")
    assert resolved is None

    print()


def test_continuity_constraints():
    print("=" * 60)
    print("Test 6: Continuity Constraints")
    print("=" * 60)

    cfg = _build_cfg()
    discontinuous_specs = [
        UAVPathSpec(
            uav_id=0,
            start_time_h=0.0,
            start_pos_u=(0, 0),
            segments=[
                {"segment_type": "line", "end_point_u": (1000, 0)},
                {
                    "segment_type": "arc",
                    "arc": ArcTurnSpec(
                        start_point_u=(1200, 0),
                        radius_u=100,
                        is_clockwise=True,
                        rotation_angle_deg=90,
                    ),
                },
            ],
        )
    ]
    try:
        UAVFleetBuilder.from_path_specs(cfg, discontinuous_specs)
        raise AssertionError("continuity mismatch should fail")
    except ValueError as e:
        print(f"  continuity check blocked invalid chain: {e}")

    print()


def test_angle_and_zero_deg_behavior():
    print("=" * 60)
    print("Test 7: Angle Mapping and 0-Degree Arc")
    print("=" * 60)

    deg_to_expected = {
        15: math.pi / 12,
        30: math.pi / 6,
        45: math.pi / 4,
        60: math.pi / 3,
        75: 5 * math.pi / 12,
        90: math.pi / 2,
        180: math.pi,
        270: 3 * math.pi / 2,
        360: 2 * math.pi,
    }
    for deg, expected in deg_to_expected.items():
        rad = UAVPathGenerator._deg_to_rad_stable(float(deg))
        assert abs(rad - expected) < 1e-12, f"angle mapping mismatch: deg={deg}"
    print("  common-angle mapping passed")

    cfg = _build_cfg()
    specs = [
        UAVPathSpec(
            uav_id=9,
            start_time_h=0.0,
            start_pos_u=(1000, 0),
            segments=[
                {
                    "segment_type": "arc",
                    "arc": ArcTurnSpec(
                        start_point_u=(1000.0, 0.0),
                        radius_u=500.0,
                        is_clockwise=True,
                        rotation_angle_deg=0.0,
                    ),
                },
                {
                    "segment_type": "line",
                    "end_point_u": (2000, 0),
                },
            ],
        )
    ]
    fleet = UAVFleetBuilder.from_path_specs(cfg, specs)
    uav = fleet.controllers[0]
    uav.update(0.0)
    assert uav.current_segment_idx >= 1, "0-degree arc should finish immediately"
    print("  zero-degree arc passed")

    print()


def test_legacy_schema_rejection():
    print("=" * 60)
    print("Test 8: Legacy Schema Rejection")
    print("=" * 60)

    old_schema_data = [
        {
            "uav_id": 0,
            "start_time_h": 0,
            "start_pos_u": [0, 0],
            "segments": [
                {
                    "end_point_u": [1000, 1000],
                    "arc_turn": {
                        "start_point_u": [1000, 1000],
                        "end_point_u": [1200, 1000],
                        "radius_u": 100,
                        "is_clockwise": True
                    }
                }
            ]
        }
    ]

    tmp = os.path.join(os.path.dirname(__file__), "_tmp_legacy_paths.json")
    with open(tmp, "w", encoding="utf-8") as f:
        import json

        json.dump(old_schema_data, f)

    try:
        try:
            UAVPathGenerator.load_custom_paths_from_json(tmp)
            raise AssertionError("legacy schema should be rejected")
        except ValueError as e:
            print(f"  legacy schema rejected as expected: {e}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    print()


if __name__ == "__main__":
    test_removed_legacy_apis()
    test_custom_paths_json()
    test_programmatic_construction()
    test_strict_validation_on_invalid_specs()
    test_main_path_source_policy()
    test_continuity_constraints()
    test_angle_and_zero_deg_behavior()
    test_legacy_schema_rejection()

    print("=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)
