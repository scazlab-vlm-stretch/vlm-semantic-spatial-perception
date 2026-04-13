"""
Stretch Primitives Test
=======================

Tests the Stretch motion primitives directly — no LLM, no perception pipeline.
Exact pick/place positions are hard-coded from the known scene geometry.

Scene (matches run_stretch_sim_demo.py):
  Brown table at (-0.4, -1.1), top surface z = 0.60 m
  Red table   at ( 0.6, -1.1), top surface z = 0.60 m
  Blue block  at (-0.4, -1.1, 0.63) — centred on the brown table

Sequence:
  1. Open gripper
  2. Navigate base toward brown table (block pickup side)
  3. Move gripper to block position
  4. Close gripper (attach block)
  5. Retract arm to home
  6. Navigate base toward red table (place side)
  7. Move gripper to place position above red table
  8. Open gripper (release block)
  9. Retract arm to home

Usage:
    uv run python examples/test_stretch_primitives.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import math

# ---------------------------------------------------------------------------
# Scene geometry (kept in sync with run_stretch_sim_demo.py)
# ---------------------------------------------------------------------------
_TABLE_Z_HALF = 0.30           # half-height → top surface at 0.60 m
_TABLE_Z      = _TABLE_Z_HALF  # table centre z
_BLOCK_Z      = _TABLE_Z_HALF * 2 + 0.03   # block centre resting on table top

BROWN_TABLE_XY = [-0.4, -1.1]
RED_TABLE_XY   = [ 0.6, -1.1]

SCENE_OBJECTS = [
    {
        "object_id":   "brown_table_1",
        "object_type": "surface",
        "affordances": ["support_surface"],
        "position_3d": [BROWN_TABLE_XY[0], BROWN_TABLE_XY[1], _TABLE_Z],
    },
    {
        "object_id":   "red_table_1",
        "object_type": "surface",
        "affordances": ["support_surface"],
        "position_3d": [RED_TABLE_XY[0], RED_TABLE_XY[1], _TABLE_Z],
    },
    {
        "object_id":   "blue_block_1",
        "object_type": "block",
        "affordances": ["graspable"],
        "position_3d": [BROWN_TABLE_XY[0], BROWN_TABLE_XY[1], _BLOCK_Z],
    },
]

OBJECT_COLORS: Dict[str, List[float]] = {
    "brown_table_1": [0.55, 0.35, 0.15, 1.0],
    "red_table_1":   [0.80, 0.10, 0.10, 1.0],
    "blue_block_1":  [0.15, 0.35, 0.85, 1.0],
}

OBJECT_HALF_EXTENTS: Dict[str, List[float]] = {
    "brown_table_1": [0.30, 0.30, _TABLE_Z_HALF],
    "red_table_1":   [0.30, 0.30, _TABLE_Z_HALF],
    "blue_block_1":  [0.03, 0.03, 0.03],
}

STRETCH_CAMERA_AIM_JOINTS = [
    0.0,              # right_wheel
    0.0,              # left_wheel
    0.65,             # lift
    0.0,              # arm_l3
    0.0,              # arm_l2
    0.0,              # arm_l1
    0.0,              # arm_l0
    0.0,              # wrist_yaw
    0.0,              # gripper_finger_left
    0.0,              # gripper_finger_right
    -math.pi / 2,     # head_pan — look along -Y toward tables
    -0.5,             # head_tilt — look slightly down
]

PAUSE = 1.0   # seconds between steps so it's easy to watch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(msg: str)   -> None: print(f"  ✓  {msg}")
def _step(msg: str) -> None: print(f"\n── {msg}")
def _fail(msg: str) -> None: print(f"  ✗  {msg}")


def _check(result: dict, label: str) -> bool:
    ok = result.get("success", False)
    if ok:
        _ok(f"{label} — {result}")
    else:
        _fail(f"{label} — {result}")
    return ok


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_test() -> int:
    # ------------------------------------------------------------------ #
    # 1. Build environment                                                 #
    # ------------------------------------------------------------------ #
    _step("Starting PyBullet Stretch environment")

    from src.kinematics.sim.scene_environment import SceneEnvironment
    from src.kinematics.stretch_pybullet_interface import _STRETCH_URDF
    from src.kinematics.sim.stretch_pybullet_primitives import StretchPyBulletPrimitives
    from src.perception.object_registry import DetectedObjectRegistry

    env = SceneEnvironment(
        urdf_path=_STRETCH_URDF,
        camera_link="camera_color_optical_frame",
        initial_joints=STRETCH_CAMERA_AIM_JOINTS,
        n_arm_joints=12,
        tcp_link_name="link_grasp_center",
        camera_use_world_up=True,
        viewer_target=[0.1, -0.8, 0.4],
        viewer_distance=2.8,
        viewer_yaw=-30,
        viewer_pitch=-25,
    )
    env.start()

    # Patch colour/size dicts so table geometry is correct
    import src.kinematics.sim.scene_environment as _se_mod
    _orig_colors  = _se_mod.OBJECT_COLORS
    _orig_extents = _se_mod.OBJECT_HALF_EXTENTS
    _se_mod.OBJECT_COLORS       = {**_orig_colors,  **OBJECT_COLORS}
    _se_mod.OBJECT_HALF_EXTENTS = {**_orig_extents, **OBJECT_HALF_EXTENTS}
    env.add_scene_objects(SCENE_OBJECTS)
    _se_mod.OBJECT_COLORS       = _orig_colors
    _se_mod.OBJECT_HALF_EXTENTS = _orig_extents

    env.set_status("Primitives test — initialising")
    env.step(1.0)
    _ok("Environment ready")

    # Query the sim for the actual block position rather than using the
    # hard-coded spawn value — this stays correct if spawn logic changes.
    block_pick_pos = env.get_object_position("blue_block_1")
    assert block_pick_pos is not None, "blue_block_1 not found in sim — check scene setup"
    _ok(f"blue_block_1 sim position: {[round(v, 3) for v in block_pick_pos]}")

    # Place target: same XY as red table, same Z as the block
    red_table_pos  = env.get_object_position("red_table_1")
    assert red_table_pos is not None, "red_table_1 not found in sim"
    block_place_pos = [red_table_pos[0], red_table_pos[1], block_pick_pos[2]]
    _ok(f"place target: {[round(v, 3) for v in block_place_pos]}")

    # Empty registry — primitives use explicit target_position so registry
    # is only needed for point_label resolution (unused in this test).
    registry = DetectedObjectRegistry()
    prims = StretchPyBulletPrimitives(env=env, registry=registry)

    failures = []

    # ------------------------------------------------------------------ #
    # 2. Open gripper                                                      #
    # ------------------------------------------------------------------ #
    _step("Step 1 — open gripper")
    env.set_status("Step 1: open gripper")
    r = prims.open_gripper()
    if not _check(r, "open_gripper"):
        failures.append("open_gripper")
    env.step(PAUSE)

    # ------------------------------------------------------------------ #
    # 3. Navigate roughly toward brown table (coarse approach)             #
    #    move_gripper_to_pose will fine-align the base before extending.   #
    # ------------------------------------------------------------------ #
    _step("Step 2 — navigate toward brown table")
    env.set_status("Step 2: navigate → brown table area")
    r = prims.navigate_to(
        target_position=block_pick_pos,
        target_yaw=0.0,
        standoff_distance=0.9,   # coarse — gripper primitive fine-aligns
    )
    if not _check(r, "navigate_to(brown_table)"):
        failures.append("navigate_to(brown_table)")
    env.step(PAUSE)

    # ------------------------------------------------------------------ #
    # 4. Move gripper to block (grasp approach)                           #
    # ------------------------------------------------------------------ #
    _step("Step 3 — move gripper to block")
    env.set_status("Step 3: move gripper → blue block")
    r = prims.move_gripper_to_pose(
        target_position=block_pick_pos,
        preset_orientation="top_down",
        is_place=False,
    )
    if not _check(r, "move_gripper_to_pose(pick)"):
        failures.append("move_gripper_to_pose(pick)")
    env.step(PAUSE)

    # ------------------------------------------------------------------ #
    # 5. Close gripper (attach block)                                      #
    # ------------------------------------------------------------------ #
    _step("Step 4 — close gripper")
    env.set_status("Step 4: close gripper")
    r = prims.close_gripper()
    if not _check(r, "close_gripper"):
        failures.append("close_gripper")
    print(f"       held object: {prims._held_object}")
    env.step(PAUSE)

    # ------------------------------------------------------------------ #
    # 6. Retract arm                                                       #
    # ------------------------------------------------------------------ #
    _step("Step 5 — retract arm")
    env.set_status("Step 5: retract arm")
    r = prims.retract_gripper()
    if not _check(r, "retract_gripper"):
        failures.append("retract_gripper")
    env.step(PAUSE)

    # ------------------------------------------------------------------ #
    # 7. Navigate roughly toward red table (coarse approach)               #
    # ------------------------------------------------------------------ #
    _step("Step 6 — navigate toward red table")
    env.set_status("Step 6: navigate → red table area")
    r = prims.navigate_to(
        target_position=block_place_pos,
        target_yaw=0.0,
        standoff_distance=0.9,
    )
    if not _check(r, "navigate_to(red_table)"):
        failures.append("navigate_to(red_table)")
    env.step(PAUSE)

    # ------------------------------------------------------------------ #
    # 8. Move gripper to place position                                    #
    # ------------------------------------------------------------------ #
    _step("Step 7 — move gripper to place position")
    env.set_status("Step 7: move gripper → place")
    r = prims.move_gripper_to_pose(
        target_position=block_place_pos,
        preset_orientation="top_down",
        is_place=True,
    )
    if not _check(r, "move_gripper_to_pose(place)"):
        failures.append("move_gripper_to_pose(place)")
    env.step(PAUSE)

    # ------------------------------------------------------------------ #
    # 9. Open gripper (release block)                                      #
    # ------------------------------------------------------------------ #
    _step("Step 8 — open gripper (release)")
    env.set_status("Step 8: open gripper")
    r = prims.open_gripper()
    if not _check(r, "open_gripper(release)"):
        failures.append("open_gripper(release)")
    env.step(PAUSE)

    # ------------------------------------------------------------------ #
    # 10. Retract arm to home                                              #
    # ------------------------------------------------------------------ #
    _step("Step 9 — retract arm to home")
    env.set_status("Step 9: retract to home")
    r = prims.retract_gripper()
    if not _check(r, "retract_gripper(home)"):
        failures.append("retract_gripper(home)")
    env.step(PAUSE)

    # ------------------------------------------------------------------ #
    # Result                                                               #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    if failures:
        print(f"FAILED — {len(failures)} primitive(s) failed: {failures}")
        env.set_status(f"✗ FAILED: {failures}", color=[1.0, 0.2, 0.2])
    else:
        print("PASSED — all primitives succeeded")
        env.set_status("✓ All primitives passed", color=[0.2, 0.9, 0.2])
    print(f"{'='*60}")

    env.step(3.0)
    env.stop()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run_test())
