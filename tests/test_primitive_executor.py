import json
from pathlib import Path

import numpy as np

from src.primitives import PrimitiveExecutor
from src.primitives.skill_plan_types import PrimitiveCall, SkillPlan


def _build_snapshot(tmp_path: Path) -> dict:
    pool_dir = tmp_path
    snap_id = "snap-test"
    snap_dir = pool_dir / "snapshots" / snap_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    depth = np.ones((2, 2), dtype=np.float32)
    depth_path = snap_dir / "depth.npz"
    np.savez_compressed(depth_path, depth_m=depth)

    intr_path = snap_dir / "intrinsics.json"
    intr_data = {
        "fx": 1.0,
        "fy": 1.0,
        "ppx": 0.0,
        "ppy": 0.0,
        "width": 2,
        "height": 2,
    }
    intr_path.write_text(json.dumps(intr_data))

    index = {
        "snapshots": {
            snap_id: {
                "files": {
                    "depth_npz": f"snapshots/{snap_id}/depth.npz",
                    "intrinsics": f"snapshots/{snap_id}/intrinsics.json",
                }
            }
        }
    }

    return {"pool_dir": pool_dir, "snapshot_id": snap_id, "snapshot_index": index}


def test_prepare_plan_translates_pixel_targets(tmp_path):
    snapshot = _build_snapshot(tmp_path)
    plan = SkillPlan(
        action_name="pick",
        primitives=[
            PrimitiveCall(
                name="move_to_pose",
                parameters={
                    "target_pixel_yx": [500, 500],
                    "depth_offset_m": 0.05,
                },
            )
        ],
    )

    executor = PrimitiveExecutor(primitives=None, perception_pool_dir=snapshot["pool_dir"])
    translated_plan = executor.prepare_plan(
        plan,
        {
            "last_snapshot_id": snapshot["snapshot_id"],
            "snapshot_index": snapshot["snapshot_index"],
        },
    )

    params = translated_plan.primitives[0].parameters
    assert "target_pixel_yx" not in params
    assert params["target_position"] == [1.0, 1.0, 1.05]


def test_execute_plan_serializes_dict_results(tmp_path):
    snapshot = _build_snapshot(tmp_path)
    plan = SkillPlan(
        action_name="pick",
        primitives=[
            PrimitiveCall(
                name="move_to_pose",
                parameters={"target_position": [0.0, 0.0, 0.1]},
            )
        ],
    )

    class FakePrimitives:
        def move_to_pose(self, target_position, **kwargs):
            del target_position, kwargs  # unused in fake planner
            js = {"position": [0.0] * 7, "velocity": None}
            return True, js, 0.1

    executor = PrimitiveExecutor(primitives=FakePrimitives(), perception_pool_dir=snapshot["pool_dir"])
    result = executor.execute_plan(
        plan,
        {
            "last_snapshot_id": snapshot["snapshot_id"],
            "snapshot_index": snapshot["snapshot_index"],
        },
    )

    assert result.executed is True
    serialized = json.dumps(result.primitive_results)
    payload = json.loads(serialized)
    assert payload[0][0] is True
    assert "position" in payload[0][1]
