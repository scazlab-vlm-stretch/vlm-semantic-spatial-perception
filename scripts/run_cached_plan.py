"""
Utility script to replay a cached plan with PrimitiveExecutor.

Examples:
  # Dry run (no robot connection)
  uv run python scripts/run_cached_plan.py \
      --plan tests/artifacts/pick_plan_prepare_plan.json \
      --world tests/assets/continuous_pick_fixture \
      --dry-run

  # Execute on the robot (default IP constant below)
  uv run python scripts/run_cached_plan.py \
      --plan tests/artifacts/pick_plan_prepare_plan.json \
      --world tests/assets/continuous_pick_fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.primitives import PrimitiveExecutor, SkillPlan  # noqa: E402

DEFAULT_ROBOT_IP = "192.168.1.224"


def _load_world_state(world_dir: Path) -> Dict[str, Any]:
    registry_path = world_dir / "registry.json"
    state_path = world_dir / "state.json"
    index_path = world_dir / "perception_pool" / "index.json"

    if not (registry_path.exists() and state_path.exists() and index_path.exists()):
        raise FileNotFoundError(
            "World directory must contain registry.json, state.json, "
            "and perception_pool/index.json"
        )

    registry = json.loads(registry_path.read_text())
    state = json.loads(state_path.read_text())
    snapshot_index = json.loads(index_path.read_text())

    world_state = {
        "registry": registry,
        "last_snapshot_id": state.get("last_snapshot_id"),
        "snapshot_index": snapshot_index,
        "robot_state": state.get("robot_state"),
    }
    return world_state


def _is_plan_dict(candidate: Any) -> bool:
    return isinstance(candidate, dict) and isinstance(candidate.get("primitives"), list)


def _extract_plan_entries(plan_payload: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Normalize different plan JSON layouts into a list of candidate plan dicts.
    """

    if isinstance(plan_payload, dict):
        for key in ("translated_plan", "plan"):
            candidate = plan_payload.get(key)
            if _is_plan_dict(candidate):
                return [(key, candidate)]
        if _is_plan_dict(plan_payload):
            return [("root", plan_payload)]
        entries: List[Tuple[str, Dict[str, Any]]] = []
        for key, value in plan_payload.items():
            if _is_plan_dict(value):
                entries.append((str(key), value))
        if entries:
            return entries

    if isinstance(plan_payload, list):
        entries = []
        for idx, value in enumerate(plan_payload):
            if _is_plan_dict(value):
                entries.append((str(idx), value))
        if entries:
            return entries

    return []


def _filter_plan_entries(
    entries: Sequence[Tuple[str, Dict[str, Any]]], selector: Optional[str]
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Select which plan entries to run based on user input.
    """

    if not entries:
        return []

    if not selector or selector.lower() == "all":
        return list(entries)

    selector = selector.strip()
    matches: List[Tuple[str, Dict[str, Any]]] = [
        entry for entry in entries if entry[0] == selector or entry[1].get("action_name") == selector
    ]

    if not matches and selector.isdigit():
        idx = int(selector)
        if 0 <= idx < len(entries):
            matches = [entries[idx]]

    if not matches:
        available = ", ".join(entry[0] or entry[1].get("action_name", f"#{idx}") for idx, entry in enumerate(entries))
        raise ValueError(
            f"Plan selector '{selector}' not found. "
            f"Available options: {available or 'no plan entries detected'}"
        )

    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Path to cached plan JSON")
    parser.add_argument(
        "--world",
        required=True,
        help="Path to world-state directory (registry/state/perception_pool)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip robot connection and only translate/validate the plan.",
    )
    parser.add_argument(
        "--robot-ip",
        default=DEFAULT_ROBOT_IP,
        help=f"Robot IP (default: {DEFAULT_ROBOT_IP}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (defaults to outputs/cached_plan_execution_log/<timestamp>.json).",
    )
    parser.add_argument(
        "--plan-key",
        default=None,
        help=(
            "Optional key/index for plan JSON files that contain multiple skill plans. "
            "Use 'all' (default) to execute every plan sequentially."
        ),
    )
    args = parser.parse_args()

    plan_payload = json.loads(Path(args.plan).read_text())
    plan_entries = _extract_plan_entries(plan_payload)
    if not plan_entries:
        raise ValueError(
            "Unable to locate a skill plan inside the provided JSON. "
            "Ensure the file contains a 'plan' object or a list/dict of plan definitions."
        )
    selected_entries = _filter_plan_entries(plan_entries, args.plan_key)

    world_dir = Path(args.world)
    world_state = _load_world_state(world_dir)

    execute = not args.dry_run
    executor = PrimitiveExecutor(
        primitives=_build_primitives_interface(execute, args.robot_ip),
        perception_pool_dir=world_dir / "perception_pool",
    )
    cumulative_results: List[Dict[str, Any]] = []
    for idx, (entry_key, entry_plan_dict) in enumerate(selected_entries):
        plan = SkillPlan.from_dict(entry_plan_dict)
        label = entry_key or plan.action_name or f"plan_{idx}"
        print(f"--> Executing plan '{label}' with {len(plan.primitives)} primitives")
        result_payload = executor.execute_plan(plan, world_state, dry_run=not execute)
        run_result: Dict[str, Any] = {
            "plan_key": entry_key,
            "action_name": plan.action_name,
            "plan": plan.to_dict(),
            "executed": result_payload.executed,
        }
        if result_payload.primitive_results:
            run_result["primitive_results"] = result_payload.primitive_results
        cumulative_results.append(run_result)

    if len(cumulative_results) == 1:
        output_payload: Dict[str, Any] | List[Dict[str, Any]] = cumulative_results[0]
    else:
        output_payload = {"plan_results": cumulative_results}

    output_json = json.dumps(output_payload, indent=2)
    
    # Determine output path: use provided path or generate timestamped default
    if args.output:
        output_path = Path(args.output)
    else:
        # Generate timestamped filename in default directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "outputs" / "cached_plan_execution_log"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{timestamp}.json"
    
    # Write to file
    output_path.write_text(output_json)
    print(f"Saved executor output to {output_path}")


def _build_primitives_interface(execute: bool, robot_ip: Optional[str]):
    if not execute:
        return None
    from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface

    print(
        f"⚠️  EXECUTION ENABLED: connecting to {robot_ip or DEFAULT_ROBOT_IP} "
        "via XArmPybulletInterface."
    )
    return XArmPybulletInterface()


if __name__ == "__main__":
    main()
