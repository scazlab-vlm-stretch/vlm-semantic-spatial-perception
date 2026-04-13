"""Replay saved worlds through the full LLM-driven planning pipeline.

This script always runs:
1) task analysis (LLM),
2) continuous detection replay (LLM-grounded perception),
3) planning using the same readiness-gated flow as the orchestrator demo.

Single-world mode:
  uv run scripts/benchmark_saved_world.py --world-dir outputs/captured_worlds/blocks

Batch mode over outputs/captured_worlds/*:
  uv run scripts/benchmark_saved_world.py --captured-root outputs/captured_worlds --run-all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.orchestrator_config import OrchestratorConfig
from src.camera.base_camera import BaseCamera, CameraIntrinsics
from src.planning.task_orchestrator import TaskOrchestrator


@dataclass
class CachedFrame:
    color: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics


class SnapshotCamera(BaseCamera):
    """Camera stub that replays cached frames in sequence."""

    def __init__(self, world_dir: Path):
        # Load all saved snapshots directly from the world bundle.
        self._frames = self._load_cached_frames(world_dir)
        self._cursor = 0
        self._current_intrinsics = self._frames[0].intrinsics

    @property
    def num_frames(self) -> int:
        """Return number of replay frames."""
        return len(self._frames)

    def _load_cached_frames(self, world_dir: Path) -> List[CachedFrame]:
        """Load cached perception snapshots as replayable frames.

        Args:
            world_dir: Saved world directory.

        Returns:
            Ordered list of cached frames.
        """
        # Snapshot metadata comes from the captured perception pool index.
        index_path = world_dir / "perception_pool" / "index.json"
        index_payload = json.loads(index_path.read_text())
        snapshots = index_payload["snapshots"]

        def _ts(entry: dict) -> float:
            # Keep replay deterministic by sorting frames by capture time.
            ts = entry.get("recorded_at") or entry.get("captured_at")
            if ts.endswith("Z"):  # Normalize common UTC suffix for fromisoformat.
                ts = ts.replace("Z", "+00:00")
            return datetime.fromisoformat(ts).timestamp()

        ordered = sorted(snapshots.items(), key=lambda item: _ts(item[1]))
        pool_dir = world_dir / "perception_pool"
        frames: List[CachedFrame] = []

        for snapshot_id, meta in ordered:
            # Each snapshot stores the color image, depth array, and camera intrinsics.
            files = meta.get("files") or {}
            color_path = pool_dir / files.get("color", "")
            depth_path = pool_dir / files.get("depth_npz", "")
            intr_path = pool_dir / files.get("intrinsics", "")

            color = np.array(Image.open(color_path).convert("RGB"))
            depth_np = np.load(depth_path)
            depth = depth_np[depth_np.files[0]]

            # Intrinsics are stored as plain JSON and mapped into CameraIntrinsics.
            intr_data = json.loads(intr_path.read_text())
            distortion = intr_data.get("distortion")
            cx = intr_data.get("cx", intr_data.get("ppx"))
            cy = intr_data.get("cy", intr_data.get("ppy"))
            intrinsics = CameraIntrinsics(
                fx=intr_data["fx"],
                fy=intr_data["fy"],
                cx=cx,
                cy=cy,
                width=intr_data["width"],
                height=intr_data["height"],
                distortion=np.array(distortion) if distortion else None,
            )
            frames.append(CachedFrame(color=color, depth=depth, intrinsics=intrinsics))

        return frames

    def start(self):
        """No-op for compatibility."""

    def stop(self):
        """No-op for compatibility."""

    def capture_frame(self) -> np.ndarray:
        return self._frames[self._cursor].color

    def get_depth(self) -> np.ndarray:
        return self._frames[self._cursor].depth

    def get_aligned_frames(self):
        frame = self._frames[self._cursor]
        self._current_intrinsics = frame.intrinsics
        if self._cursor < len(self._frames) - 1:
            self._cursor += 1
        return frame.color, frame.depth

    def get_camera_intrinsics(self) -> CameraIntrinsics:
        return self._current_intrinsics


class NullRobot:
    """Robot stub to avoid opening a hardware robot connection."""

    def get_robot_state(self):
        return None


def _parse_args() -> argparse.Namespace:
    """Parse CLI args for single-world or batch replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-dir", default=None, help="Single world bundle directory")
    parser.add_argument(
        "--captured-root",
        default="outputs/captured_worlds",
        help="Root containing named world bundles for --run-all",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run benchmark on each subdir under --captured-root that contains TASK.md",
    )
    parser.add_argument("--task", default=None, help="Override task text (single-world mode only)")
    parser.add_argument(
        "--solver-backend",
        default="pyperplan",
        help="Solver backend: auto|pyperplan|fast-downward-docker|fast-downward-apptainer",
    )
    parser.add_argument(
        "--solver-timeout",
        type=float,
        default=120.0,
        help="Planner timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--benchmark-output-root",
        default="outputs/benchmarks/saved_world_replays",
        help="Root for replay outputs and summary files",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional explicit JSON summary path. Defaults under --benchmark-output-root.",
    )
    return parser.parse_args()


async def main() -> None:
    """CLI entrypoint with orchestration flow aligned to orchestrator demo."""
    args = _parse_args()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY/GOOGLE_API_KEY. Saved-world replay requires live staged analysis."
        )

    benchmark_output_root = Path(args.benchmark_output_root).resolve()
    benchmark_output_root.mkdir(parents=True, exist_ok=True)

    # Batch mode scans child dirs; single-world mode uses one explicit path.
    if args.run_all:
        captured_root = Path(args.captured_root).resolve()
        world_dirs = [d for d in sorted(captured_root.iterdir()) if d.is_dir() and (d / "TASK.md").exists()]
    else:
        world_dirs = [Path(args.world_dir).resolve()]

    results: List[Dict[str, Any]] = []
    for world_dir in world_dirs:
        # Read task directly from CLI override or TASK.md for this world.
        task = args.task if (args.task and not args.run_all) else (world_dir / "TASK.md").read_text(
            encoding="utf-8"
        ).strip()
        print(f"[replay] world={world_dir.name} task={task}")

        # Camera owns loading of cached replay frames from the saved world.
        camera = SnapshotCamera(world_dir)
        num_snapshots = camera.num_frames
        num_objects_source = json.loads((world_dir / "registry.json").read_text()).get("num_objects", 0)

        # Every replay writes to its own run directory to avoid state carry-over.
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_state_dir = benchmark_output_root / f"{world_dir.name}_{run_ts}"

        cfg = OrchestratorConfig(
            # Mirror orchestrator demo behavior for state updates and planning.
            api_key=api_key,
            state_dir=run_state_dir,
            update_interval=2.0,
            min_observations=2,
            solver_backend=args.solver_backend,
            solver_timeout=args.solver_timeout,
            auto_refine_on_failure=True,
            max_refinement_attempts=3,
            auto_save=True,
            auto_save_on_detection=True,
            auto_save_on_state_change=True,
        )
        cfg.robot = NullRobot()

        orchestrator = TaskOrchestrator(cfg, camera=camera)

        start = time.time()
        await orchestrator.initialize()
        try:
            # Always re-run task analysis and grounded perception from replayed snapshots.
            await orchestrator.process_task_request(task)
            await orchestrator.start_detection()

            # Match TAMP plan_task flow: check monitor state but proceed with planning attempt.
            decision = await orchestrator.monitor.determine_state()
            print(f"  • Task monitor state: {decision.state.value} (confidence={decision.confidence:.2f})")
            if decision.blockers:
                print(f"  • Blockers: {', '.join(decision.blockers)}")

            # Use the same planning path as TAMP: solve with wait_for_objects=True so
            # refinement (including goal-object mismatch updates) can run on failures.
            if orchestrator.config.auto_refine_on_failure:
                result = await orchestrator.solve_and_plan_with_refinement(wait_for_objects=True)
            else:
                result = await orchestrator.solve_and_plan(wait_for_objects=True)
            elapsed = time.time() - start

            diagnostics = (orchestrator.task_analysis.diagnostics if orchestrator.task_analysis else {})
            last_validation = diagnostics.get("last_validation", {})
            grounding = orchestrator.task_analysis.grounding_summary if orchestrator.task_analysis else None
            error_message = result.error_message or ""

            solver_success = bool(result.success) and result.plan_length > 0
            payload: Dict[str, Any] = {
                "world_name": world_dir.name,
                "world_dir": str(world_dir.resolve()),
                "task": task,
                "num_objects_source": num_objects_source,
                "num_snapshots_source": num_snapshots,
                "detection_callbacks": orchestrator.detection_count,
                "num_objects_replayed": len(orchestrator.get_detected_objects()),
                "solver_success": bool(result.success),
                "success": solver_success,
                "plan": result.plan,
                "plan_length": result.plan_length,
                "error_message": error_message,
                "refinement_attempts": orchestrator.refinement_attempts,
                "solver_backend": orchestrator.solver.backend.value,
                "search_time": result.search_time,
                "elapsed_seconds": round(elapsed, 3),
                "llm_call_count": diagnostics.get("llm_call_count"),
                "llm_elapsed_seconds": diagnostics.get("llm_elapsed_seconds"),
                "validation_valid": last_validation.get("valid"),
                "validation_layers": last_validation.get("layer_validity", {}),
                "validation_issues": last_validation.get("issues", []),
                "grounding_missing_references": grounding.missing_references if grounding else [],
                "grounded_goal_literals": grounding.grounded_goal_literals if grounding else [],
                "undefined_symbol_error": any(
                    token in error_message.lower() for token in ["undefined", "not defined", "unknown object"]
                ),
                "run_state_dir": str(run_state_dir.resolve()),
            }
        finally:
            await orchestrator.shutdown()

        results.append(payload)
        print(
            f"[replay] done world={world_dir.name} success={payload['success']} "
            f"solver_success={payload['solver_success']} plan_length={payload['plan_length']} "
            f"refinements={payload['refinement_attempts']} elapsed={payload['elapsed_seconds']}s"
        )

    # Compute aggregate metrics after all worlds are replayed.
    total = len(results)
    successes = sum(1 for r in results if r.get("success"))
    solver_successes = sum(1 for r in results if r.get("solver_success"))
    non_empty = sum(1 for r in results if (r.get("plan_length") or 0) > 0)
    failures = sum(1 for r in results if not r.get("success"))
    avg_refine = (sum(r.get("refinement_attempts", 0) for r in results) / total) if total else 0.0
    avg_elapsed = (sum(r.get("elapsed_seconds", 0.0) for r in results) / total) if total else 0.0
    avg_llm_calls = (sum(r.get("llm_call_count", 0) or 0 for r in results) / total) if total else 0.0
    avg_grounding_missing = (
        sum(len(r.get("grounding_missing_references", [])) for r in results) / total
    ) if total else 0.0
    undefined_symbol_errors = sum(1 for r in results if r.get("undefined_symbol_error"))

    summary = {
        "total_worlds": total,
        "success_count": successes,
        "failure_count": failures,
        "success_rate": round(successes / total, 3) if total else 0.0,
        "solver_success_count": solver_successes,
        "solver_success_rate": round(solver_successes / total, 3) if total else 0.0,
        "non_empty_plan_count": non_empty,
        "non_empty_plan_rate": round(non_empty / total, 3) if total else 0.0,
        "avg_refinement_attempts": round(avg_refine, 3),
        "avg_elapsed_seconds": round(avg_elapsed, 3),
        "avg_llm_call_count": round(avg_llm_calls, 3),
        "avg_grounding_missing_references": round(avg_grounding_missing, 3),
        "undefined_symbol_error_count": undefined_symbol_errors,
        "results": results,
    }

    print(json.dumps(summary, indent=2))

    if args.output_json:
        out_path = Path(args.output_json).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(args.benchmark_output_root).resolve() / f"summary_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary JSON: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
