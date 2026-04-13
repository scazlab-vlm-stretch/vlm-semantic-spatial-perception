"""
Capture a reusable world snapshot bundle from a live RealSense stream.

Output bundle includes:
- state.json
- registry.json
- perception_pool/

This does NOT execute robot primitives.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.orchestrator_config import OrchestratorConfig
from src.planning.task_orchestrator import TaskOrchestrator


class NullRobot:
    """Stub robot provider so no robot connection is attempted."""

    def get_robot_state(self):
        return None


def _check_realsense_connected() -> None:
    """Fail fast with a clear message when no RealSense device is visible."""
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        raise RuntimeError(
            "pyrealsense2 import failed. Install/activate RealSense Python bindings."
        ) from exc

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError(
            "No RealSense device detected. Check USB/power/permissions and run "
            "`rs-enumerate-devices` to confirm visibility."
        )


def _require_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    return key


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Task used to ground perception/predicates")
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="How long to run live detection in seconds (default: 15)",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/captured_worlds",
        help="Root directory for captured bundles",
    )
    parser.add_argument(
        "--update-interval",
        type=float,
        default=2.0,
        help="Detection interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--min-detection-callbacks",
        type=int,
        default=1,
        help="Require at least this many detection callbacks before saving (default: 1)",
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        default=1,
        help="Require at least this many tracked objects before saving (default: 1)",
    )
    parser.add_argument(
        "--delete-failed",
        action="store_true",
        help="Delete output directory when capture quality checks fail (default: keep for debugging)",
    )
    return parser.parse_args()


async def _run_capture(args: argparse.Namespace) -> Path:
    _check_realsense_connected()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    state_dir = Path(args.output_root).resolve() / ts

    cfg = OrchestratorConfig(
        api_key=_require_api_key(),
        update_interval=args.update_interval,
        min_observations=2,
        state_dir=state_dir,
        auto_save=True,
        auto_save_on_detection=True,
        auto_save_on_state_change=True,
    )
    cfg.robot = NullRobot()

    orchestrator = TaskOrchestrator(cfg)
    await orchestrator.initialize()

    max_objects_seen = 0
    total_detections_seen = 0
    try:
        await orchestrator.process_task_request(args.task)
        await orchestrator.start_detection()
        deadline = asyncio.get_event_loop().time() + args.duration
        while asyncio.get_event_loop().time() < deadline:
            callbacks = orchestrator.detection_count
            objects = len(orchestrator.get_detected_objects())
            stats = await orchestrator.tracker.get_stats() if orchestrator.tracker else None
            total_detections = stats.total_detections if stats else 0

            max_objects_seen = max(max_objects_seen, objects)
            total_detections_seen = max(total_detections_seen, total_detections)

            print(
                f"[capture] callbacks={callbacks} objects_now={objects} "
                f"max_objects={max_objects_seen} total_detections={total_detections_seen}"
            )
            if (
                callbacks >= args.min_detection_callbacks
                and max_objects_seen >= args.min_objects
            ):
                break
            await asyncio.sleep(1.0)

        await orchestrator.stop_detection()

        callbacks = orchestrator.detection_count
        objects_now = len(orchestrator.get_detected_objects())
        if callbacks < args.min_detection_callbacks or max_objects_seen < args.min_objects:
            # Persist debug artifacts anyway so failure is inspectable.
            await orchestrator.save_snapshot(reason="manual-final-failed")
            await orchestrator.save_state()
            raise RuntimeError(
                "Capture quality check failed: "
                f"detection_callbacks={callbacks} (min {args.min_detection_callbacks}), "
                f"objects_now={objects_now}, max_objects_seen={max_objects_seen} "
                f"(min {args.min_objects}), total_detections_seen={total_detections_seen}. "
                f"Artifacts kept at {state_dir}."
            )

        # Ensure perception_pool exists even when periodic snapshot timing is unlucky.
        await orchestrator.save_snapshot(reason="manual-final")
        await orchestrator.save_state()
    finally:
        await orchestrator.shutdown()

    return state_dir


def main() -> None:
    args = _parse_args()
    try:
        output_dir = asyncio.run(_run_capture(args))
    except RuntimeError as exc:
        print(f"Capture failed: {exc}")
        if args.delete_failed:
            # Optional cleanup if caller only wants successful bundles.
            msg = str(exc)
            if "Artifacts kept at " in msg:
                failed_path = Path(msg.split("Artifacts kept at ", 1)[1].rstrip("."))
                if failed_path.exists():
                    import shutil
                    shutil.rmtree(failed_path, ignore_errors=True)
        return
    except KeyboardInterrupt:
        print("Capture cancelled.")
        return
    print(f"Saved capture bundle: {output_dir}")


if __name__ == "__main__":
    main()
