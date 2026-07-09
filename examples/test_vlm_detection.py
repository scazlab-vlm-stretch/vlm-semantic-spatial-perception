"""
Minimal VLM detection demo using the same orchestrator-backed perception path as
`examples/run_stretch_sim_demo.py`, but without simulation, PDDL generation,
planning, or execution.

This script:
1. Loads a single RGB image from disk.
2. Creates the same LLM client family used by the Stretch demo.
3. Initializes `TaskOrchestrator`, which in turn builds the VLM-based
   `ContinuousObjectTracker`.
4. Injects a static frame provider that repeatedly serves the same image.
5. Runs a small number of detection cycles and prints the detected objects.

Usage:
    export GEMINI_API_KEY="..."
    uv run python examples/test_vlm_detection.py --image path/to/scene.png

Optional env vars:
    HF_MODEL        HuggingFace model ID (falls back to Gemini if unset)
    GEMINI_MODEL    Gemini model for perception (default: gemini-3-flash-preview)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
_config_path = PROJECT_ROOT / "config"
if str(_config_path) not in sys.path:
    sys.path.insert(0, str(_config_path))

load_dotenv(PROJECT_ROOT / ".env")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal orchestrator-backed VLM detection test.")
    parser.add_argument("--image", type=Path, required=True, help="Path to an RGB image.")
    parser.add_argument(
        "--task",
        default="Identify the visible objects in the scene.",
        help="Task context passed to the VLM detection prompt.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of detection callbacks to wait for before stopping.",
    )
    parser.add_argument(
        "--update-interval",
        type=float,
        default=2.0,
        help="Seconds between continuous detection attempts.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Skip detailed interaction-point analysis for faster detection.",
    )
    return parser.parse_args()


def _load_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


class StaticFrameProvider:
    """Serve a fixed RGB frame for orchestrator-backed detection."""

    def __init__(self, color: np.ndarray) -> None:
        self._color = color

    def __call__(self) -> Tuple[np.ndarray, None, None, None]:
        return self._color.copy(), None, None, None


class DetectionTracker:
    """Wait for a target number of detection callbacks."""

    def __init__(self, min_cycles: int) -> None:
        self._min_cycles = min_cycles
        self.cycle = 0
        self._ready = asyncio.Event()

    def on_detection(self, object_count: int) -> None:
        self.cycle += 1
        print(f"[cycle {self.cycle}] detected {object_count} object(s)")
        if self.cycle >= self._min_cycles:
            self._ready.set()

    async def wait_until_ready(self, timeout: float = 60.0) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


def _print_objects(objects: List[Any]) -> None:
    print()
    print("=" * 60)
    print("DETECTED OBJECTS")
    print("=" * 60)

    if not objects:
        print("No objects detected")
        return

    for i, obj in enumerate(objects, 1):
        print(f"{i}. {obj.object_id} ({obj.object_type})")
        if obj.bounding_box_2d:
            print(f"   bbox_2d: {obj.bounding_box_2d}")
        if obj.position_2d:
            print(f"   position_2d: {obj.position_2d}")
        if obj.affordances:
            print(f"   affordances: {sorted(obj.affordances)}")


async def _build_llm_client(api_key: Optional[str]) -> Optional[Any]:
    hf_model = os.getenv("HF_MODEL", "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip()

    from src.llm_interface import GoogleGenAIClient, Qwen3VLClient

    if hf_model:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading HuggingFace VLM: {hf_model} on {device}")
        return Qwen3VLClient(
            model=hf_model,
            device="auto" if device == "cuda" else device,
            load_in_4bit=device == "cuda",
        )

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY must be set when HF_MODEL is not used.")

    print(f"Using Google GenAI model: {gemini_model}")
    return GoogleGenAIClient(model=gemini_model, api_key=api_key)


async def main() -> int:
    args = _parse_args()
    image_path = args.image.resolve()
    color = _load_rgb_image(image_path)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    llm_client = await _build_llm_client(api_key)

    from orchestrator_config import OrchestratorConfig
    from src.planning.task_orchestrator import TaskOrchestrator

    detect_tracker = DetectionTracker(min_cycles=max(1, args.cycles))

    config = OrchestratorConfig(
        api_key=api_key or "unused",
        llm_client=llm_client,
        use_sim_camera=True,
        enable_depth=False,
        update_interval=args.update_interval,
        min_observations=max(1, args.cycles),
        fast_mode=args.fast_mode,
        auto_save=False,
        auto_save_on_detection=False,
        auto_save_on_state_change=False,
        enable_snapshots=False,
        debug_frames_dir=None,
        solver_backend="auto",
        auto_solve_when_ready=False,
        auto_refine_on_failure=False,
        use_layered_generation=False,
        on_detection_update=detect_tracker.on_detection,
    )

    orchestrator = TaskOrchestrator(config=config, camera=None)
    await orchestrator.initialize()

    orchestrator.tracker.set_frame_provider(StaticFrameProvider(color))
    orchestrator.current_task = args.task
    orchestrator.tracker.set_task_context(task_description=args.task)

    print(f"Image: {image_path}")
    print(f"Task context: {args.task}")
    print(f"Detection cycles: {args.cycles}")
    print()

    try:
        await orchestrator.start_detection()
        ready = await detect_tracker.wait_until_ready()
        if not ready:
            print("Timed out waiting for detection callbacks.")
    finally:
        await orchestrator.stop_detection()

    objects = orchestrator.get_detected_objects()
    _print_objects(objects)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
