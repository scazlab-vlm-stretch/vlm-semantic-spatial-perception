"""
Stretch Sim Demo
================

End-to-end demo using the Hello Robot Stretch RE1V0 in PyBullet.

Scene:
  - Brown table with a blue block on top (pickup location)
  - Red table (dropoff location)
  - Stretch robot starts between the two tables

Task:
  Pick up the blue block from the brown table and place it on the red table.

Perception backend:
  LLM (Gemini/Qwen3-VL) by default; set USE_GSAM2=1 for RAM+ + GroundingDINO + SAM2.

Usage:
    export GEMINI_API_KEY="..."
    uv run python examples/run_stretch_sim_demo.py

Optional env vars:
    DEMO_TASK          Override task description
    DEMO_DETECT_CYCLES Max detection cycles (default: 3)
    DEMO_STEP_PAUSE    Pause between milestones in seconds (default: 2)
    DEMO_OUTPUT_DIR    Base output dir (default: outputs/stretch_demo)
    HF_MODEL           HuggingFace model ID (falls back to Gemini if unset)
    GEMINI_MODEL       Gemini model for planning (default: gemini-3-flash-preview)
    DECOMPOSER_MODEL   Gemini model for skill decomposition
    USE_GSAM2          Set to 1 to use GSAM2 perception backend
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
_config_path = PROJECT_ROOT / "config"
if str(_config_path) not in sys.path:
    sys.path.insert(0, str(_config_path))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"
_BLUE   = "\033[34m"
_GREY   = "\033[90m"
_MAG    = "\033[35m"


def _setup_logging() -> logging.Logger:
    class _Fmt(logging.Formatter):
        _MAP = {"DEBUG": _GREY, "INFO": _CYAN, "WARNING": _YELLOW,
                "ERROR": _RED, "CRITICAL": _RED + _BOLD}
        def format(self, r):
            color = self._MAP.get(r.levelname, "")
            ts = self.formatTime(r, "%H:%M:%S")
            name = r.name.split(".")[-1][:16]
            return f"{_GREY}{ts}{_RESET} {color}{r.levelname:<8}{_RESET} {_GREY}[{name}]{_RESET} {r.getMessage()}"

    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_Fmt())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(logging.INFO)
    for noisy in ("urllib3", "httpx", "httpcore", "google", "PIL", "models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("stretch_demo")


log = _setup_logging()


def _banner(msg: str) -> None:
    w = 72
    print()
    print(_BOLD + _BLUE + "=" * w + _RESET)
    print(_BOLD + _BLUE + f"  {msg}" + _RESET)
    print(_BOLD + _BLUE + "=" * w + _RESET)


def _section(msg: str) -> None:
    print()
    print(_BOLD + _CYAN + f"── {msg} " + "─" * max(0, 68 - len(msg)) + _RESET)


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET} {msg}")


def _info(label: str, value: Any = "") -> None:
    print(f"    {_YELLOW}{label}:{_RESET} {value}")


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------

TASK = os.getenv(
    "DEMO_TASK",
    "Pick up the blue block from the brown table and place it on the red table.",
)

HF_MODEL        = os.getenv("HF_MODEL", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DECOMPOSER_MODEL= os.getenv("DECOMPOSER_MODEL", "gemini-2.5-flash")

USE_GSAM2             = os.getenv("USE_GSAM2", "").strip() in ("1", "true", "yes")
GSAM2_SAM2_CFG        = os.getenv("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")
GSAM2_SAM2_CKPT       = os.getenv("SAM2_CKPT", "")
GSAM2_RAM_CKPT        = os.getenv("RAM_CKPT", "")
GSAM2_GROUNDING_MODEL = os.getenv("GROUNDING_MODEL", "IDEA-Research/grounding-dino-tiny")
GSAM2_DET_INTERVAL    = int(os.getenv("GSAM2_DET_INTERVAL", "20"))
GSAM2_TAG_INTERVAL    = int(os.getenv("GSAM2_TAG_INTERVAL", "30"))

# Scene layout (world frame, metres):
#   Stretch spawns at origin. The arm extends along -Y (robot right side).
#   Tables are placed along -Y so the arm can reach them directly.
#   Brown table: directly to the robot's right at (0, -1.0).
#   Red table:   further right and forward at (0.5, -1.0).
#   Blue block sits on top of the brown table.
#
# Table geometry: pedestal boxes so top surface at 0.60m — within lift range.
_TABLE_Z_HALF = 0.30    # half-height → top surface at 0.60 m
_TABLE_Z_CTR  = _TABLE_Z_HALF
_BLOCK_Z      = _TABLE_Z_HALF * 2 + 0.03   # block centre resting on table top

SCENE_OBJECTS = [
    {
        "object_id":   "brown_table_1",
        "object_type": "surface",
        "affordances": ["support_surface"],
        "position_3d": [-0.4, -1.1, _TABLE_Z_CTR],
    },
    {
        "object_id":   "red_table_1",
        "object_type": "surface",
        "affordances": ["support_surface"],
        "position_3d": [0.6, -1.1, _TABLE_Z_CTR],
    },
    {
        "object_id":   "blue_block_1",
        "object_type": "block",
        "affordances": ["graspable", "stackable"],
        "position_3d": [-0.4, -1.1, _BLOCK_Z],
    },
]

# Custom colours for the Stretch scene (object_id → RGBA)
OBJECT_COLORS: Dict[str, List[float]] = {
    "brown_table_1": [0.55, 0.35, 0.15, 1.0],
    "red_table_1":   [0.80, 0.10, 0.10, 1.0],
    "blue_block_1":  [0.15, 0.35, 0.85, 1.0],
}

# Half-extents: tables are pedestals (flat top, full-height body with no separate legs)
OBJECT_HALF_EXTENTS: Dict[str, List[float]] = {
    "brown_table_1": [0.30, 0.30, _TABLE_Z_HALF],
    "red_table_1":   [0.30, 0.30, _TABLE_Z_HALF],
    "blue_block_1":  [0.03, 0.03, 0.03],
}

MAX_DETECT_CYCLES = int(os.getenv("DEMO_DETECT_CYCLES", "3"))
STEP_PAUSE        = float(os.getenv("DEMO_STEP_PAUSE", "2"))

_RUN_ID          = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_BASE_OUTPUT_DIR = Path(os.getenv("DEMO_OUTPUT_DIR", "outputs/stretch_demo"))
OUTPUT_DIR       = _BASE_OUTPUT_DIR / "runs" / _RUN_ID

# Stretch movable joint ordering from URDF (12 total, including grippers):
#  [0]  joint_right_wheel          (R) PyBullet idx 0
#  [1]  joint_left_wheel           (R) PyBullet idx 1
#  [2]  joint_lift                 (P) PyBullet idx 4
#  [3]  joint_arm_l3               (P) PyBullet idx 6
#  [4]  joint_arm_l2               (P) PyBullet idx 7
#  [5]  joint_arm_l1               (P) PyBullet idx 8
#  [6]  joint_arm_l0               (P) PyBullet idx 9
#  [7]  joint_wrist_yaw            (R) PyBullet idx 10
#  [8]  joint_gripper_finger_left  (R) PyBullet idx 13
#  [9]  joint_gripper_finger_right (R) PyBullet idx 15
#  [10] joint_head_pan             (R) PyBullet idx 22
#  [11] joint_head_tilt            (R) PyBullet idx 23
#
# The arm extends along -Y. Pan the head -π/2 to look along -Y toward the tables.
# head_tilt=-0.5 looks slightly down at the 0.60m table top from ~1.0m camera height.
import math as _math
STRETCH_CAMERA_AIM_JOINTS = [
    0.0,              # [0]  right_wheel
    0.0,              # [1]  left_wheel
    0.65,             # [2]  lift — camera at ~1.0 m, sees table top at 0.60 m
    0.0,              # [3]  arm_l3
    0.0,              # [4]  arm_l2
    0.0,              # [5]  arm_l1
    0.0,              # [6]  arm_l0
    0.0,              # [7]  wrist_yaw
    0.0,              # [8]  gripper_finger_left
    0.0,              # [9]  gripper_finger_right
    -_math.pi / 2,   # [10] head_pan — rotate to look along -Y (arm/table side)
    -0.5,             # [11] head_tilt — look slightly down at table surface
]


# ---------------------------------------------------------------------------
# Simulated frame provider
# ---------------------------------------------------------------------------

class SimFrameProvider:
    def __init__(self, env) -> None:
        self._env = env

    def __call__(self) -> Tuple:
        color, depth, intrinsics = self._env.capture_camera_frame()
        robot_state = self._env.get_robot_state()
        if color is None:
            return None, None, None, None
        return color, depth, intrinsics, robot_state


# ---------------------------------------------------------------------------
# Detection progress tracker
# ---------------------------------------------------------------------------

class DetectionTracker:
    def __init__(self, env, min_cycles: int) -> None:
        self._env        = env
        self._min_cycles = min_cycles
        self.cycle       = 0
        self._ready      = asyncio.Event()

    def on_detection(self, object_count: int) -> None:
        self.cycle += 1
        log.info("Detection cycle %d — %d objects observed", self.cycle, object_count)
        self._env.set_status(
            f"Detection {self.cycle}/{self._min_cycles}: {object_count} objects",
            color=[0.3, 0.8, 1.0],
        )
        if self.cycle >= self._min_cycles:
            self._ready.set()

    async def wait_until_ready(self, timeout: float = 120.0) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def run_demo() -> int:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    use_hf  = bool(HF_MODEL)

    _timings: Dict[str, float] = {}
    _t_total = time.monotonic()

    def _tick(label: str) -> float:
        return time.monotonic()

    def _tock(label: str, t0: float) -> float:
        elapsed = time.monotonic() - t0
        _timings[label] = elapsed
        return elapsed

    _banner("Stretch Simulation Demo")
    _info("Task",        TASK)
    _info("Scene",       [o["object_id"] for o in SCENE_OBJECTS])
    _info("Output",      OUTPUT_DIR)
    if USE_GSAM2:
        _info("Perception", "GSAM2 (RAM+ + GroundingDINO + SAM2)")
    else:
        _info("Perception", f"HuggingFace ({HF_MODEL})" if use_hf else f"Google GenAI ({GEMINI_MODEL})")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _latest = _BASE_OUTPUT_DIR / "runs" / "latest"
    if _latest.is_symlink() or _latest.exists():
        _latest.unlink()
    _latest.symlink_to(OUTPUT_DIR.resolve())
    _info("Run ID", _RUN_ID)

    # ------------------------------------------------------------------
    # 0. LLM client
    # ------------------------------------------------------------------
    llm_client = None
    from src.llm_interface import GoogleGenAIClient, Qwen3VLClient

    if use_hf:
        _section("Loading Qwen3-VL model")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _t0 = _tick("model_load")
        llm_client = Qwen3VLClient(
            model=HF_MODEL,
            device="auto" if device == "cuda" else device,
            load_in_4bit=device == "cuda",
        )
        _tock("model_load", _t0)
        _ok(f"Qwen3-VL loaded on {device}")
    else:
        if not api_key and not USE_GSAM2:
            log.error("GEMINI_API_KEY is not set.")
            return 1
        if api_key:
            llm_client = GoogleGenAIClient(model=GEMINI_MODEL, api_key=api_key)
            _ok(f"Google GenAI client ready (model={GEMINI_MODEL})")

    # ------------------------------------------------------------------
    # 1. Start PyBullet GUI with Stretch URDF
    # ------------------------------------------------------------------
    _section("Starting PyBullet Stretch environment")
    from src.kinematics.sim.scene_environment import SceneEnvironment
    from src.kinematics.stretch_pybullet_interface import _STRETCH_URDF

    env = SceneEnvironment(
        urdf_path=_STRETCH_URDF,
        camera_link="camera_color_optical_frame",
        initial_joints=STRETCH_CAMERA_AIM_JOINTS,
        n_arm_joints=12,   # all 12 movable joints (includes grippers + head pan/tilt)
        tcp_link_name="link_grasp_center",
        # Use world-up so the rendered image is upright despite the Stretch camera mounting.
        camera_use_world_up=True,
        # Third-person viewer: look from +X/+Y corner down at robot and tables along -Y
        viewer_target=[0.2, -0.6, 0.4],
        viewer_distance=2.8,
        viewer_yaw=-30,
        viewer_pitch=-25,
    )
    env.start()

    # Patch module-level colour/size dicts before spawning objects so that
    # the Stretch-specific colours and table sizes are used, then restore.
    import src.kinematics.sim.scene_environment as _se_mod
    _orig_colors  = _se_mod.OBJECT_COLORS
    _orig_extents = _se_mod.OBJECT_HALF_EXTENTS
    _se_mod.OBJECT_COLORS       = {**_orig_colors,  **OBJECT_COLORS}
    _se_mod.OBJECT_HALF_EXTENTS = {**_orig_extents, **OBJECT_HALF_EXTENTS}
    env.add_scene_objects(SCENE_OBJECTS)
    _se_mod.OBJECT_COLORS       = _orig_colors
    _se_mod.OBJECT_HALF_EXTENTS = _orig_extents

    env.set_status("Initialising…")
    env.step(0.5)
    _ok("PyBullet GUI started — Stretch robot in scene")

    frame_provider = SimFrameProvider(env)

    color, depth, intrinsics, _ = frame_provider()
    assert color is not None, "Sim camera returned no frame — check Stretch URDF camera link"
    _ok(f"Sim camera verified: {color.shape[1]}×{color.shape[0]} RGB")

    try:
        from PIL import Image
        Image.fromarray(color).save(OUTPUT_DIR / "initial_camera_view.png")
        _info("Camera preview", OUTPUT_DIR / "initial_camera_view.png")
    except Exception:
        pass

    env.step(0.3)

    # ------------------------------------------------------------------
    # 2. Orchestrator config
    # ------------------------------------------------------------------
    _section("Configuring TaskOrchestrator")
    from orchestrator_config import OrchestratorConfig
    from src.planning.task_orchestrator import TaskOrchestrator

    detect_tracker = DetectionTracker(env, min_cycles=MAX_DETECT_CYCLES)

    config = OrchestratorConfig(
        api_key=api_key or "unused",
        llm_client=llm_client,

        use_sim_camera=True,
        enable_depth=True,

        update_interval=4.0,
        min_observations=MAX_DETECT_CYCLES,
        fast_mode=True,

        state_dir=OUTPUT_DIR / "orchestrator_state",
        auto_save=True,
        auto_save_on_detection=True,
        auto_save_on_state_change=True,

        enable_snapshots=True,
        snapshot_every_n_detections=1,
        perception_pool_dir=OUTPUT_DIR / "perception_pool",
        debug_frames_dir=OUTPUT_DIR / "debug_frames",

        solver_backend="auto",
        solver_timeout=60.0,
        solver_verbose=False,
        auto_solve_when_ready=False,
        max_refinement_attempts=3,
        auto_refine_on_failure=True,

        use_layered_generation=True,
        dkb_dir=OUTPUT_DIR / "dkb",

        robot=env,

        on_state_change=lambda old, new: (
            log.info("Orchestrator state: %s → %s", old.value, new.value),
            env.set_status(f"State: {new.value}", color=[0.9, 0.7, 0.2]),
        ),
        on_detection_update=detect_tracker.on_detection,
        on_plan_generated=lambda result: (
            _ok(f"Plan: {result.plan_length} steps — {result.plan}"),
            env.set_status(f"Plan: {result.plan_length} steps", color=[0.2, 0.9, 0.2]),
        ),
    )
    _ok("Config built")

    # ------------------------------------------------------------------
    # 3. Initialise orchestrator
    # ------------------------------------------------------------------
    _section("Initialising orchestrator subsystems")
    orchestrator = TaskOrchestrator(config=config, camera=None)
    await orchestrator.initialize()
    _ok("Orchestrator initialised")

    # ------------------------------------------------------------------
    # 3a. Optionally swap to GSAM2 backend
    # ------------------------------------------------------------------
    if USE_GSAM2:
        _section("Swapping to GSAM2 perception backend")
        if not GSAM2_SAM2_CKPT:
            log.error("SAM2_CKPT env var is required when USE_GSAM2=1")
            return 1
        import torch
        from src.perception import GSAM2ContinuousObjectTracker
        gsam2_device = "cuda" if torch.cuda.is_available() else "cpu"
        _t0 = _tick("gsam2_model_load")
        gsam2_tracker = GSAM2ContinuousObjectTracker(
            sam2_model_cfg=GSAM2_SAM2_CFG,
            sam2_ckpt_path=GSAM2_SAM2_CKPT,
            grounding_model_id=GSAM2_GROUNDING_MODEL,
            ram_ckpt_path=GSAM2_RAM_CKPT or None,
            detection_interval=GSAM2_DET_INTERVAL,
            device=gsam2_device,
            tag_interval=GSAM2_TAG_INTERVAL,
            update_interval=config.update_interval,
            on_detection_complete=detect_tracker.on_detection,
            llm_client=llm_client,
            logger=log.getChild("GSAM2Tracker"),
            debug_save_dir=OUTPUT_DIR / "debug_frames",
        )
        _tock("gsam2_model_load", _t0)
        orchestrator.tracker = gsam2_tracker
        gsam2_tracker.on_detection_complete = orchestrator._on_detection_callback
        _ok(f"GSAM2 tracker active on {gsam2_device}")

    orchestrator.tracker.set_frame_provider(frame_provider)
    _ok("Sim frame provider injected")
    env.step(0.3)

    # ------------------------------------------------------------------
    # 4. Perception pre-pass
    # ------------------------------------------------------------------
    _section("Initial perception pass")
    env.set_status("Detecting scene objects…", color=[0.8, 0.6, 0.2])
    orchestrator.current_task = TASK
    orchestrator.tracker.set_task_context(task_description=TASK)

    _t0 = _tick("perception_prepass")
    await orchestrator.start_detection()
    await detect_tracker.wait_until_ready(timeout=60.0)
    await orchestrator.stop_detection()
    _tock("perception_prepass", _t0)

    pre_detected = orchestrator.get_detected_objects()
    if pre_detected:
        _ok(f"Pre-detected {len(pre_detected)} object(s): {[o.object_id for o in pre_detected]}")
    else:
        _warn("No objects detected in pre-pass")

    # ------------------------------------------------------------------
    # 5. Layered domain generation
    # ------------------------------------------------------------------
    _section("Processing task — layered domain generation")
    env.set_status("L1–L5 domain generation…", color=[0.3, 0.8, 1.0])

    _t0 = _tick("domain_generation")
    task_analysis = await orchestrator.process_task_request(TASK, environment_image=color)
    _tock("domain_generation", _t0)

    _ok(f"Domain generated in {_timings['domain_generation']:.1f}s")
    _info("goal_predicates",  task_analysis.goal_predicates)
    _info("goal_objects",     task_analysis.goal_objects)
    env.step(STEP_PAUSE)

    # ------------------------------------------------------------------
    # 6. Continuous detection
    # ------------------------------------------------------------------
    _section("Continuous detection")
    env.set_status("Detecting…", color=[0.3, 0.8, 1.0])
    detect_tracker._ready.clear()
    detect_tracker.cycle = 0

    _t0 = _tick("continuous_detection")
    await orchestrator.start_detection()
    ready = await detect_tracker.wait_until_ready(timeout=MAX_DETECT_CYCLES * 30.0)
    if not ready:
        _warn("Detection timeout — proceeding with what was observed")
    await orchestrator.stop_detection()
    _tock("continuous_detection", _t0)

    detected = orchestrator.get_detected_objects()
    _ok(f"Detection complete — {len(detected)} objects in registry")
    for obj in detected:
        _info(f"  {obj.object_id}", f"type={obj.object_type} pos3d={getattr(obj,'position_3d',None)}")

    _latest_debug = OUTPUT_DIR / "debug_frames" / "latest.png"
    if _latest_debug.exists():
        _ok(f"Debug frames → {OUTPUT_DIR / 'debug_frames'} (latest: latest.png)")
    else:
        _warn("No debug frames written yet")

    env.highlight_objects([o.object_id for o in detected
                           if o.object_id in {s["object_id"] for s in SCENE_OBJECTS}])
    env.step(STEP_PAUSE)

    # ------------------------------------------------------------------
    # 7. PDDL generation
    # ------------------------------------------------------------------
    _section("Generating PDDL domain & problem files")
    env.set_status("Generating PDDL…", color=[0.9, 0.7, 0.2])

    _t0 = _tick("pddl_generation")
    pddl_paths = await orchestrator.generate_pddl_files(
        output_dir=OUTPUT_DIR / "pddl",
        set_goals=True,
    )
    _tock("pddl_generation", _t0)

    domain_path  = pddl_paths.get("domain_path")
    problem_path = pddl_paths.get("problem_path")
    _ok(f"Domain  → {domain_path}")
    _ok(f"Problem → {problem_path}")

    if domain_path and Path(domain_path).exists():
        _section("Generated PDDL Domain")
        print(Path(domain_path).read_text())
    if problem_path and Path(problem_path).exists():
        _section("Generated PDDL Problem")
        print(Path(problem_path).read_text())

    env.step(STEP_PAUSE)

    # ------------------------------------------------------------------
    # 8. Solve + execute
    # ------------------------------------------------------------------
    _section("Solving PDDL plan (with refinement)")
    env.set_status("Solving…", color=[0.9, 0.7, 0.2])

    _t0 = _tick("pddl_solving")
    result = await orchestrator.solve_and_plan_with_refinement(
        output_dir=OUTPUT_DIR / "pddl",
        wait_for_objects=False,
    )
    _tock("pddl_solving", _t0)

    _info("success",     result.success)
    _info("plan_length", result.plan_length)
    _info("search_time", f"{result.search_time:.2f}s" if result.search_time else "n/a")

    if result.success:
        _ok("Plan found!")
        for i, step in enumerate(result.plan, 1):
            _info(f"  step {i}", step)
        env.set_status(f"✓ Plan: {result.plan_length} steps", color=[0.2, 0.9, 0.2])

        # ------------------------------------------------------------------
        # 8b. Skill decomposition + Stretch primitive execution
        # ------------------------------------------------------------------
        _section("Skill decomposition & execution")

        from src.primitives.skill_decomposer import SkillDecomposer
        from src.primitives.primitive_executor import PrimitiveExecutor
        from src.kinematics.sim.stretch_pybullet_primitives import StretchPyBulletPrimitives

        stretch_primitives = StretchPyBulletPrimitives(
            env=env,
            registry=orchestrator.tracker.registry,
        )
        executor = PrimitiveExecutor(
            primitives=stretch_primitives,
            perception_pool_dir=OUTPUT_DIR / "perception_pool",
        )
        decomposer = SkillDecomposer(
            api_key=api_key or os.getenv("GOOGLE_API_KEY", ""),
            model_name=DECOMPOSER_MODEL,
            orchestrator=orchestrator,
        )

        _step_decompose_times: List[float] = []
        _step_execute_times:   List[float] = []

        _t0_exec = _tick("execution_total")
        for i, step in enumerate(result.plan, 1):
            parts = step.strip("()").split()
            if not parts:
                continue
            action_name = parts[0]
            param_keys  = [f"param{j+1}" for j in range(len(parts) - 1)]
            parameters  = dict(zip(param_keys, parts[1:]))
            if parts[1:]:
                parameters["object_ids"] = parts[1:]

            env.set_status(f"Step {i}/{result.plan_length}: {step}", color=[0.3, 0.8, 1.0])
            _info(f"step {i}", step)

            # ------------------------------------------------------------------
            # Sensing actions (check-*): resolve immediately from the registry.
            # The planner uses check-X ?obj to discover sensed facts at runtime.
            # Since perception has already run, we read the affordances directly
            # from the registry and assert checked-X TRUE — no motion needed.
            # Example: (check-object-graspable blue_block_2) → look up blue_block_2,
            # confirm "graspable" is in its affordances, set checked-object-graspable TRUE.
            # ------------------------------------------------------------------
            if action_name.startswith("check-"):
                # The base affordance name is everything after "check-"
                # e.g. "check-object-graspable" → base_affordance = "object-graspable"
                # Normalize to underscore form for registry affordance lookup.
                checked_pred = f"checked-{action_name[len('check-'):]}"
                base_affordance_dashed = action_name[len("check-"):]
                base_affordance_under = base_affordance_dashed.replace("-", "_")
                # Also handle "object-graspable" → "graspable" (strip "object-" prefix)
                stripped = base_affordance_under
                if stripped.startswith("object_"):
                    stripped = stripped[len("object_"):]

                obj_ids_in_step = parts[1:]  # all object arguments to this action
                resolved_count = 0
                for obj_id in obj_ids_in_step:
                    obj = orchestrator.tracker.registry.get_object(obj_id)
                    if obj is None:
                        _warn(f"  [check] Object '{obj_id}' not in registry — skipping")
                        continue
                    affs = {a.replace("-", "_").lower() for a in (obj.affordances or set())}
                    # Accept the affordance if any of the normalized forms match
                    has_it = (
                        base_affordance_under in affs
                        or stripped in affs
                        or base_affordance_dashed.replace("-", "_") in affs
                    )
                    if has_it:
                        try:
                            orchestrator.pddl.add_initial_literal(checked_pred, [obj_id], negated=False)
                            resolved_count += 1
                            _ok(f"  [check] {checked_pred}({obj_id}) ← TRUE (affordance '{stripped}' confirmed)")
                        except Exception as exc:
                            _warn(f"  [check] Could not assert {checked_pred}({obj_id}): {exc}")
                    else:
                        _warn(f"  [check] {checked_pred}({obj_id}) stays FALSE — '{stripped}' not in affordances {sorted(affs)}")
                _ok(f"  [check] Sensing action resolved ({resolved_count}/{len(obj_ids_in_step)} objects)")
                env.step(STEP_PAUSE)
                continue

            try:
                _t0_dec = time.monotonic()
                skill_plan = decomposer.plan(action_name, parameters)
                _dt_dec = time.monotonic() - _t0_dec
                _step_decompose_times.append(_dt_dec)
                _ok(f"  → {len(skill_plan.primitives)} primitive(s): "
                    f"{[pc.name for pc in skill_plan.primitives]} ({_dt_dec:.2f}s)")

                world_state = orchestrator.get_world_state_snapshot() if hasattr(orchestrator, "get_world_state_snapshot") else {}
                _t0_run = time.monotonic()
                exec_result = executor.execute_plan(skill_plan, world_state=world_state)
                _dt_run = time.monotonic() - _t0_run
                _step_execute_times.append(_dt_run)
                _ok(f"  executed={exec_result.executed} ({_dt_run:.2f}s)")

            except Exception as exc:
                _warn(f"  Decomposition/execution failed: {exc}")

            env.step(STEP_PAUSE)

        _tock("execution_total", _t0_exec)
        _timings["skill_decomposition"] = sum(_step_decompose_times)
        _timings["primitive_execution"]  = sum(_step_execute_times)

        env.set_status("✓ Execution complete", color=[0.2, 0.9, 0.2])
        env.step(STEP_PAUSE)

    else:
        _warn(f"No plan found: {result.error_message}")
        env.set_status("✗ No plan found", color=[1.0, 0.3, 0.3])

    env.step(STEP_PAUSE * 1.5)

    # ------------------------------------------------------------------
    # 9. Save + shutdown
    # ------------------------------------------------------------------
    _section("Saving orchestrator state")
    saved_path = await orchestrator.save_state()
    _ok(f"State saved → {saved_path}")

    _section("Shutting down")
    await orchestrator.shutdown()
    env.step(1.0)
    env.stop()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _t_total_elapsed = time.monotonic() - _t_total

    _banner("Stretch Demo Complete")
    _info("Task",         TASK)
    _info("Run ID",       _RUN_ID)
    _info("Plan success", result.success)
    if result.success:
        _info("Plan length", result.plan_length)
        _info("Plan",        result.plan)
    _info("Output dir",  OUTPUT_DIR)

    _TIMING_LABELS = [
        ("model_load",           "Model load"),
        ("gsam2_model_load",     "Model load (GSAM2)"),
        ("perception_prepass",   "Perception pre-pass"),
        ("domain_generation",    "Domain generation (L1–L5)"),
        ("continuous_detection", "Continuous detection"),
        ("pddl_generation",      "PDDL file generation"),
        ("pddl_solving",         "PDDL solving (+ refinement)"),
        ("skill_decomposition",  "Skill decomposition (total)"),
        ("primitive_execution",  "Primitive execution (total)"),
        ("execution_total",      "Execution phase (total)"),
    ]

    print()
    print(_BOLD + _BLUE + "── Timing Report " + "─" * 55 + _RESET)
    col_w = 34
    for key, label in _TIMING_LABELS:
        if key in _timings:
            t = _timings[key]
            pct = 100.0 * t / _t_total_elapsed if _t_total_elapsed > 0 else 0.0
            print(f"  {_YELLOW}{label:<{col_w}}{_RESET} {t:6.2f}s  ({pct:4.1f}%)")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_demo()))
