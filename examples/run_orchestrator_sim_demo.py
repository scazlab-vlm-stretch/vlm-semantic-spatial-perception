"""
Orchestrator Simulation Demo
=============================

Full end-to-end demo of the TaskOrchestrator with:
  • PyBullet GUI — xArm7 robot + coloured scene objects
  • Simulated wrist camera feeding the perception loop
  • Layered PDDL domain generation (L1–L5 gated pipeline)
  • Continuous object detection updating the PDDL domain
  • PDDL solver producing an executable plan
  • Terminal logging of every pipeline stage with timing

Perception backends (set USE_GSAM2=1 to use RAM+ + GroundingDINO + SAM2):
  • LLM (default) — Qwen3-VL or Google Gemini via GEMINI_API_KEY
  • GSAM2          — RAM+ tagger → GroundingDINO → SAM2 segmentation (no LLM needed)

Usage:
    # LLM backend (default):
    export GEMINI_API_KEY="..."
    uv run python examples/run_orchestrator_sim_demo.py

    # GSAM2 backend:
    USE_GSAM2=1 SAM2_CKPT=/path/to/sam2.1_hiera_large.pt \\
        RAM_CKPT=/path/to/ram_plus_swin_large_14m.pth \\
        uv run python examples/run_orchestrator_sim_demo.py

Optional flags (set as env vars):
    DEMO_TASK          Task description (default: color-sorting task — red block → red container, blue block → blue container)
    DEMO_DETECT_CYCLES Max detection cycles before forcing solve (default: 3)
    DEMO_STEP_PAUSE    Seconds to pause at each milestone (default: 2)
    DEMO_OUTPUT_DIR    Base output directory (default: outputs/sim_demo).
                       Each run is saved under <base>/runs/<YYYYMMDD_HHMMSS>/.
                       A symlink <base>/runs/latest always points to the most recent run.

GSAM2-specific env vars (only used when USE_GSAM2=1):
    SAM2_CFG           SAM2 config file (default: configs/sam2.1/sam2.1_hiera_l.yaml)
    SAM2_CKPT          SAM2 checkpoint path (required)
    RAM_CKPT           RAM+ checkpoint path (required)
    GROUNDING_MODEL    GroundingDINO model ID (default: IDEA-Research/grounding-dino-tiny)
    GSAM2_DET_INTERVAL GroundingDINO re-detection interval in frames (default: 20)
    GSAM2_TAG_INTERVAL RAM+ re-tagging interval in frames (default: 30)
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
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[32m"
_YELLOW= "\033[33m"
_CYAN  = "\033[36m"
_RED   = "\033[31m"
_BLUE  = "\033[34m"
_GREY  = "\033[90m"
_MAG   = "\033[35m"


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
    # Silence noisy sub-loggers
    for noisy in ("urllib3", "httpx", "httpcore", "google", "PIL", "models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("sim_demo")


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

TASK = os.getenv("DEMO_TASK", "Sort the blocks into the container that matches each block's color: put red blocks in the red container and blue blocks in the blue container")

# HuggingFace model — used for planning when set. Empty string falls back to Google GenAI.
HF_MODEL = os.getenv("HF_MODEL", "")

# Google GenAI models (used when HF_MODEL is not set)
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")          # domain generation / planning
DECOMPOSER_MODEL  = os.getenv("DECOMPOSER_MODEL", "gemini-robotics-er-1.5-preview")  # skill decomposition

# GSAM2 perception backend (set USE_GSAM2=1 to enable)
USE_GSAM2 = os.getenv("USE_GSAM2", "").strip() in ("1", "true", "yes")
GSAM2_SAM2_CFG        = os.getenv("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")
GSAM2_SAM2_CKPT       = os.getenv("SAM2_CKPT", "")
GSAM2_RAM_CKPT        = os.getenv("RAM_CKPT", "")
GSAM2_GROUNDING_MODEL = os.getenv("GROUNDING_MODEL", "IDEA-Research/grounding-dino-tiny")
GSAM2_DET_INTERVAL    = int(os.getenv("GSAM2_DET_INTERVAL", "20"))
GSAM2_TAG_INTERVAL    = int(os.getenv("GSAM2_TAG_INTERVAL", "30"))

SCENE_OBJECTS = [
    # Blocks to be sorted
    {
        "object_id":   "red_block_1",
        "object_type": "block",
        "affordances": ["graspable", "stackable"],
        "position_3d": [0.35, 0.08, 0.04],
    },
    {
        "object_id":   "blue_block_1",
        "object_type": "block",
        "affordances": ["graspable", "stackable"],
        "position_3d": [0.35, -0.08, 0.04],
    },
    # Target containers (open trays)
    {
        "object_id":   "red_container_1",
        "object_type": "container",
        "affordances": ["containable", "support_surface"],
        "position_3d": [0.52, 0.12, 0.02],
    },
    {
        "object_id":   "blue_container_1",
        "object_type": "container",
        "affordances": ["containable", "support_surface"],
        "position_3d": [0.52, -0.12, 0.02],
    },
    # Work surface
    {
        "object_id":   "table_1",
        "object_type": "surface",
        "affordances": ["support_surface"],
        "position_3d": [0.42, 0.0, 0.0],
    },
]

MAX_DETECT_CYCLES  = int(os.getenv("DEMO_DETECT_CYCLES", "3"))
STEP_PAUSE         = float(os.getenv("DEMO_STEP_PAUSE", "2"))

_RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_BASE_OUTPUT_DIR   = Path(os.getenv("DEMO_OUTPUT_DIR", "outputs/sim_demo"))
OUTPUT_DIR         = _BASE_OUTPUT_DIR / "runs" / _RUN_ID


# ---------------------------------------------------------------------------
# Simulated frame provider
# ---------------------------------------------------------------------------

class SimFrameProvider:
    """
    Wraps a SceneEnvironment and provides (color, depth, intrinsics, robot_state)
    tuples for the ContinuousObjectTracker frame provider slot.

    Also exposes robot_state via the XArmPybulletInterface so the orchestrator
    can snapshot joint angles alongside each detection.
    """

    def __init__(self, env, robot) -> None:
        self._env   = env    # SceneEnvironment (GUI)
        self._robot = robot  # XArmPybulletInterface (DIRECT, for FK)

    def __call__(self) -> Tuple:
        color, depth, intrinsics = self._env.capture_camera_frame()
        robot_state = self._robot.get_robot_state() if self._robot is not None else None
        if color is None:
            return None, None, None, None
        return color, depth, intrinsics, robot_state


# ---------------------------------------------------------------------------
# Detection progress tracker (callback)
# ---------------------------------------------------------------------------

class DetectionTracker:
    """Receives detection callbacks from the orchestrator and tracks progress."""

    def __init__(self, env, min_cycles: int) -> None:
        self._env       = env
        self._min_cycles = min_cycles
        self.cycle      = 0
        self.object_ids: List[str] = []
        self._ready     = asyncio.Event()

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
# Perception debug image
# ---------------------------------------------------------------------------

def _save_perception_debug_image(
    perception_pool_dir: Path,
    output_path: Path,
) -> Optional[Path]:
    """
    Draw bounding boxes and object labels onto the latest snapshot color image
    and save a debug PNG to *output_path*.

    Reads the snapshot index to locate the newest snapshot, then overlays:
      - A coloured bounding box per detected object (normalised 0-1000 coords)
      - The object_id label above each box
      - A small dot at position_2d (centroid) when no bounding box is present

    Returns the saved path, or None if anything is missing.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log.warning("Pillow not available — skipping perception debug image")
        return None

    index_path = perception_pool_dir / "index.json"
    if not index_path.exists():
        log.warning("No perception pool index found at %s", index_path)
        return None

    with open(index_path) as f:
        index = json.load(f)

    snapshots = index.get("snapshots") or {}
    if not snapshots:
        log.warning("Perception pool has no snapshots yet")
        return None

    # Pick the most recent snapshot by captured_at timestamp
    latest_sid = max(
        snapshots,
        key=lambda sid: snapshots[sid].get("captured_at", ""),
    )
    snap_meta = snapshots[latest_sid]
    snap_dir = perception_pool_dir / "snapshots" / latest_sid

    color_path = snap_dir / "color.png"
    det_path   = snap_dir / "detections.json"

    if not color_path.exists():
        log.warning("Snapshot %s missing color.png", latest_sid)
        return None

    img = Image.open(color_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # Try to get a font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # Colour palette cycling for distinct boxes
    _PALETTE = [
        (255,  80,  80, 200),   # red
        ( 80, 120, 255, 200),   # blue
        ( 80, 220,  80, 200),   # green
        (255, 200,  50, 200),   # yellow
        (220,  80, 220, 200),   # magenta
        ( 80, 220, 220, 200),   # cyan
        (255, 160,  50, 200),   # orange
    ]

    objects = []
    if det_path.exists():
        with open(det_path) as f:
            det = json.load(f)
        objects = det.get("objects") or []

    for i, obj in enumerate(objects):
        color = _PALETTE[i % len(_PALETTE)]
        oid   = obj.get("object_id", f"obj_{i}")
        bbox  = obj.get("bounding_box_2d")   # [y1, x1, y2, x2] in 0-1000 normalised
        pos2d = obj.get("position_2d")        # [y, x] in 0-1000 normalised

        if bbox and len(bbox) == 4:
            y1n, x1n, y2n, x2n = bbox
            x1 = int(x1n / 1000 * w)
            y1 = int(y1n / 1000 * h)
            x2 = int(x2n / 1000 * w)
            y2 = int(y2n / 1000 * h)
            # Filled semi-transparent rect + solid border
            draw.rectangle([x1, y1, x2, y2], fill=(*color[:3], 50), outline=color, width=2)
            label_y = max(0, y1 - 18)
            draw.rectangle([x1, label_y, x1 + len(oid) * 8 + 6, label_y + 16],
                           fill=(*color[:3], 200))
            draw.text((x1 + 3, label_y + 1), oid, fill=(255, 255, 255, 255), font=font)
        elif pos2d and len(pos2d) >= 2:
            yn, xn = pos2d[0], pos2d[1]
            cx = int(xn / 1000 * w)
            cy = int(yn / 1000 * h)
            r = 6
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         fill=(*color[:3], 200), outline=(255, 255, 255, 255), width=2)
            draw.text((cx + r + 2, cy - 8), oid, fill=(*color[:3], 255), font=font)

    # Draw interaction points as small crosses
    for i, obj in enumerate(objects):
        color = _PALETTE[i % len(_PALETTE)]
        for ip_name, ip in (obj.get("interaction_points") or {}).items():
            pos = ip.get("position_2d")
            if pos and len(pos) >= 2:
                cx = int(pos[1] / 1000 * w)
                cy = int(pos[0] / 1000 * h)
                arm = 4
                draw.line([cx - arm, cy, cx + arm, cy], fill=(255, 255, 255, 230), width=2)
                draw.line([cx, cy - arm, cx, cy + arm], fill=(255, 255, 255, 230), width=2)
                draw.text((cx + 5, cy - 6), ip_name, fill=(220, 220, 220, 200), font=font)

    # Watermark with snapshot id and object count
    stamp = f"snapshot: {latest_sid}  |  objects: {len(objects)}"
    draw.rectangle([0, h - 20, w, h], fill=(0, 0, 0, 160))
    draw.text((4, h - 17), stamp, fill=(200, 200, 200, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def run_demo() -> int:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    use_hf = bool(HF_MODEL)  # HF_MODEL controls planning LLM; independent of perception backend

    # -- Timing bookkeeping --------------------------------------------------
    _timings: Dict[str, float] = {}
    _t_total = time.monotonic()

    def _tick(label: str) -> float:
        """Start a timer; returns t0 for use with _tock."""
        return time.monotonic()

    def _tock(label: str, t0: float) -> float:
        """Record elapsed time for a stage."""
        elapsed = time.monotonic() - t0
        _timings[label] = elapsed
        return elapsed

    _banner("Orchestrator Simulation Demo")
    _info("Task", TASK)
    _info("Sim objects placed", [o["object_id"] for o in SCENE_OBJECTS])
    _info("Output", OUTPUT_DIR)
    if USE_GSAM2:
        _info("Perception", "GSAM2 (RAM+ + GroundingDINO + SAM2)")
        _info("SAM2 ckpt", GSAM2_SAM2_CKPT or "(not set — will fail)")
        _info("RAM+ ckpt", GSAM2_RAM_CKPT or "(not set — will fail)")
    else:
        _info("Perception", f"HuggingFace ({HF_MODEL})" if use_hf else f"Google GenAI ({GEMINI_MODEL})")
    _info("Planning LLM", f"HuggingFace ({HF_MODEL})" if use_hf else f"Google GenAI ({GEMINI_MODEL})")
    _info("Decomposer LLM", f"Google GenAI ({DECOMPOSER_MODEL})")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep a "latest" symlink pointing to this run
    _latest = _BASE_OUTPUT_DIR / "runs" / "latest"
    if _latest.is_symlink() or _latest.exists():
        _latest.unlink()
    _latest.symlink_to(OUTPUT_DIR.resolve())
    _info("Run ID", _RUN_ID)

    # ------------------------------------------------------------------
    # 0. Build LLM client for planning (always, independent of perception backend)
    # ------------------------------------------------------------------
    llm_client = None

    if True:
        from src.llm_interface import GoogleGenAIClient, Qwen3VLClient

        if use_hf:
            _section("Loading Qwen3-VL model")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _info("Model", HF_MODEL)
            _info("Device", device)
            _t0 = _tick("model_load")
            llm_client = Qwen3VLClient(
                model=HF_MODEL,
                device="auto" if device == "cuda" else device,
                load_in_4bit=device == "cuda",
                model_kwargs={"max_memory": {0: "6GiB", "cpu": "16GiB"}} if device == "cuda" else None,
            )
            _tock("model_load", _t0)
            _ok(f"Qwen3-VL model loaded on {device}")
        else:
            if not api_key:
                log.error("GEMINI_API_KEY is not set. Set HF_MODEL to use a local model or USE_GSAM2=1 for GSAM2.")
                return 1
            llm_client = GoogleGenAIClient(model=GEMINI_MODEL, api_key=api_key)
            _ok(f"Google GenAI client ready (model={GEMINI_MODEL})")

    # ------------------------------------------------------------------
    # 1. Start PyBullet GUI environment
    # ------------------------------------------------------------------
    _section("Starting PyBullet simulation environment")
    from src.kinematics.sim import SceneEnvironment, CAMERA_AIM_JOINTS

    env = SceneEnvironment()
    env.start()
    env.add_scene_objects(SCENE_OBJECTS)
    env.set_status("Initialising orchestrator…")
    env.step(0.5)
    _ok("PyBullet GUI started — xArm7 aimed at work surface")

    # SceneEnvironment acts as both the visualiser and the robot state provider.
    # No second PyBullet instance is created.
    frame_provider = SimFrameProvider(env, env)

    # Verify we can capture a frame before handing to orchestrator
    color, depth, intrinsics, _ = frame_provider()
    assert color is not None, "Sim camera returned no frame — check URDF camera link"
    _ok(f"Sim camera verified: {color.shape[1]}×{color.shape[0]} RGB, depth [{depth.min():.2f}, {depth.max():.2f}]m")

    # Save a preview of the initial camera view
    preview_path = OUTPUT_DIR / "initial_camera_view.png"
    try:
        from PIL import Image
        Image.fromarray(color).save(preview_path)
        _info("Camera preview saved", preview_path)
    except Exception:
        pass

    env.step(0.3)

    # ------------------------------------------------------------------
    # 2. Build OrchestratorConfig
    # ------------------------------------------------------------------
    _section("Configuring TaskOrchestrator")
    from orchestrator_config import OrchestratorConfig
    from src.planning.task_orchestrator import TaskOrchestrator

    detect_tracker = DetectionTracker(env, min_cycles=MAX_DETECT_CYCLES)

    config = OrchestratorConfig(
        api_key=api_key or "unused",   # required field; ignored when llm_client is set
        llm_client=llm_client,

        # Sim camera — skip RealSense init entirely
        use_sim_camera=True,
        enable_depth=True,

        # Detection
        update_interval=4.0,          # seconds between VLM calls (sim is slow to render)
        min_observations=MAX_DETECT_CYCLES,
        fast_mode=True,               # skip interaction-point refinement for speed

        # Persistence
        state_dir=OUTPUT_DIR / "orchestrator_state",
        auto_save=True,
        auto_save_on_detection=True,
        auto_save_on_state_change=True,

        # Snapshots
        enable_snapshots=True,
        snapshot_every_n_detections=1,
        perception_pool_dir=OUTPUT_DIR / "perception_pool",
        debug_frames_dir=OUTPUT_DIR / "debug_frames",

        # PDDL solver
        solver_backend="auto",
        solver_timeout=60.0,
        solver_verbose=False,
        auto_solve_when_ready=False,   # we drive this manually
        max_refinement_attempts=3,
        auto_refine_on_failure=True,

        # Layered generation — the whole point of this demo
        use_layered_generation=True,
        dkb_dir=OUTPUT_DIR / "dkb",

        # SceneEnvironment implements the duck-typed robot provider interface
        robot=env,

        # Callbacks
        on_state_change=lambda old, new: (
            log.info("Orchestrator state: %s → %s", old.value, new.value),
            env.set_status(f"State: {new.value}", color=[0.9, 0.7, 0.2]),
        ),
        on_detection_update=detect_tracker.on_detection,
        on_plan_generated=lambda result: (
            _ok(f"Plan generated! {result.plan_length} steps: {result.plan}"),
            env.set_status(f"Plan: {result.plan_length} steps", color=[0.2, 0.9, 0.2]),
        ),
    )

    perception_label = "GSAM2" if USE_GSAM2 else ("HuggingFace" if use_hf else "GenAI")
    planning_label   = "HuggingFace" if use_hf else "GenAI"
    _ok(f"Config built — use_layered_generation=True, perception={perception_label}, planning={planning_label}, solver=auto")
    _info("state_dir", config.state_dir)
    _info("dkb_dir",   config.dkb_dir)

    # ------------------------------------------------------------------
    # 3. Initialise orchestrator — no physical camera
    # ------------------------------------------------------------------
    _section("Initialising orchestrator subsystems")
    env.set_status("Initialising…")

    # Pass no camera — we'll inject frames via set_frame_provider after init
    orchestrator = TaskOrchestrator(config=config, camera=None)
    await orchestrator.initialize()
    _ok("Orchestrator initialised")

    # ------------------------------------------------------------------
    # 3a. Optionally swap to GSAM2 perception backend
    # ------------------------------------------------------------------
    if USE_GSAM2:
        _section("Swapping to GSAM2 perception backend")
        if not GSAM2_SAM2_CKPT:
            log.error("SAM2_CKPT env var is required when USE_GSAM2=1")
            return 1

        import torch
        from src.perception import GSAM2ContinuousObjectTracker

        gsam2_device = "cuda" if torch.cuda.is_available() else "cpu"
        _info("Device", gsam2_device)
        _info("Loading", "GroundingDINO + SAM2 (this may take 30-60s)…")
        log.info("Loading GSAM2 models — SAM2 ckpt: %s", GSAM2_SAM2_CKPT)
        if GSAM2_RAM_CKPT:
            log.info("RAM+ ckpt: %s", GSAM2_RAM_CKPT)
        _t0_gsam2 = _tick("gsam2_model_load")
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
        _tock("gsam2_model_load", _t0_gsam2)
        # Replace the LLM-based tracker with the GSAM2 tracker
        orchestrator.tracker = gsam2_tracker
        # Wire the orchestrator's detection callback so snapshots, PDDL updates,
        # and detection_count all work correctly through the GSAM2 tracker.
        gsam2_tracker.on_detection_complete = orchestrator._on_detection_callback
        _ok(f"GSAM2 tracker active (SAM2 + {'RAM+' if GSAM2_RAM_CKPT else 'no tagger'}) on {gsam2_device} "
            f"({_timings['gsam2_model_load']:.1f}s)")

    # Override frame provider with our sim camera
    orchestrator.tracker.set_frame_provider(frame_provider)
    _ok("Sim frame provider injected into tracker")

    env.step(0.3)

    # ------------------------------------------------------------------
    # 4. Quick perception pass — identify scene objects before planning
    # ------------------------------------------------------------------
    _section("Initial perception pass")
    env.set_status("Detecting scene objects…", color=[0.8, 0.6, 0.2])
    orchestrator.current_task = TASK  # needed by tracker before domain generation
    orchestrator.tracker.set_task_context(task_description=TASK)  # inject task hints before detection

    _t0 = _tick("perception_prepass")
    await orchestrator.start_detection()
    ready = await detect_tracker.wait_until_ready(timeout=60.0)
    await orchestrator.stop_detection()
    _tock("perception_prepass", _t0)
    pre_detected = orchestrator.get_detected_objects()
    if pre_detected:
        _ok(f"Pre-detected {len(pre_detected)} object(s): {[o.object_id for o in pre_detected]}")
    else:
        _warn("No objects detected in pre-pass — domain generation will use empty scene")

    # ------------------------------------------------------------------
    # 5. Process task request (L1→L5 layered domain generation)
    # ------------------------------------------------------------------
    _section("Processing task request — layered domain generation")
    env.set_status("L1–L5 domain generation…", color=[0.3, 0.8, 1.0])
    log.info("Task: %s", TASK)

    _t0 = _tick("domain_generation")
    # Pass the initial sim camera frame as environment context
    task_analysis = await orchestrator.process_task_request(
        TASK,
        environment_image=color,
    )
    _tock("domain_generation", _t0)

    _ok(f"Domain generated in {_timings['domain_generation']:.1f}s")
    _info("goal_predicates",    task_analysis.goal_predicates)
    _info("goal_objects",       task_analysis.goal_objects)
    _info("required_actions",   [a.get("name") for a in task_analysis.required_actions])
    _info("relevant_predicates",task_analysis.relevant_predicates)

    env.highlight_objects([o["object_id"] for o in SCENE_OBJECTS
                           if any(g in " ".join(task_analysis.goal_predicates)
                                  for g in [o["object_id"]])])
    env.step(STEP_PAUSE)

    # ------------------------------------------------------------------
    # 6. Start continuous detection — sim camera feeds VLM
    # ------------------------------------------------------------------
    _section("Starting continuous VLM detection")
    env.set_status("Detecting objects…", color=[0.3, 0.8, 1.0])
    _t0 = _tick("continuous_detection")
    await orchestrator.start_detection()
    _ok(f"Detection started — waiting for {MAX_DETECT_CYCLES} cycles")

    ready = await detect_tracker.wait_until_ready(timeout=MAX_DETECT_CYCLES * 30.0)
    if not ready:
        _warn("Detection timeout — proceeding with whatever was observed")

    await orchestrator.stop_detection()
    _tock("continuous_detection", _t0)

    detected = orchestrator.get_detected_objects()
    _ok(f"Detection complete — {len(detected)} objects in registry")
    for obj in detected:
        _info(f"  {obj.object_id}", f"type={obj.object_type} affordances={obj.affordances} pos3d={getattr(obj,'position_3d',None)}")

    # Debug frames are saved live during detection — report the latest
    _latest_debug = OUTPUT_DIR / "debug_frames" / "latest.png"
    if _latest_debug.exists():
        _ok(f"Perception debug frames → {OUTPUT_DIR / 'debug_frames'} (latest: latest.png)")
    else:
        _warn("No debug frames written yet")

    # Highlight everything detected
    env.highlight_objects([o.object_id for o in detected if o.object_id in {s["object_id"] for s in SCENE_OBJECTS}])
    env.step(STEP_PAUSE)

    # ------------------------------------------------------------------
    # 6. Check orchestrator status
    # ------------------------------------------------------------------
    _section("Orchestrator status")
    status = await orchestrator.get_status()
    _info("state",             status.get("orchestrator_state"))
    _info("detection_count",   status.get("detection_count"))
    _info("ready_for_planning",status.get("ready_for_planning"))
    monitor = status.get("monitor", {})
    if monitor:
        _info("task_state_decision", getattr(monitor, "state", monitor))

    # ------------------------------------------------------------------
    # 7. Generate PDDL files
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
    # 8. Solve — with automatic refinement on failure
    # ------------------------------------------------------------------
    _section("Solving PDDL plan (with refinement)")
    env.set_status("Solving…", color=[0.9, 0.7, 0.2])

    _t0 = _tick("pddl_solving")
    result = await orchestrator.solve_and_plan_with_refinement(
        output_dir=OUTPUT_DIR / "pddl",
        wait_for_objects=False,   # objects already detected above
    )
    _tock("pddl_solving", _t0)

    _section("Solver Result")
    _info("success",      result.success)
    _info("plan_length",  result.plan_length)
    _info("search_time",  f"{result.search_time:.2f}s" if result.search_time else "n/a")
    _info("solve_total",  f"{_timings['pddl_solving']:.2f}s")

    if result.success:
        _ok("Plan found!")
        for i, step in enumerate(result.plan, 1):
            _info(f"  step {i}", step)
        env.set_status(f"✓ Plan: {result.plan_length} steps", color=[0.2, 0.9, 0.2])

        # ------------------------------------------------------------------
        # 8b. Skill decomposition + PyBullet execution
        # ------------------------------------------------------------------
        _section("Skill decomposition & execution")

        from src.primitives.skill_decomposer import SkillDecomposer
        from src.primitives.primitive_executor import PrimitiveExecutor
        from src.kinematics.sim.pybullet_primitives import PyBulletPrimitives

        pybullet_primitives = PyBulletPrimitives(
            env=env,
            registry=orchestrator.tracker.registry,
        )
        executor = PrimitiveExecutor(
            primitives=pybullet_primitives,
            perception_pool_dir=OUTPUT_DIR / "perception_pool",
        )
        decomposer = SkillDecomposer(
            api_key=api_key or os.getenv("GOOGLE_API_KEY", ""),
            model_name=DECOMPOSER_MODEL,
            orchestrator=orchestrator,
        )

        _step_decompose_times: List[float] = []
        _step_execute_times:   List[float] = []

        _t0_exec_total = _tick("execution_total")
        for i, step in enumerate(result.plan, 1):
            parts = step.strip("()").split()
            if not parts:
                continue
            action_name = parts[0]
            # Build parameter dict: PDDL params are positional — use generic keys
            param_keys = [f"param{j+1}" for j in range(len(parts) - 1)]
            parameters = dict(zip(param_keys, parts[1:]))
            # Also expose the raw object ids for the decomposer's world-state lookup
            if parts[1:]:
                parameters["object_ids"] = parts[1:]

            env.set_status(f"Step {i}/{result.plan_length}: {step}", color=[0.3, 0.8, 1.0])
            _info(f"step {i}", step)
            _info("  decomposing", action_name)

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
                _warn(f"  Skill decomposition/execution failed: {exc}")

            env.step(STEP_PAUSE)

        _tock("execution_total", _t0_exec_total)
        _timings["skill_decomposition"] = sum(_step_decompose_times)
        _timings["primitive_execution"]  = sum(_step_execute_times)

        env.set_status("✓ Execution complete", color=[0.2, 0.9, 0.2])
        env.step(STEP_PAUSE)

    else:
        _warn(f"No plan found: {result.error_message}")
        env.set_status("✗ No plan found", color=[1.0, 0.3, 0.3])

    env.step(STEP_PAUSE * 1.5)

    # ------------------------------------------------------------------
    # 9. Save state snapshot
    # ------------------------------------------------------------------
    _section("Saving orchestrator state")
    saved_path = await orchestrator.save_state()
    _ok(f"State saved → {saved_path}")

    # ------------------------------------------------------------------
    # 10. Shutdown
    # ------------------------------------------------------------------
    _section("Shutting down")
    await orchestrator.shutdown()
    env.step(1.0)
    env.stop()

    # ------------------------------------------------------------------
    # Summary + Timing report
    # ------------------------------------------------------------------
    _t_total_elapsed = time.monotonic() - _t_total

    _banner("Demo Complete")
    _info("Task",         TASK)
    _info("Run ID",       _RUN_ID)
    _info("Plan success", result.success)
    if result.success:
        _info("Plan length", result.plan_length)
        _info("Plan",        result.plan)
    _info("Output dir",  OUTPUT_DIR)

    # Timing breakdown
    _TIMING_LABELS = [
        ("model_load",           "Model load (HF)"),
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

    # Derived totals
    planning_keys = {"domain_generation", "pddl_generation", "pddl_solving"}
    perception_keys = {"perception_prepass", "continuous_detection"}
    t_planning   = sum(_timings.get(k, 0.0) for k in planning_keys)
    t_perception = sum(_timings.get(k, 0.0) for k in perception_keys)
    t_execution  = _timings.get("execution_total", 0.0)

    print(_BOLD + _BLUE + "  " + "─" * 55 + _RESET)
    for label, t in [
        ("Total perception time",  t_perception),
        ("Total planning time",    t_planning),
        ("Total execution time",   t_execution),
        ("Total task time",        _t_total_elapsed),
    ]:
        pct = 100.0 * t / _t_total_elapsed if _t_total_elapsed > 0 else 0.0
        print(f"  {_BOLD}{label:<{col_w}}{_RESET} {t:6.2f}s  ({pct:4.1f}%)")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_demo()))
