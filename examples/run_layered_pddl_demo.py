"""
Layered PDDL Pipeline — Visual Demo

Runs each layer of the LayeredDomainGenerator in sequence, visualising the
scene and robot in PyBullet GUI and logging structured output at every step.

Usage:
    export GEMINI_API_KEY="..."
    uv run python examples/run_layered_pddl_demo.py

Controls:
    • The PyBullet window shows the xArm7 and scene objects in 3D.
    • Each test step pauses for STEP_PAUSE seconds so you can inspect the view.
    • Set STEP_PAUSE = 0 to run without pauses.

Requirements:
    pybullet, scipy, google-genai, python-dotenv (all in pyproject.toml)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
config_path = PROJECT_ROOT / "config"
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging — colour-coded, timestamped
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"
_BLUE   = "\033[34m"
_GREY   = "\033[90m"


def _setup_logging() -> logging.Logger:
    class _Fmt(logging.Formatter):
        _LEVEL_COLORS = {
            "DEBUG":    _GREY,
            "INFO":     _CYAN,
            "WARNING":  _YELLOW,
            "ERROR":    _RED,
            "CRITICAL": _RED + _BOLD,
        }
        def format(self, record):
            color = self._LEVEL_COLORS.get(record.levelname, "")
            ts = self.formatTime(record, "%H:%M:%S")
            name = record.name.split(".")[-1]
            msg = record.getMessage()
            return f"{_GREY}{ts}{_RESET} {color}{record.levelname:<8}{_RESET} {_GREY}[{name}]{_RESET} {msg}"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_Fmt())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return logging.getLogger("demo")


log = _setup_logging()

from src.kinematics.sim import SceneEnvironment

# Alias so all the step functions keep working without renaming
SceneVisualiser = SceneEnvironment


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

STEP_PAUSE = 2.0  # seconds to display each step's result in the GUI


def _banner(title: str) -> None:
    width = 70
    print()
    print(_BOLD + _BLUE + "=" * width + _RESET)
    print(_BOLD + _BLUE + f"  {title}" + _RESET)
    print(_BOLD + _BLUE + "=" * width + _RESET)


def _section(title: str) -> None:
    print()
    print(_BOLD + _CYAN + f"── {title} " + "─" * max(0, 66 - len(title)) + _RESET)


def _ok(msg: str) -> None:
    print(_GREEN + "  ✓ " + _RESET + msg)


def _warn(msg: str) -> None:
    print(_YELLOW + "  ⚠ " + _RESET + msg)


def _fail(msg: str) -> None:
    print(_RED + "  ✗ " + _RESET + msg)


def _info(label: str, value: Any) -> None:
    v = value if isinstance(value, str) else json.dumps(value, indent=2)
    # Indent multi-line values
    lines = v.splitlines()
    if len(lines) == 1:
        print(f"    {_YELLOW}{label}:{_RESET} {lines[0]}")
    else:
        print(f"    {_YELLOW}{label}:{_RESET}")
        for line in lines:
            print(f"      {line}")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = False
        self.error: Optional[str] = None
        self.duration: float = 0.0


async def _run_step(
    name: str,
    coro,
    vis: SceneVisualiser,
    results: List[TestResult],
) -> None:
    result = TestResult(name)
    _banner(name)
    vis.set_status(f"Running: {name}", color=[0.3, 0.8, 1.0])
    vis.step(0.3)

    t0 = time.monotonic()
    try:
        await coro
        result.passed = True
        result.duration = time.monotonic() - t0
        _ok(f"PASSED in {result.duration:.1f}s")
        vis.set_status(f"✓ {name}", color=[0.2, 0.9, 0.2])
    except AssertionError as e:
        result.duration = time.monotonic() - t0
        result.error = str(e)
        _fail(f"ASSERTION FAILED: {e}")
        vis.set_status(f"✗ {name}", color=[1.0, 0.3, 0.3])
    except Exception as e:
        result.duration = time.monotonic() - t0
        result.error = f"{type(e).__name__}: {e}"
        _fail(f"ERROR: {result.error}")
        vis.set_status(f"✗ {name}", color=[1.0, 0.3, 0.3])

    results.append(result)
    vis.step(STEP_PAUSE)


# ---------------------------------------------------------------------------
# Individual test coroutines
# ---------------------------------------------------------------------------

async def step_l1(generator, vis: SceneVisualiser) -> None:
    from src.planning.layered_domain_generator import extract_predicate_name_from_literal

    _section("L1 — Goal Extraction")
    vis.set_status("L1: extracting goal predicates…")

    l1 = await generator._run_l1(TASK, SCENE_OBJECTS)

    _info("goal_predicates", l1.goal_predicates)
    _info("goal_objects", l1.goal_objects)
    _info("global_predicates", l1.global_predicates)

    # Highlight objects mentioned in goals
    goal_text = " ".join(l1.goal_predicates)
    mentioned = [o["object_id"] for o in SCENE_OBJECTS if o["object_id"] in goal_text]
    if mentioned:
        vis.highlight_objects(mentioned)
        log.info("L1 goal references: %s", mentioned)

    assert len(l1.goal_predicates) > 0, "L1 must produce at least one goal predicate"
    for gp in l1.goal_predicates:
        assert gp.startswith("(") and gp.endswith(")"), f"Not a PDDL literal: {gp}"
        assert "?" not in gp, f"Goal must be grounded (no ?-vars): {gp}"
    assert "red_block_1" in goal_text or "blue_block_1" in goal_text, (
        "Goal doesn't mention either block"
    )

    errors = generator._validate_l1(l1, SCENE_OBJECTS)
    if errors:
        _warn(f"Validation warnings: {errors}")
    else:
        _ok("L1 validation clean")

    generator._l1_cache = l1  # stash for downstream steps


async def step_l1_empty_scene(generator, vis: SceneVisualiser) -> None:
    _section("L1 — Empty Scene (graceful degradation)")
    vis.set_status("L1 empty scene test…")

    l1 = await generator._run_l1(TASK, scene_objects=[])
    _info("goal_predicates (empty scene)", l1.goal_predicates)

    errors = generator._validate_l1(l1, scene_objects=[])
    _info("validation_errors", errors)
    log.info("Empty-scene L1 completed without exception — %d errors recorded", len(errors))


async def step_l2(generator, vis: SceneVisualiser) -> None:
    from src.planning.layered_domain_generator import extract_predicate_name_from_literal

    _section("L2 — Predicate Vocabulary")
    vis.set_status("L2: building predicate vocabulary…")

    l1 = getattr(generator, "_l1_cache", None)
    if l1 is None:
        l1 = await generator._run_l1(TASK, SCENE_OBJECTS)
        generator._l1_cache = l1

    l2 = await generator._run_l2(TASK, l1, SCENE_OBJECTS)
    _info("predicate_signatures", l2.predicate_signatures)
    _info("sensed_predicates", l2.sensed_predicates)
    _info("checked_variants (auto-generated)", l2.checked_variants)

    assert len(l2.predicate_signatures) > 0

    # Goal coverage
    defined = {extract_predicate_name_from_literal(s) for s in l2.predicate_signatures}
    for gp in l1.goal_predicates:
        name = extract_predicate_name_from_literal(gp)
        assert name in defined, f"Goal predicate '{name}' missing from L2 vocabulary"

    errors = generator._validate_l2(l2, l1)
    if errors:
        _warn(f"L2 validation warnings: {errors}")
    else:
        _ok("L2 validation clean")

    generator._l2_cache = l2


async def step_l2_autorepair(vis: SceneVisualiser) -> None:
    from src.planning.layered_domain_generator import (
        LayeredDomainGenerator,
        extract_predicate_name_from_literal,
    )
    from src.planning.utils.task_types import L2PredicateArtifact

    _section("L2 — Auto-repair of checked-X variants (no LLM call)")
    vis.set_status("L2 auto-repair test…")

    gen = LayeredDomainGenerator.__new__(LayeredDomainGenerator)
    gen.dkb = None

    l2 = L2PredicateArtifact(
        predicate_signatures=["(on ?obj ?surface)", "(holding ?obj)", "(graspable ?obj)"],
        sensed_predicates=["graspable"],
        checked_variants=[],
    )
    repaired = gen._repair_l2_auto(l2)

    _info("signatures after repair", repaired.predicate_signatures)
    _info("checked_variants", repaired.checked_variants)

    checked = {extract_predicate_name_from_literal(s) for s in repaired.predicate_signatures}
    assert "checked-graspable" in checked, "Auto-repair must add checked-graspable"
    _ok("checked-graspable auto-generated correctly")


async def step_l3(generator, vis: SceneVisualiser) -> None:
    from src.planning.layered_domain_generator import (
        extract_predicate_name_from_literal,
        extract_predicates_from_formula,
    )

    _section("L3 — Action Schemas")
    vis.set_status("L3: generating action schemas…")

    l1 = getattr(generator, "_l1_cache", None) or await generator._run_l1(TASK, SCENE_OBJECTS)
    l2 = getattr(generator, "_l2_cache", None) or await generator._run_l2(TASK, l1, SCENE_OBJECTS)
    generator._l1_cache = l1
    generator._l2_cache = l2

    l3 = await generator._run_l3(TASK, l1, l2)

    _info("actions", [a.get("name") for a in l3.actions])
    _info("sensing_actions", [a.get("name") for a in l3.sensing_actions])

    for action in l3.actions:
        name = action.get("name", "<unnamed>")
        _info(f"  {name} params", action.get("parameters", []))
        _info(f"  {name} precondition", action.get("precondition", ""))
        _info(f"  {name} effect", action.get("effect", ""))

    # Symbolic closure check
    vocab = {extract_predicate_name_from_literal(s) for s in l2.predicate_signatures}
    for action in l3.actions:
        for formula in [action.get("precondition", ""), action.get("effect", "")]:
            used = extract_predicates_from_formula(formula)
            undefined = used - vocab
            assert not undefined, (
                f"Action '{action.get('name')}' uses undefined predicates: {undefined}"
            )

    errors = generator._validate_l3(l3, l2, l1)
    if errors:
        _warn(f"L3 validation warnings: {errors}")
    else:
        _ok("L3 validation clean — symbolic closure verified")

    generator._l3_cache = l3


async def step_l3_reachability(generator, vis: SceneVisualiser) -> None:
    from src.planning.layered_domain_generator import (
        bfs_reachable_predicates,
        extract_predicate_name_from_literal,
    )

    _section("L3 — Goal Achievability (BFS Relaxed Planning Graph)")
    vis.set_status("L3 goal achievability check…")

    l1 = getattr(generator, "_l1_cache", None) or await generator._run_l1(TASK, SCENE_OBJECTS)
    l2 = getattr(generator, "_l2_cache", None) or await generator._run_l2(TASK, l1, SCENE_OBJECTS)
    l3 = getattr(generator, "_l3_cache", None) or await generator._run_l3(TASK, l1, l2)

    all_actions = l3.actions + l3.sensing_actions
    global_preds = {pr.strip("()").split()[0] for pr in l1.global_predicates}
    vocab = {extract_predicate_name_from_literal(s) or "" for s in l2.predicate_signatures}
    reachable = bfs_reachable_predicates(all_actions, global_preds | vocab)

    _info("initial predicate set", sorted(global_preds | vocab))
    _info("reachable predicates", sorted(reachable))

    for gp in l1.goal_predicates:
        name = extract_predicate_name_from_literal(gp)
        if name in reachable:
            _ok(f"Goal predicate '{name}' is reachable")
        else:
            assert False, f"Goal predicate '{name}' NOT reachable"


async def step_l4(generator, vis: SceneVisualiser) -> None:
    _section("L4 — Grounding Pre-check (algorithmic)")
    vis.set_status("L4: grounding pre-check…")

    l1 = getattr(generator, "_l1_cache", None) or await generator._run_l1(TASK, SCENE_OBJECTS)
    l2 = getattr(generator, "_l2_cache", None) or await generator._run_l2(TASK, l1, SCENE_OBJECTS)
    l3 = getattr(generator, "_l3_cache", None) or await generator._run_l3(TASK, l1, l2)

    l4 = generator._run_l4_precheck(l1, l3, SCENE_OBJECTS)

    _info("object_bindings", l4.object_bindings)
    if l4.warnings:
        for w in l4.warnings:
            _warn(w)
    else:
        _ok("No grounding warnings — all action types have matching objects")

    # Highlight objects that were bound
    bound_ids = [v for v in l4.object_bindings.values() if v in {o["object_id"] for o in SCENE_OBJECTS}]
    if bound_ids:
        vis.highlight_objects(bound_ids)

    generator._l4_cache = l4


async def step_l5(generator, vis: SceneVisualiser) -> None:
    _section("L5 — Initial State Construction (algorithmic)")
    vis.set_status("L5: building initial state…")

    l1 = getattr(generator, "_l1_cache", None) or await generator._run_l1(TASK, SCENE_OBJECTS)
    l2 = getattr(generator, "_l2_cache", None) or await generator._run_l2(TASK, l1, SCENE_OBJECTS)
    l3 = getattr(generator, "_l3_cache", None) or await generator._run_l3(TASK, l1, l2)

    l5 = generator._run_l5(l2, l3, SCENE_OBJECTS, l1)

    _info(f"true_literals ({len(l5.true_literals)} total)", l5.true_literals)
    _info(f"false_literals ({len(l5.false_literals)} total, first 5)", l5.false_literals[:5])

    assert len(l5.true_literals) > 0, "L5 produced no true literals"

    checked_in_true = [(pred, args) for (pred, args) in l5.true_literals if pred.startswith("checked-")]
    assert not checked_in_true, f"checked-* must be FALSE, not TRUE: {checked_in_true}"
    _ok("All checked-* predicates initialised to FALSE")

    known_ids = {o["object_id"] for o in SCENE_OBJECTS}
    for pred, args in l5.true_literals + l5.false_literals:
        for arg in args:
            assert arg in known_ids, f"Unknown object '{arg}' in literal ({pred} {args})"
    _ok("All literal arguments are known scene objects")

    generator._l5_cache = l5


async def step_full_pipeline(generator, vis: SceneVisualiser) -> None:
    _section("Full Pipeline — generate_domain()")
    vis.set_status("Full pipeline: generating layered artifact…")

    from src.planning.utils.task_types import LayeredDomainArtifact

    artifact = await generator.generate_domain(TASK, SCENE_OBJECTS)

    _info("L1 goal_predicates", artifact.l1.goal_predicates)
    _info("L2 predicate_signatures", artifact.l2.predicate_signatures)
    _info("L3 actions", [a.get("name") for a in artifact.l3.actions])
    _info("L3 sensing_actions", [a.get("name") for a in artifact.l3.sensing_actions])
    _info("L4 warnings", artifact.l4.warnings if artifact.l4 else [])
    _info("L5 true_literal count", len(artifact.l5.true_literals) if artifact.l5 else 0)

    assert isinstance(artifact, LayeredDomainArtifact)
    assert len(artifact.l1.goal_predicates) > 0
    assert len(artifact.l2.predicate_signatures) > 0
    assert len(artifact.l3.actions) > 0
    assert artifact.l4 is not None
    assert artifact.l5 is not None

    _ok("LayeredDomainArtifact produced with all layers populated")
    generator._artifact_cache = artifact


async def step_backward_compat(generator, vis: SceneVisualiser) -> None:
    from src.planning.utils.task_types import TaskAnalysis

    _section("Backward Compatibility — to_task_analysis() bridge")
    vis.set_status("Bridge: LayeredDomainArtifact → TaskAnalysis…")

    artifact = getattr(generator, "_artifact_cache", None)
    if artifact is None:
        artifact = await generator.generate_domain(TASK, SCENE_OBJECTS)

    ta = artifact.to_task_analysis()

    _info("TaskAnalysis.goal_predicates", ta.goal_predicates)
    _info("TaskAnalysis.required_actions", [a.get("name") for a in ta.required_actions])
    _info("TaskAnalysis.relevant_predicates", ta.relevant_predicates)

    assert isinstance(ta, TaskAnalysis)
    assert len(ta.goal_predicates) > 0
    assert len(ta.relevant_predicates) > 0
    assert len(ta.required_actions) > 0
    _ok("Bridge produced valid TaskAnalysis — existing write path compatible")


async def step_e2e_solve(generator, api_key: str, vis: SceneVisualiser) -> None:
    from src.planning.pddl_representation import PDDLRepresentation
    from src.planning.pddl_domain_maintainer import PDDLDomainMaintainer
    from src.planning.pddl_solver import PDDLSolver, SolverBackend

    _section("End-to-End — NL → PDDL domain → Solver")
    vis.set_status("E2E: solving blocksworld…")

    artifact = getattr(generator, "_artifact_cache", None)
    if artifact is None:
        artifact = await generator.generate_domain(TASK, SCENE_OBJECTS)

    pddl = PDDLRepresentation(domain_name="blocksworld", problem_name="test")
    maintainer = PDDLDomainMaintainer(pddl, api_key=api_key)
    ta = await maintainer.initialize_from_layered_artifact(artifact)
    _info("TaskAnalysis from maintainer", ta.goal_predicates)

    objects_list = [
        {"object_id": o["object_id"], "object_type": o["object_type"],
         "affordances": o["affordances"], "position_3d": o["position_3d"]}
        for o in SCENE_OBJECTS
    ]
    await maintainer.update_from_observations(objects_list,
                                              predicates = [
            "graspable red_block_1",
            "hand-empty",
            "graspable blue_block_1",
        ])
    await maintainer.set_goal_from_task_analysis()
    await maintainer.validate_and_fix_action_predicates()

    output_dir = PROJECT_ROOT / "outputs" / "layered_pddl_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_text = pddl.generate_domain_pddl()
    problem_text = pddl.generate_problem_pddl()
    domain_path = output_dir / "domain.pddl"
    problem_path = output_dir / "problem.pddl"
    domain_path.write_text(domain_text, encoding="utf-8")
    problem_path.write_text(problem_text, encoding="utf-8")

    _info("Domain written to", str(domain_path))
    _section("Generated PDDL Domain")
    print(domain_text)
    _section("Generated PDDL Problem")
    print(problem_text)

    log.info("Running PDDL solver…")
    solver = PDDLSolver(backend=SolverBackend.AUTO, verbose=False)
    result = await solver.solve(str(domain_path), str(problem_path), timeout=60.0)

    _info("solver.success", result.success)
    _info("plan", result.plan)
    _info("plan_length", result.plan_length)
    if result.error_message:
        _info("error_message", result.error_message)

    if result.success:
        _ok(f"Solver found plan of length {result.plan_length}")
        vis.set_status(f"✓ Plan found! {result.plan_length} steps", color=[0.2, 0.9, 0.2])
    else:
        assert result.success, (
            f"Solver failed.\nError: {result.error_message}\n"
            f"Domain:\n{domain_text}\n\nProblem:\n{problem_text}"
        )


async def step_sim_perception(api_key: str, vis: SceneVisualiser) -> None:
    """
    Capture an RGB+depth frame from the PyBullet camera and run the VLM
    object detector on it, demonstrating the full sim perception loop.
    """
    from src.perception.object_tracker import ObjectTracker

    _section("Simulated Camera Perception")
    vis.set_status("Capturing from sim camera…")

    color, depth, intrinsics = vis.capture_camera_frame()
    assert color is not None, "Sim camera returned no frame"

    # Save the rendered image for inspection
    output_dir = PROJECT_ROOT / "outputs" / "layered_pddl_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(color).save(output_dir / "sim_camera_frame.png")
        _ok(f"Saved sim camera frame → {output_dir / 'sim_camera_frame.png'}")
    except Exception as e:
        _warn(f"Could not save image: {e}")

    _info("frame shape", list(color.shape))
    _info("depth range (m)", f"{float(depth.min()):.3f} – {float(depth.max()):.3f}")

    vis.set_status("Running VLM object detection on sim frame…")

    tracker = ObjectTracker(
        api_key=api_key,
        model_name="gemini-2.0-flash",
        task_context=TASK,
    )

    detected = await tracker.detect_objects(
        color_frame=color,
        depth_frame=depth,
        camera_intrinsics=intrinsics,
    )

    _info("detected objects", [
        {"id": d.object_id, "type": d.object_type, "affordances": d.affordances,
         "position_3d": d.position_3d}
        for d in detected
    ])

    assert len(detected) > 0, (
        "VLM detected no objects in the sim frame. "
        "Check that the robot pose aims the camera at the scene objects."
    )
    _ok(f"VLM detected {len(detected)} object(s) from simulated camera frame")

    # Highlight detected objects in the visualiser
    detected_ids = [d.object_id for d in detected]
    vis.highlight_objects([oid for oid in detected_ids if oid in vis._obj_ids])

    # Stash for downstream use
    vis._detected_objects = detected


async def step_dkb(generator, vis: SceneVisualiser) -> None:
    _section("Domain Knowledge Base — record_execution()")
    vis.set_status("DKB: recording execution…")

    from src.planning.domain_knowledge_base import DomainKnowledgeBase

    dkb_dir = PROJECT_ROOT / "outputs" / "layered_pddl_demo" / "dkb"
    dkb = DomainKnowledgeBase(dkb_dir)
    dkb.load()

    artifact = getattr(generator, "_artifact_cache", None)
    if artifact is None:
        artifact = await generator.generate_domain(TASK, SCENE_OBJECTS)

    dkb.record_execution(TASK, artifact)

    history_path = dkb_dir / "execution_history.jsonl"
    assert history_path.exists(), "execution_history.jsonl not created"
    lines = history_path.read_text(encoding="utf-8").strip().splitlines()
    entry = json.loads(lines[-1])

    _info("DKB history entry", entry)
    _info("predicate library size", len(dkb._predicates))
    _info("action library size", len(dkb._actions))

    assert entry["task"] == TASK
    assert entry.get("l1_goal_count", 0) > 0
    _ok(f"DKB recorded execution. History file: {history_path}")


# ---------------------------------------------------------------------------
# Scene fixture
# ---------------------------------------------------------------------------

TASK = "Put the red block on the blue block"

SCENE_OBJECTS = [
    {
        "object_id": "red_block_1",
        "object_type": "block",
        "affordances": ["graspable", "stackable"],
        "position_3d": [0.3, 0.0, 0.12],
    },
    {
        "object_id": "blue_block_1",
        "object_type": "block",
        "affordances": ["graspable", "stackable"],
        "position_3d": [0.5, 0.0, 0.12],
    },
    {
        "object_id": "table_1",
        "object_type": "surface",
        "affordances": ["support_surface"],
        "position_3d": [0.4, 0.0, 0.0],
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY environment variable is not set. Exiting.")
        sys.exit(1)

    from src.planning.layered_domain_generator import LayeredDomainGenerator

    _banner("Layered PDDL Pipeline — Visual Demo")
    log.info("Task: %s", TASK)
    log.info("Scene objects: %s", [o["object_id"] for o in SCENE_OBJECTS])
    log.info("PyBullet available: %s", PYBULLET_AVAILABLE)

    # Start visualiser
    vis = SceneVisualiser()
    vis.start()
    vis.add_scene_objects(SCENE_OBJECTS)
    vis.step(1.0)

    generator = LayeredDomainGenerator(api_key=api_key, max_layer_retries=2)

    results: List[TestResult] = []

    steps = [
        ("Sim Camera Perception",        lambda: step_sim_perception(api_key, vis)),
        ("L1: Goal Extraction",         lambda: step_l1(generator, vis)),
        ("L1: Empty Scene Graceful",     lambda: step_l1_empty_scene(generator, vis)),
        ("L2: Predicate Vocabulary",     lambda: step_l2(generator, vis)),
        ("L2: Auto-Repair checked-X",    lambda: step_l2_autorepair(vis)),
        ("L3: Action Schemas",           lambda: step_l3(generator, vis)),
        ("L3: Goal Achievability (BFS)", lambda: step_l3_reachability(generator, vis)),
        ("L4: Grounding Pre-check",      lambda: step_l4(generator, vis)),
        ("L5: Initial State",            lambda: step_l5(generator, vis)),
        ("Full Pipeline",                lambda: step_full_pipeline(generator, vis)),
        ("Backward Compat Bridge",       lambda: step_backward_compat(generator, vis)),
        ("E2E: NL → PDDL → Solve",       lambda: step_e2e_solve(generator, api_key, vis)),
        ("DKB: Record Execution",        lambda: step_dkb(generator, vis)),
    ]

    for name, coro_factory in steps:
        await _run_step(name, coro_factory(), vis, results)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    _banner("Results Summary")
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    for r in results:
        if r.passed:
            print(f"  {_GREEN}PASS{_RESET}  {r.name:<45} {_GREY}{r.duration:.1f}s{_RESET}")
        else:
            print(f"  {_RED}FAIL{_RESET}  {r.name:<45} {_GREY}{r.duration:.1f}s{_RESET}")
            print(f"       {_RED}{r.error}{_RESET}")

    print()
    print(f"  {_BOLD}{_GREEN}{len(passed)} passed{_RESET}, {_BOLD}{_RED}{len(failed)} failed{_RESET} out of {len(results)} steps")

    if failed:
        vis.set_status(f"{len(failed)} steps failed — see terminal", color=[1.0, 0.3, 0.3])
    else:
        vis.set_status("All steps passed!", color=[0.2, 0.9, 0.2])

    vis.step(STEP_PAUSE * 2)
    vis.stop()

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    asyncio.run(main())
