"""
Task and Motion Planner (TAMP)

High-level integration class that combines:
- TaskOrchestrator: Manages perception, PDDL domain/problem generation, and task planning
- SkillDecomposer: Decomposes symbolic PDDL actions into primitive motion skills
- PrimitiveExecutor: Executes low-level primitives with metric translation

This provides a complete pipeline from natural language task → perception →
symbolic planning → skill decomposition → primitive execution.

Architecture:
    Natural Language Task
           ↓
    TaskOrchestrator (perception + PDDL planning)
           ↓
    PDDL Plan [action1, action2, ...]
           ↓
    SkillDecomposer (action → primitives)
           ↓
    Skill Plans [primitives for each action]
           ↓
    PrimitiveExecutor (translate + execute)
           ↓
    Robot Motion

Example:
    >>> config = TAMPConfig(
    ...     api_key=os.getenv("GEMINI_API_KEY"),
    ...     state_dir=Path("./outputs/tamp_state")
    ... )
    >>> tamp = TaskAndMotionPlanner(config)
    >>>
    >>> # Initialize and perceive environment
    >>> await tamp.initialize()
    >>> await tamp.perceive_environment(duration=10.0)
    >>>
    >>> # Plan and execute task
    >>> result = await tamp.plan_and_execute_task(
    ...     task="pick up the red block and place it on the blue block",
    ...     dry_run=False  # Set True to validate without executing
    ... )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from src.planning.task_orchestrator import TaskOrchestrator, OrchestratorState
from src.planning.pddl_solver import SolverResult
from src.primitives.skill_decomposer import SkillDecomposer
from src.primitives.primitive_executor import PrimitiveExecutor, PrimitiveExecutionResult
from src.primitives.skill_plan_types import SkillPlan
from src.utils.genai_logging import configure_genai_logging
from src.camera.realsense_camera import RealSenseCamera
# Import orchestrator config
import sys
config_path = Path(__file__).parent.parent / "config"
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))
from orchestrator_config import OrchestratorConfig


class TAMPState(Enum):
    """TAMP pipeline execution states."""
    UNINITIALIZED = "uninitialized"
    IDLE = "idle"
    PERCEIVING = "perceiving"
    PLANNING = "planning"
    DECOMPOSING = "decomposing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TAMPConfig:
    """Configuration for Task and Motion Planner."""

    # API and model configuration
    api_key: Optional[str] = None
    orchestrator_model: str = "gemini-robotics-er-1.5-preview"
    decomposer_model: str = "gemini-robotics-er-1.5-preview"

    # Paths
    state_dir: Path = field(default_factory=lambda: Path("./outputs/tamp_state"))
    perception_pool_dir: Optional[Path] = None
    primitive_catalog_path: Optional[Path] = None
    task_analyzer_prompts_path: Optional[Path] = None
    genai_log_dir: Optional[Path] = None

    # Orchestrator settings (perception + planning)
    update_interval: float = 2.0
    min_observations: int = 3
    auto_refine_on_failure: bool = True
    max_refinement_attempts: int = 5

    # Solver settings
    solver_backend: str = "auto"  # "auto", "pyperplan", "fast-downward-docker", etc.
    solver_algorithm: str = "lama-first"
    solver_timeout: float = 60.0

    # Execution settings
    dry_run_default: bool = False  # Default dry-run mode for execution
    primitives_interface: Optional[Any] = None  # Robot primitives interface (set at runtime)

    # Callbacks
    on_state_change: Optional[Callable[[TAMPState], None]] = None
    on_plan_generated: Optional[Callable[[SolverResult], None]] = None
    on_action_decomposed: Optional[Callable[[str, SkillPlan], None]] = None
    on_action_executed: Optional[Callable[[str, PrimitiveExecutionResult], None]] = None

    def to_orchestrator_config(self) -> OrchestratorConfig:
        """Convert to OrchestratorConfig for TaskOrchestrator initialization."""
        return OrchestratorConfig(
            api_key=self.api_key,
            model_name=self.orchestrator_model,
            state_dir=self.state_dir,
            perception_pool_dir=self.perception_pool_dir or (self.state_dir / "perception_pool"),
            update_interval=self.update_interval,
            min_observations=self.min_observations,
            auto_refine_on_failure=self.auto_refine_on_failure,
            max_refinement_attempts=self.max_refinement_attempts,
            solver_backend=self.solver_backend,
            solver_algorithm=self.solver_algorithm,
            solver_timeout=self.solver_timeout,
            task_analyzer_prompts_path=self.task_analyzer_prompts_path,
            genai_log_dir=self.genai_log_dir,
            on_plan_generated=self.on_plan_generated,
        )


@dataclass
class TAMPResult:
    """Result of a complete TAMP execution."""
    success: bool
    task_description: str

    # Planning phase results
    pddl_plan: Optional[List[str]] = None
    planning_time: float = 0.0
    refinement_attempts: int = 0

    # Decomposition phase results
    skill_plans: Dict[str, SkillPlan] = field(default_factory=dict)
    decomposition_time: float = 0.0

    # Execution phase results
    execution_results: List[PrimitiveExecutionResult] = field(default_factory=list)
    execution_time: float = 0.0

    # Error tracking
    error_message: Optional[str] = None
    failed_at_stage: Optional[str] = None  # "planning", "decomposition", "execution"

    def __str__(self) -> str:
        if self.success:
            return (
                f"✓ TAMP Success: {len(self.pddl_plan or [])} actions planned, "
                f"{len(self.skill_plans)} decomposed, "
                f"{len(self.execution_results)} executed "
                f"(plan: {self.planning_time:.1f}s, decomp: {self.decomposition_time:.1f}s, "
                f"exec: {self.execution_time:.1f}s)"
            )
        else:
            return f"✗ TAMP Failed at {self.failed_at_stage}: {self.error_message}"


class TaskAndMotionPlanner:
    """
    Complete Task and Motion Planning system.

    Integrates perception, symbolic planning, skill decomposition, and primitive execution
    into a unified pipeline for robot task execution.
    """

    def __init__(self, config: TAMPConfig):
        """
        Initialize the TAMP system.

        Args:
            config: TAMP configuration
        """
        self.config = config
        self.state = TAMPState.UNINITIALIZED
        self.logger = logging.getLogger("TaskAndMotionPlanner")

        # Create output directories
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        perception_pool_dir = self.config.perception_pool_dir or (self.config.state_dir / "perception_pool")
        perception_pool_dir.mkdir(parents=True, exist_ok=True)
        # Enable GenAI logging if requested
        configure_genai_logging(self.config.genai_log_dir)
        # Initialize components
        orchestrator_config = config.to_orchestrator_config()
        self.orchestrator = TaskOrchestrator(orchestrator_config)

        self.skill_decomposer = SkillDecomposer(
            api_key=config.api_key,
            model_name=config.decomposer_model,
            orchestrator=self.orchestrator,
            primitive_catalog_path=config.primitive_catalog_path,
        )

        self.primitive_executor = PrimitiveExecutor(
            primitives=config.primitives_interface,
            perception_pool_dir=perception_pool_dir,
            logger=self.orchestrator.logger.getChild("PrimitiveExecutor"),
        )

        # Execution tracking
        self.current_task: Optional[str] = None
        self.last_result: Optional[TAMPResult] = None

        print(f"✓ TaskAndMotionPlanner initialized")
        print(f"  • State directory: {self.config.state_dir}")
        print(f"  • Orchestrator model: {config.orchestrator_model}")
        print(f"  • Decomposer model: {config.decomposer_model}")
        print(f"  • Solver backend: {config.solver_backend}")

    async def initialize(self) -> None:
        """Initialize the orchestrator and perception systems."""
        await self.orchestrator.initialize()
        self._set_state(TAMPState.IDLE)
        print("✓ TAMP system ready")

    async def shutdown(self) -> None:
        """Shutdown all systems cleanly."""
        await self.orchestrator.shutdown()
        print("✓ TAMP system shutdown")

    async def perceive_environment(
        self,
        duration: Optional[float] = None,
        min_observations: Optional[int] = None
    ) -> bool:
        """
        Run perception until environment is sufficiently observed.

        Args:
            duration: Max time to perceive (seconds). None = use orchestrator config
            min_observations: Min observations per object. None = use orchestrator config

        Returns:
            True if perception successful, False otherwise
        """
        self._set_state(TAMPState.PERCEIVING)

        print(f"\n{'='*70}")
        print("PERCEIVING ENVIRONMENT")
        print(f"{'='*70}")

        # Start continuous detection
        detection_start = asyncio.get_event_loop().time()
        await self.orchestrator.start_detection()
        detection_init_time = asyncio.get_event_loop().time() - detection_start
        self.logger.info(f"[TIMING] Detection started in {detection_init_time:.3f}s")

        # Wait for sufficient observations or timeout
        start_time = asyncio.get_event_loop().time()
        target_observations = min_observations or self.config.min_observations
        iteration_count = 0

        while True:
            iteration_count += 1
            loop_iter_start = asyncio.get_event_loop().time()

            status = await self.orchestrator.get_status()
            elapsed = asyncio.get_event_loop().time() - start_time

            # Log detailed status every iteration
            self.logger.info(
                f"[TIMING] Perception loop iteration {iteration_count} at {elapsed:.1f}s: "
                f"objects={status.get('num_objects', 0)}, "
                f"ready={self.orchestrator.is_ready_for_planning()}, "
                f"state={status.get('orchestrator_state', 'unknown')}, "
                f"task_decision={self.orchestrator.last_task_decision}"
            )

            # Check if ready
            if self.orchestrator.is_ready_for_planning():
                print(f"✓ Environment sufficiently observed ({status['num_objects']} objects)")
                self.logger.info(f"[TIMING] Perception ready after {elapsed:.1f}s, {iteration_count} iterations")
                await self.orchestrator.pause_detection()
                self._set_state(TAMPState.IDLE)
                return True

            # Check timeout
            if duration and elapsed > duration:
                print(f"⚠ Perception timeout after {duration}s")
                self.logger.warning(
                    f"[TIMING] Perception timeout after {duration}s, {iteration_count} iterations. "
                    f"Objects detected: {status.get('num_objects', 0)}"
                )
                await self.orchestrator.pause_detection()
                self._set_state(TAMPState.IDLE)
                return False

            # Wait and check again
            iter_time = asyncio.get_event_loop().time() - loop_iter_start
            self.logger.debug(f"[TIMING] Iteration {iteration_count} took {iter_time:.3f}s")
            await asyncio.sleep(1.0)

    async def plan_task(
        self,
        task: str,
        use_refinement: bool = True
    ) -> Optional[SolverResult]:
        """
        Generate a symbolic PDDL plan for the task.

        Uses orchestrator's task state monitor to determine readiness for planning.
        Handles empty plans as a refinement trigger.

        Args:
            task: Natural language task description
            use_refinement: Whether to use automatic domain refinement on failures

        Returns:
            SolverResult if planning succeeded, None otherwise
        """
        self._set_state(TAMPState.PLANNING)
        self.current_task = task

        print(f"\n{'='*70}")
        print("TASK PLANNING")
        print(f"{'='*70}")
        print(f"Task: {task}")
        print()

        # Process task request (updates domain with task-specific info)
        await self.orchestrator.process_task_request(task)
        await self.orchestrator.start_detection()
        # Check task state monitor decision
        print("  • Checking task state monitor...")
        decision = await self.orchestrator.monitor.determine_state()
        print(f"    State: {decision.state.value}")
        print(f"    Confidence: {decision.confidence:.2f}")

        if decision.blockers:
            print(f"    Blockers: {', '.join(decision.blockers)}")
        if decision.recommendations:
            print(f"    Recommendations: {', '.join(decision.recommendations)}")

        # Check if we should proceed with planning
        from src.planning.utils.task_types import TaskState
        if decision.state != TaskState.PLAN_AND_EXECUTE:
            print(f"\n⚠ Task state monitor indicates not ready to plan: {decision.state.value}")
            if decision.blockers:
                print(f"   Blockers: {', '.join(decision.blockers)}")
            # Continue anyway for now, but log the warning
            print("   Proceeding with planning attempt...")

        # Solve with or without refinement. Wait for object detection first
        # to avoid generating empty problem files when detection is still warming up.
        if use_refinement and self.config.auto_refine_on_failure:
            result = await self.orchestrator.solve_and_plan_with_refinement(
                wait_for_objects=True
            )
        else:
            result = await self.orchestrator.solve_and_plan(
                wait_for_objects=True
            )

        # Check for successful plan but empty (no actions needed)
        if result.success and result.plan_length == 0:
            print(f"\n⚠ Planning succeeded but returned empty plan (0 actions)")
            print(f"   This usually indicates:")
            print(f"   - Goals already satisfied in initial state")
            print(f"   - Goals not properly set from task analysis")
            print(f"   - Domain/problem mismatch")

            # Treat empty plan as a failure for refinement purposes
            if use_refinement and self.config.auto_refine_on_failure:
                print(f"\n🔧 Triggering targeted representation repair for empty plan...")
                error_msg = "Planning returned empty plan (0 actions). Goals may not be properly set or are already satisfied."
                pddl_dir = self.orchestrator.config.state_dir / "pddl"
                await self.orchestrator.refine_domain_from_failure(
                    error_message=error_msg,
                    pddl_files={
                        "domain_path": str(pddl_dir / f"{self.orchestrator.pddl.domain_name}_domain.pddl"),
                        "problem_path": str(pddl_dir / f"{self.orchestrator.pddl.domain_name}_problem.pddl"),
                    },
                )

                # Retry planning once after refinement
                print(f"\n🔄 Retrying planning after refinement...")
                result = await self.orchestrator.solve_and_plan()

        if result.success and result.plan_length > 0:
            print(f"✓ Generated plan with {result.plan_length} actions")
            self._set_state(TAMPState.IDLE)
        else:
            if result.success and result.plan_length == 0:
                print(f"✗ Planning returned empty plan even after refinement")
            else:
                print(f"✗ Planning failed: {result.error_message}")
            self._set_state(TAMPState.FAILED)

        return result if (result.success and result.plan_length > 0) else None

    async def decompose_action(
        self,
        action: str,
        parameters: Dict[str, Any],
        temperature: float = 0.1
    ) -> Optional[SkillPlan]:
        """
        Decompose a single PDDL action into primitive skills.

        Args:
            action: PDDL action name (e.g., "pick", "place")
            parameters: Action parameters from PDDL plan
            temperature: LLM temperature for decomposition

        Returns:
            SkillPlan if decomposition succeeded, None otherwise
        """
        print(f"\n  Decomposing: {action}({', '.join(f'{k}={v}' for k, v in parameters.items())})")

        try:
            skill_plan = self.skill_decomposer.plan(
                action_name=action,
                parameters=parameters,
                temperature=temperature
            )

            if skill_plan and skill_plan.primitives:
                print(f"    ✓ Generated {len(skill_plan.primitives)} primitives")
                if self.config.on_action_decomposed:
                    self.config.on_action_decomposed(action, skill_plan)
                return skill_plan
            else:
                print(f"    ✗ No primitives generated")
                return None

        except Exception as e:
            print(f"    ✗ Decomposition failed: {e}")
            return None

    async def execute_skill_plan(
        self,
        action_name: str,
        skill_plan: SkillPlan,
        dry_run: bool = False
    ) -> PrimitiveExecutionResult:
        """
        Execute a skill plan (sequence of primitives).

        Args:
            action_name: Name of the action being executed
            skill_plan: Skill plan to execute
            dry_run: If True, validate but don't execute

        Returns:
            Execution result with success status; exceptions propagate on failure.
        """
        print(f"\n  Executing: {action_name} ({'DRY RUN' if dry_run else 'LIVE'})")

        # Get complete world state from orchestrator (matches run_cached_plan.py format)
        # This includes: registry, last_snapshot_id, snapshot_index, robot_state
        world_state = self.orchestrator.get_world_state_snapshot()

        result = self.primitive_executor.execute_plan(
            plan=skill_plan,
            world_state=world_state,
            dry_run=dry_run
        )

        if result.executed:
            print(f"    ✓ Executed {len(skill_plan.primitives)} primitives")
        else:
            print(f"    ✓ Validated {len(skill_plan.primitives)} primitives (dry run)")

        if self.config.on_action_executed:
            self.config.on_action_executed(action_name, result)

        return result

    async def plan_and_execute_task(
        self,
        task: str,
        dry_run: Optional[bool] = None,
        use_refinement: bool = True,
        decompose_temperature: float = 0.1
    ) -> TAMPResult:
        """
        Complete TAMP pipeline: plan → decompose → execute.

        Args:
            task: Natural language task description
            dry_run: If True, validate but don't execute. None = use config default
            use_refinement: Whether to use automatic domain refinement
            decompose_temperature: LLM temperature for skill decomposition

        Returns:
            TAMPResult with complete execution details
        """
        import time

        if dry_run is None:
            dry_run = self.config.dry_run_default

        result = TAMPResult(
            success=False,
            task_description=task
        )

        print(f"\n{'='*70}")
        print(f"TASK AND MOTION PLANNING")
        print(f"{'='*70}")
        print(f"Task: {task}")
        print(f"Mode: {'DRY RUN (validation only)' if dry_run else 'LIVE EXECUTION'}")
        print(f"{'='*70}\n")
        await self.orchestrator.update_task(task)
        await self.orchestrator.start_detection()
        # Phase 1: Task Planning
        planning_start = time.time()
        solver_result = await self.plan_task(task, use_refinement=use_refinement)
        result.planning_time = time.time() - planning_start

        if not solver_result or not solver_result.success:
            result.failed_at_stage = "planning"
            result.error_message = solver_result.error_message if solver_result else "Planning failed"
            return result

        result.pddl_plan = solver_result.plan
        result.refinement_attempts = self.orchestrator.refinement_attempts

        # Phase 2: Skill Decomposition
        # Split plan into segments at 'observe' actions for phased execution
        self._set_state(TAMPState.DECOMPOSING)
        decomp_start = time.time()

        print(f"\n{'='*70}")
        print("SKILL DECOMPOSITION")
        print(f"{'='*70}")

        # Split the plan at observe actions
        plan_segments = []
        current_segment = []
        for action_str in solver_result.plan:
            parts = action_str.strip("()").split()
            action_name = parts[0]

            if action_name.lower() == "observe":
                # Found an observe action - complete current segment
                if current_segment:
                    plan_segments.append(current_segment)
                    current_segment = []
                # Add observe as its own marker segment
                plan_segments.append(["observe"])
            else:
                current_segment.append(action_str)

        # Add final segment if any
        if current_segment:
            plan_segments.append(current_segment)

        print(f"  • Plan split into {len(plan_segments)} segment(s) at observe boundaries")

        # Decompose only the first segment (before first observe)
        skill_plans = {}
        first_segment = plan_segments[0] if plan_segments else []

        for i, action_str in enumerate(first_segment, 1):
            # Parse action string: "action_name obj1 obj2 ..."
            parts = action_str.strip("()").split()
            action_name = parts[0]
            action_params = {f"param{j}": param for j, param in enumerate(parts[1:], 1)}

            print(f"\n[{i}/{len(first_segment)}] {action_str}")

            skill_plan = await self.decompose_action(
                action=action_name,
                parameters=action_params,
                temperature=decompose_temperature
            )

            if not skill_plan:
                result.failed_at_stage = "decomposition"
                result.error_message = f"Failed to decompose action: {action_str}"
                result.decomposition_time = time.time() - decomp_start
                self._set_state(TAMPState.FAILED)
                return result

            skill_plans[action_str] = skill_plan

        result.skill_plans = skill_plans
        result.decomposition_time = time.time() - decomp_start
        print(f"\n✓ Decomposed {len(skill_plans)} actions in first segment")

        # Persist decomposed skill plans to disk for inspection/debugging
        try:
            decomp_dir = Path(self.config.state_dir) / "decomposed_plans"
            decomp_dir.mkdir(parents=True, exist_ok=True)

            all_plans = {}
            for idx, (action_str, skill_plan) in enumerate(skill_plans.items(), 1):
                # Create a filesystem-safe name prefix
                safe_name = action_str.strip("()").replace(" ", "_").replace("/", "_")
                file_name = f"{idx:02d}_{safe_name}.json"
                plan_path = decomp_dir / file_name
                try:
                    plan_path.write_text(json.dumps(skill_plan.to_dict(), indent=2))
                    print(f"  • Wrote decomposed plan: {plan_path}")
                except Exception as e:
                    print(f"  ⚠ Failed to write decomposed plan {plan_path}: {e}")

                all_plans[action_str] = skill_plan.to_dict()

            # Write aggregated file
            agg_path = decomp_dir / "all_skill_plans.json"
            try:
                agg_path.write_text(json.dumps(all_plans, indent=2))
                print(f"  • Wrote aggregated decomposed plans: {agg_path}")
            except Exception:
                pass
        except Exception:
            pass

        # Phase 3: Segmented Execution with Observation Updates
        self._set_state(TAMPState.EXECUTING)
        exec_start = time.time()

        print(f"\n{'='*70}")
        print(f"PRIMITIVE EXECUTION {'(DRY RUN)' if dry_run else ''}")
        print(f"{'='*70}")

        execution_results = []
        all_skill_plans = {}

        # Process each segment
        for segment_idx, segment in enumerate(plan_segments):
            # Check if this is an observe marker
            if segment == ["observe"]:
                print(f"\n{'='*70}")
                print(f"OBSERVE ACTION - TRIGGERING STATE UPDATE")
                print(f"{'='*70}")

                # Trigger object tracking update
                print("  • Moving to home position for observation...")
                if self.config.primitives_interface and not dry_run:
                    try:
                        # Retract to home position
                        retract_method = getattr(self.config.primitives_interface, "retract_gripper", None)
                        if callable(retract_method):
                            retract_method()
                            print("    ✓ Moved to home position")
                    except Exception as e:
                        print(f"    ⚠ Could not retract to home: {e}")

                print("  • Running object detection to update world state...")
                # Run a perception update
                perception_start = asyncio.get_event_loop().time()
                await self.orchestrator.start_detection()
                await asyncio.sleep(5.0)  # Allow time for multiple detection cycles
                await self.orchestrator.pause_detection()
                perception_time = asyncio.get_event_loop().time() - perception_start

                print(f"    ✓ Observation complete ({perception_time:.1f}s)")

                # Get updated status
                status = await self.orchestrator.get_status()
                print(f"    • Detected objects: {status.get('registry', {}).get('num_objects', 0)}")

                # Decompose remaining segments with updated world state
                if segment_idx + 1 < len(plan_segments):
                    remaining_segments = plan_segments[segment_idx + 1:]
                    print(f"\n  • Decomposing {sum(len(s) for s in remaining_segments if s != ['observe'])} remaining actions with updated state...")

                    decomp_start_remaining = time.time()
                    for remaining_segment in remaining_segments:
                        if remaining_segment == ["observe"]:
                            continue  # Skip subsequent observe markers for now

                        for action_str in remaining_segment:
                            parts = action_str.strip("()").split()
                            action_name = parts[0]
                            action_params = {f"param{j}": param for j, param in enumerate(parts[1:], 1)}

                            print(f"    → Decomposing {action_str}")

                            skill_plan = await self.decompose_action(
                                action=action_name,
                                parameters=action_params,
                                temperature=decompose_temperature
                            )

                            if not skill_plan:
                                result.failed_at_stage = "decomposition"
                                result.error_message = f"Failed to decompose action after observe: {action_str}"
                                result.decomposition_time += time.time() - decomp_start_remaining
                                result.execution_time = time.time() - exec_start
                                self._set_state(TAMPState.FAILED)
                                return result

                            skill_plans[action_str] = skill_plan

                    result.decomposition_time += time.time() - decomp_start_remaining
                    print(f"    ✓ Decomposed remaining actions")

                continue  # Move to next segment

            # Execute current segment's actions
            if segment_idx == 0:
                segment_skill_plans = skill_plans
            else:
                # For segments after observe, we've already decomposed them
                segment_skill_plans = {action: skill_plans[action] for action in segment if action in skill_plans}

            for i, (action_str, skill_plan) in enumerate(segment_skill_plans.items(), 1):
                print(f"\n[Segment {segment_idx + 1}] [{i}/{len(segment_skill_plans)}] {action_str}")

                exec_result = await self.execute_skill_plan(
                    action_name=action_str,
                    skill_plan=skill_plan,
                    dry_run=dry_run
                )

                execution_results.append(exec_result)
                all_skill_plans[action_str] = skill_plan

                if not exec_result.executed and not dry_run:
                    result.failed_at_stage = "execution"
                    result.error_message = f"Failed to execute action: {action_str}"
                    result.execution_results = execution_results
                    result.execution_time = time.time() - exec_start
                    result.skill_plans = all_skill_plans
                    self._set_state(TAMPState.FAILED)
                    return result

        result.execution_results = execution_results
        result.skill_plans = all_skill_plans
        result.execution_time = time.time() - exec_start

        # Success!
        result.success = True
        self._set_state(TAMPState.COMPLETED)
        self.last_result = result

        print(f"\n{'='*70}")
        print(f"{'✓ TAMP COMPLETED' if not dry_run else '✓ TAMP VALIDATION COMPLETED'}")
        print(f"{'='*70}")
        print(f"Planning: {result.planning_time:.1f}s ({len(result.pddl_plan)} actions)")
        print(f"Decomposition: {result.decomposition_time:.1f}s ({len(result.skill_plans)} skill plans)")
        print(f"Execution: {result.execution_time:.1f}s ({len(result.execution_results)} actions)")
        print(f"Total: {result.planning_time + result.decomposition_time + result.execution_time:.1f}s")
        print(f"{'='*70}\n")

        return result

    def _set_state(self, state: TAMPState) -> None:
        """Update TAMP state and trigger callback if configured."""
        self.state = state
        if self.config.on_state_change:
            self.config.on_state_change(state)

    async def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        orch_status = await self.orchestrator.get_status()

        return {
            "tamp_state": self.state.value,
            "orchestrator_state": orch_status["state"],
            "current_task": self.current_task,
            "num_objects": orch_status["num_objects"],
            "last_result": str(self.last_result) if self.last_result else None,
        }
