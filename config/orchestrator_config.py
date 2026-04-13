"""
Orchestrator Configuration

Configuration dataclass for the Task Orchestrator.
"""

from pathlib import Path
from typing import Optional, Callable, Any, Literal
from dataclasses import dataclass, field

# Import types needed for type hints
# These are imported at runtime only for type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.planning.task_orchestrator import OrchestratorState
    from src.planning.task_state_monitor import TaskStateDecision


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    # API Configuration
    api_key: str
    model_name: str = "gemini-robotics-er-1.5-preview"

    # Camera Configuration
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    enable_depth: bool = True
    use_sim_camera: bool = False  # Skip RealSense init; frame provider will be injected manually

    # Detection Configuration
    update_interval: float = 2.0  # Seconds between detections
    min_observations: int = 3  # Minimum objects before planning
    fast_mode: bool = False  # Skip interaction points for speed

    # Persistence Configuration
    state_dir: Path = field(default_factory=lambda: Path("outputs/orchestrator_state"))
    auto_save: bool = True  # Auto-save state on updates
    auto_save_on_detection: bool = True  # Save after each detection update
    auto_save_on_state_change: bool = True  # Save on state changes

    # Snapshot / Perception Pool (optional, non-breaking)
    enable_snapshots: bool = True  # Enabled by default
    snapshot_every_n_detections: int = 1  # Save every detection update by default
    # When None, will default to state_dir / "perception_pool" at runtime
    perception_pool_dir: Optional[Path] = None
    max_snapshot_count: Optional[int] = 200  # Rotate oldest if exceeded
    # Stream annotated debug frames as detection runs; None disables
    debug_frames_dir: Optional[Path] = None
    depth_encoding: Literal["npz"] = "npz"  # Keep simple for now
    # Optional LLM client (src.llm_interface.LLMClient).
    # When set, all sub-components (ObjectTracker, LLMTaskAnalyzer, LayeredDomainGenerator)
    # use this client instead of creating their own genai.Client.
    # api_key / model_name are still accepted for backward-compat but ignored when
    # llm_client is provided.
    llm_client: Optional[Any] = None  # LLMClient instance

    # Optional robot provider (duck-typed):
    # - get_robot_joint_state() -> List[float] | np.ndarray
    # - get_robot_tcp_pose() -> Tuple[List[float], List[float]] (position, quaternion_xyzw)
    # - get_camera_transform() -> Tuple[List[float], AnyRotation] (position, rotation)
    robot: Optional[Any] = None

    # Task Configuration
    exploration_timeout: float = 60.0  # Max time for exploration before forcing decision

    # PDDL Solver Configuration
    solver_backend: Optional[str] = None  # "auto", "pyperplan", "fast-downward-docker", "fast-downward-apptainer"
    solver_algorithm: str = "lama-first"  # Search algorithm to use
    solver_timeout: float = 120.0  # Timeout in seconds for solver
    solver_verbose: bool = False  # Print solver output
    auto_solve_when_ready: bool = False  # Automatically solve when ready for planning
    max_refinement_attempts: int = 5  # Max attempts to refine domain after planning failures
    auto_refine_on_failure: bool = True  # Automatically refine domain when planning fails

    # Prompt Configuration
    task_analyzer_prompts_path: Optional[Path] = None  # Override for LLMTaskAnalyzer prompt file
    genai_log_dir: Optional[Path] = None  # Optional GenAI request/response log directory

    # Layered Domain Generation (gated/guardrailed pipeline)
    use_layered_generation: bool = False  # Enable 5-layer validated domain generation
    dkb_dir: Optional[Path] = None  # Domain Knowledge Base directory (default: outputs/dkb)

    # Callbacks
    on_state_change: Optional[Callable[["OrchestratorState", "OrchestratorState"], None]] = None
    on_detection_update: Optional[Callable[[int], None]] = None
    on_task_state_change: Optional[Callable[["TaskStateDecision"], None]] = None
    on_save_state: Optional[Callable[[Path], None]] = None  # Called after successful save
    on_plan_generated: Optional[Callable[[Any], None]] = None  # Called after plan is generated (receives SolverResult)
