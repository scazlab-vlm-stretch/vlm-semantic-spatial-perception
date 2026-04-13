# VLM-Based Task and Motion Planning System

## Overview

This system implements an end-to-end **Task and Motion Planning (TAMP)** pipeline that translates natural language commands into executable robot actions. It combines vision-language models (VLMs), symbolic planning (PDDL), and motion planning to achieve autonomous task execution.

**Key Innovation**: The system uses Gemini's vision and reasoning capabilities to bridge the gap between high-level task descriptions and low-level robot control, enabling robots to understand and execute complex manipulation tasks through natural language.

---

## Quick Start

### Prerequisites

- `uv` for package management
- Gemini API key
- RealSense camera for perception
- xArm robot for physical execution

### Installation

```bash
# Install dependencies
uv sync

# Set API key
export GEMINI_API_KEY="your-api-key-here"
```

### Running the System

#### Interactive Mode (Recommended for First Use)
```bash
uv run examples/tamp_demo.py
```

This launches an interactive terminal where you can:
- Issue natural language commands
- Monitor perception and planning
- Execute tasks step-by-step

#### Single Task Execution
```bash
# Dry run (validation only, no robot execution)
uv run examples/tamp_demo.py --task "pick up the red block" --dry-run

# Live execution (requires robot connection)
uv run examples/tamp_demo.py --task "stack the blue block on the red block" --live
```

---

## System Architecture

Class-level flow in task-execution call order:

```text
+--------------------+    +-----------------------------------------------+
| examples/tamp_demo | -> | TaskAndMotionPlanner                          |
+--------------------+    | (initialize -> perceive -> plan -> execute)   |
                          +-------------------------+---------------------+
                                                    |
                                                    v
                          +-------------------------+---------------------+
                          | TaskOrchestrator                              |
                          +----------------+----------------+-------------+
                                           |                |
                               perception  |                |  planning
                                  loop     v                v   loop
                          +----------------------+   +--------------------+
                          | RealSenseCamera      |   | ContinuousObject-  |
                          +----------+-----------+-->| Tracker            |
                                     |  frames       +----------+---------+
                                     |                           |
                                     |                     detections
                                     |                           v
                          +--------------------------------------+--------+
                          | LLMTaskAnalyzer + PDDLDomainMaintainer        |
                          | + PDDLRepresentation + PDDLSolver             |
                          +------------------------------+----------------+
                                                         |
                                              symbolic action sequence
                                                         v
                          +------------------------------+----------------+
                          | SkillDecomposer -> PrimitiveExecutor          |
                          | -> CuRoboMotionPlanner                        |
                          +-----------------------------------------------+
```

---

## Entry Point and Execution Flow

**Main Entry Point**: [examples/tamp_demo.py](examples/tamp_demo.py)


**Core TAMP Planner**: [TaskAndMotionPlanner](src/task_motion_planner.py)

The main planner class that coordinates all components:
- `initialize()` - Sets up orchestrator, decomposer, executor
- `perceive_environment()` - Runs continuous object detection
- `plan_task()` - Generates PDDL plan with refinement
- `decompose_action()` - Converts symbolic actions to primitives
- `execute_skill_plan()` - Executes primitives on robot
- `plan_and_execute_task()` - Complete pipeline orchestration

---

## System Components

### 1. Task Analysis

**Purpose**: Build a staged symbolic representation that separates intent from grounding.

**Implementation**:
- **Main Class**: [TaskOrchestrator](src/planning/task_orchestrator.py)
- **Key Method**: `process_task_request()` - Runs the staged representation builder
- **Domain Maintainer**: [PDDLDomainMaintainer](src/planning/pddl_domain_maintainer.py) - Builds and validates the four planning layers
- **Configuration**: [llm_task_analyzer_prompts.yaml](config/llm_task_analyzer_prompts.yaml)

Staged build order:
- task language -> abstract goal
- abstract goal -> minimal predicate inventory
- predicates + goal -> action schemas
- predicates + actions + observed world -> grounding summary

Failure handling works in reverse abstraction order:
- repair actions first
- repair predicates next
- repair the goal last

**How it works**:
- Takes natural language input (e.g., "pick up the red block")
- Uses Gemini to extract:
  - Goal objects (what objects are involved)
  - Required predicates (spatial relationships like "on", "clear", "graspable")
  - Necessary actions (pick, place, stack, etc.)
  - Task complexity and estimated steps

**Output**: A structured task analysis that guides the rest of the pipeline

**Example**:
```
Input: "stack the red block on the blue block"
Output:
  - Goal objects: [red_block, blue_block]
  - Predicates: [on, clear, graspable, holding, empty]
  - Actions: [pick, place]
  - Complexity: medium
```

---

### 2. Environment Perception

**Purpose**: Build a 3D understanding of the workspace

**Implementation**:
- **Main Orchestration**: [TaskAndMotionPlanner.perceive_environment()](src/task_motion_planner.py)
- **Detection Manager**: [TaskOrchestrator.start_detection()](src/planning/task_orchestrator.py)
- **Object Tracker**: [ContinuousObjectTracker](src/perception/object_tracker.py)
- **Snapshot Management**: [TaskOrchestrator.save_snapshot()](src/planning/task_orchestrator.py)
- **Camera Interface**: [RealSenseCamera](src/camera/realsense_camera.py)
- **Utilities**: [snapshot_utils.py](src/planning/utils/snapshot_utils.py)

**Components**:

#### a. Continuous Object Tracker
- Captures RGB-D images from RealSense camera
- Runs Gemini Vision API to detect objects
- Tracks objects across frames for temporal consistency

#### b. Task-Grounded Detection
- Uses task context to focus on relevant objects
- Detects affordances (what actions can be performed on each object)
- Identifies interaction points (specific locations for manipulation)

**What gets detected**:
- **Object type**: "block", "cup", "table", "gripper"
- **Affordances**: ["graspable", "stackable", "containable"]
- **Interaction points**:
  - 2D pixel coordinates in image
  - 3D positions in camera frame
  - Alternative grasp locations
- **Spatial properties**: position, bounding box, orientation

**Snapshot System**:
- Every detection cycle saves a snapshot containing:
  - RGB image (`color.png`)
  - Depth map (`depth.npz`)
  - Camera intrinsics (`intrinsics.json`)
  - Object detections (`detections.json`)
  - Robot state at capture time (`robot_state.json`)
- Snapshots enable reproducible execution and debugging

---

### 3. PDDL Symbolic Planning

**Purpose**: Generate a high-level action sequence to achieve the task goal

**Implementation**:
- **Main Orchestration**: [TaskAndMotionPlanner.plan_task()](src/task_motion_planner.py)
- **PDDL Generation**: [TaskOrchestrator.generate_pddl_files()](src/planning/task_orchestrator.py)
- **Solver Interface**: [TaskOrchestrator.solve_and_plan()](src/planning/task_orchestrator.py)
- **Domain Refinement**: [TaskOrchestrator.refine_domain_from_failure()](src/planning/task_orchestrator.py)
- **PDDL Solver**: [PDDLSolver](src/planning/pddl_solver.py)
- **Domain Representation**: [PDDLRepresentation](src/planning/pddl_representation.py)

**Process**:

#### a. Domain Generation
The system automatically generates a PDDL domain from observations:

```pddl
(:types object block surface manipulator)

(:predicates
  (on ?obj - object ?surface - surface)
  (clear ?obj - object)
  (graspable ?obj - object)
  (holding ?obj - object)
  (empty ?gripper - manipulator)
)

(:action pick
  :parameters (?gripper - manipulator ?obj - object)
  :precondition (and (clear ?obj) (graspable ?obj) (empty ?gripper))
  :effect (and (holding ?obj) (not (empty ?gripper)))
)
```

#### b. Problem Definition
Current state and goals are automatically populated:

```pddl
(:init
  (on red_block table)
  (on blue_block table)
  (clear red_block)
  (clear blue_block)
  (empty gripper)
)

(:goal
  (and (on red_block blue_block))
)
```

#### c. Planning with Refinement
- Uses Pyperplan or Fast Downward solver
- If planning fails, uses LLM to refine the domain
- Iterates until a valid plan is found or max attempts reached

**Output**: Sequence of symbolic actions
```
1. (pick gripper red_block)
2. (place gripper red_block blue_block)
```

---

### 4. Skill Decomposition

**Purpose**: Convert each symbolic action into executable robot primitives

**Implementation**:
- **Main Orchestration**: [TaskAndMotionPlanner.decompose_action()](src/task_motion_planner.py)
- **Decomposer**: [SkillDecomposer](src/primitives/skill_decomposer.py)
- **Key Methods**:
  - `SkillDecomposer.plan()` - Main decomposition entry point
  - `SkillDecomposer._build_prompt()` - Constructs LLM prompt with context
  - `SkillDecomposer._call_llm()` - Executes Gemini API call
  - `SkillDecomposer._parse_plan()` - Parses JSON response
- **Data Types**: [SkillPlan, PrimitiveCall](src/primitives/skill_plan_types.py)
- **Configuration**: [skill_decomposer_prompts.yaml](config/skill_decomposer_prompts.yaml)
- **Primitive Catalog**: [primitive_descriptions.md](config/primitive_descriptions.md)

**How it works**:

For each symbolic action (e.g., `pick gripper red_block`):

1. **Gather Context**:
   - Load relevant snapshot with object detections
   - Get interaction points for target object
   - Retrieve robot state and camera pose

2. **Build LLM Prompt** with:
   - Primitive catalog (available low-level commands)
   - Object information (position, affordances, interaction points)
   - Scene image for visual grounding
   - Current robot configuration

3. **Generate Primitive Sequence**:
   Gemini outputs a JSON plan like:
   ```json
   {
     "primitives": [
       {
         "name": "move_to_pre_grasp",
         "parameters": {
           "target_pixel_yx": [0.52, 0.61],
           "depth_offset_m": -0.05,
           "approach_normal": [0.0, 0.0, 1.0]
         },
         "references": {
           "object_id": "red_block",
           "interaction_point": "grasp_top"
         }
       },
       {
         "name": "open_gripper",
         "parameters": {"width_m": 0.08}
       },
       {
         "name": "grasp",
         "parameters": {
           "target_pixel_yx": [0.52, 0.61],
           "grasp_force": 20.0
         }
       }
     ]
   }
   ```

4. **Validation**: Check all primitives exist and parameters are valid

**Why this works**: The VLM can reason about spatial relationships in the image and select appropriate interaction points based on visual information.

---

### 5. Primitive Execution

**Purpose**: Execute robot primitives with metric translation

**Implementation**:
- **Main Orchestration**: [TaskAndMotionPlanner.execute_skill_plan()](src/task_motion_planner.py)
- **Executor**: [PrimitiveExecutor](src/primitives/primitive_executor.py)
- **Key Methods**:
  - `PrimitiveExecutor.execute_plan()` - Main execution entry point
  - `PrimitiveExecutor.prepare_plan()` - Translates and validates primitives
  - `PrimitiveExecutor._translate_helper_parameters()` - Back-projects pixels to 3D
  - `PrimitiveExecutor._back_project()` - Converts 2D pixel → 3D camera frame
- **Coordinate Utilities**: [compute_3d_position()](src/perception/utils/coordinates.py)
- **Snapshot Loading**: [load_snapshot_artifacts()](src/planning/utils/snapshot_utils.py)

**Process**:

#### a. Coordinate Translation
Primitives reference image coordinates, which must be converted to robot coordinates:

1. **Back-projection**: Convert 2D pixel → 3D position in camera frame
   - Use depth map to get distance
   - Apply camera intrinsics to compute 3D point

2. **Frame transformation**: Convert camera frame → robot base frame
   - Get camera pose from robot joints at snapshot time
   - Apply rotation and translation

**Example**:
```
LLM Output: target_pixel_yx=[0.52, 0.61] (normalized image coordinates)
           ↓
Back-projection: [0.15, -0.03, 0.42] meters (camera frame)
           ↓
Frame transform: [0.45, 0.12, 0.30] meters (robot base frame)
           ↓
Robot Command: move_to(target_position=[0.45, 0.12, 0.30])
```

#### b. Motion Planning & Execution
- Each primitive calls CuRobo motion planner
- Generates collision-free trajectories
- Executes on robot hardware (or validates in dry-run mode)

**Available Primitives**:
- `move_to_pre_grasp`: Position above target with approach direction
- `open_gripper`: Open gripper to specified width
- `grasp`: Close gripper with force control
- `move_to`: Move end-effector to position
- `twist`: Rotate about axis
- `release`: Open gripper to drop object

---

## Output Structure

Each execution creates a timestamped run directory:

```
outputs/tamp_demo/run_20250110_143022/
├── run_20250110_143022.log              # Complete execution log
├── run_20250110_143022_timing.log       # Performance timing data
├── run_metadata.json                    # Run configuration
│
├── genai_logs/                          # LLM request/response logs
│   ├── 001_task_analysis/
│   │   ├── metadata.json
│   │   ├── prompt.txt
│   │   ├── response.json
│   │   └── input_image.png
│   ├── 002_perception_cycle/
│   └── ...
│
├── pddl/                                # Generated PDDL files
│   ├── task_execution_domain.pddl
│   └── task_execution_problem.pddl
│
├── decomposed_plans/                    # Skill decomposition outputs
│   ├── 01_pick_gripper_red_block.json
│   ├── 02_place_gripper_red_block_blue_block.json
│   └── all_skill_plans.json
│
├── perception_pool/                     # Perception snapshots
│   ├── index.json                       # Snapshot catalog
│   └── snapshots/
│       └── 20250110_143045_123-abc456/
│           ├── color.png                # RGB image
│           ├── depth.npz                # Depth map
│           ├── intrinsics.json          # Camera parameters
│           ├── detections.json          # Object detections
│           ├── robot_state.json         # Robot configuration
│           └── manifest.json            # Snapshot metadata
│
├── task_001/                            # First task results
│   ├── task_metadata.json               # Task info, success, timing
│   ├── decompositions.json              # Skill plans per action
│   ├── execution_results.json           # Execution outcomes
│   └── pddl -> ../pddl                  # Symlink to PDDL files
│
└── task_002/                            # Second task results
    └── ...
```

This structure enables:
- **Debugging**: Full traces of LLM calls and intermediate results
- **Reproducibility**: Snapshots capture exact world state
- **Analysis**: Timing logs show performance bottlenecks
- **Reuse**: Perception data can be loaded for multiple planning attempts

---

## Key Features

### 1. Task-Grounded Perception
Instead of detecting all objects in a scene, the system focuses on objects relevant to the task. This improves detection quality and reduces processing time.

### 2. Automatic Domain Refinement
If PDDL planning fails (e.g., due to incorrect predicates or missing actions), the system uses Gemini to analyze the error and fix the domain automatically. This eliminates manual PDDL engineering.

### 3. Visual Grounding
Primitives reference specific pixel locations in images rather than abstract 3D coordinates. This allows the VLM to use visual reasoning to select interaction points (e.g., "grasp the handle" vs "grasp the side").

### 4. Snapshot-Based Execution
All primitives are grounded to specific perception snapshots. This ensures:
- Temporal consistency (robot moves to where object *was* in the snapshot)
- Reproducibility (can replay with same world state)
- Debugging capability (can inspect what the VLM saw)

### 5. Dry-Run Validation
The system can validate complete execution pipelines without touching robot hardware, enabling rapid development and testing.

---

## Configuration

### Task Analyzer Prompts
**File**: `config/llm_task_analyzer_prompts.yaml`

Controls how tasks are analyzed:
- Prompt templates for task decomposition
- Predicate extraction guidelines
- Action library definitions

### Skill Decomposer Prompts
**File**: `config/skill_decomposer_prompts.yaml`

Defines how actions are decomposed:
- Prompt template with placeholders
- JSON response schema
- Interaction point selection criteria

### Primitive Catalog
**File**: `config/primitive_descriptions.md`

Documents available robot primitives:
- Function signatures
- Parameter descriptions
- Usage examples

### System Settings
**File**: `config/orchestrator_config.py`

Configure system behavior:
- Perception settings (update interval, min observations)
- Planning settings (solver, timeout, refinement attempts)
- Camera parameters
- Logging levels

---

## Advanced Usage

### Custom Prompts

Use custom task analyzer prompts for domain-specific tasks:

```bash
python examples/tamp_demo.py \
    --task "make a sandwich" \
    --task-analyzer-prompts config/kitchen_prompts.yaml
```

### Perception-Only Mode

Run just the perception and snapshot system:

```bash
python examples/orchestrator_demo.py
```

This provides a TUI for:
- Viewing live detection results
- Manually triggering snapshots
- Inspecting object registries

### Plan Replay

Execute a previously generated plan without re-planning:

```bash
python scripts/run_cached_plan.py \
    --plan outputs/tamp_demo/run_XXX/decomposed_plans/all_skill_plans.json \
    --world outputs/tamp_demo/run_XXX/perception_pool
```

### GenAI Log Inspection

Browse all LLM request/response traces:

```bash
python examples/genai_viewer.py outputs/tamp_demo/run_XXX/genai_logs
```

Navigate with arrow keys to see:
- Full prompts sent to Gemini
- Complete responses
- Input images
- Metadata (tokens, latency, etc.)

### Using the Observe Action

The `observe` action allows you to pause execution, update the world state through object tracking, and then resume with newly decomposed actions that take into account state changes from previous actions.

**Use Case**: When subsequent actions depend on state changes from previous actions (e.g., after placing a block, the scene changes and you need fresh observations before picking the next block).

**Example**:
```pddl
1. pick(block1)
2. place(block1, block2)
3. observe
4. pick(block3)
5. place(block3, block2)
```

**How it works**:
1. Actions 1-2 are decomposed and executed based on the initial world state
2. When `observe` is encountered:
   - The robot moves to home position
   - Object tracking runs for ~5 seconds to capture updated scene
   - World state is refreshed with new object positions and configurations
3. Actions 4-5 are then decomposed using the updated world state
4. Execution continues with the newly decomposed actions

**When to use**:
- Multi-step tasks where the environment changes significantly between actions
- Tasks requiring accurate perception after object manipulation
- Any scenario where initial observations become stale after manipulation

**Configuration**:
The observe action is handled automatically by the TAMP system - simply include it as a symbolic action in your PDDL domain. The system will detect it during execution and trigger the observation update workflow.

Execution note: [TaskAndMotionPlanner.plan_and_execute_task()](src/task_motion_planner.py) executes plans in segments split at `observe` actions, runs a perception refresh, then re-decomposes remaining actions against the updated world state.


## Code Reference Summary

### Core System Files

| Component | File | Description |
|-----------|------|-------------|
| **Entry Point** | [tamp_demo.py](examples/tamp_demo.py) | Main demo script with CLI interface |
| **TAMP Planner** | [task_motion_planner.py](src/task_motion_planner.py) | Main orchestrator coordinating all components |
| **Task Orchestrator** | [task_orchestrator.py](src/planning/task_orchestrator.py) | Manages perception, PDDL generation, planning |
| **Skill Decomposer** | [skill_decomposer.py](src/primitives/skill_decomposer.py) | Translates symbolic actions to primitives using VLM |
| **Primitive Executor** | [primitive_executor.py](src/primitives/primitive_executor.py) | Executes primitives with coordinate transformation |
| **PDDL Solver** | [pddl_solver.py](src/planning/pddl_solver.py) | Unified interface to PDDL planners |
| **Domain Representation** | [pddl_representation.py](src/planning/pddl_representation.py) | In-memory/domain-problem PDDL representation and file generation |
| **Domain Maintainer** | [pddl_domain_maintainer.py](src/planning/pddl_domain_maintainer.py) | LLM-based domain refinement |
| **Object Tracker** | [object_tracker.py](src/perception/object_tracker.py) | Object tracking and continuous detection with VLM |
| **Camera** | [realsense_camera.py](src/camera/realsense_camera.py) | RealSense RGB-D camera interface |

### Key Data Types

| Type | File | Purpose |
|------|------|---------|
| **TAMPConfig** | [task_motion_planner.py](src/task_motion_planner.py) | Configuration for entire TAMP system |
| **SkillPlan** | [skill_plan_types.py](src/primitives/skill_plan_types.py) | Sequence of primitives for an action |
| **PrimitiveCall** | [skill_plan_types.py](src/primitives/skill_plan_types.py) | Individual primitive with parameters |
| **SolverResult** | [pddl_solver.py](src/planning/pddl_solver.py) | PDDL planning result |
| **PrimitiveExecutionResult** | [primitive_executor.py](src/primitives/primitive_executor.py) | Execution outcome with warnings/errors |

### Utility Modules

| Module | File | Purpose |
|--------|------|---------|
| **Snapshot Utils** | [snapshot_utils.py](src/planning/utils/snapshot_utils.py) | Load/save perception snapshots |
| **Coordinate Utils** | [coordinates.py](src/perception/utils/coordinates.py) | Back-projection and coordinate transforms |
| **GenAI Logging** | [genai_logging.py](src/utils/genai_logging.py) | Log all LLM requests/responses |


## Contributors

### Liam Hoffmeister

Liam created the foundational TAMP system architecture, implementing the core object detection and tracking pipeline, PDDL planning integration with automatic domain generation, and the TaskOrchestrator that coordinates the entire system. His work spans from initial repository setup through recent refinements in prompt engineering, performance optimizations, and the addition of new manipulation capabilities like the twist action.

### Enyan Zhang

Enyan built essential infrastructure and developer tools that make the system maintainable and debuggable, including the comprehensive GenAI logging system with visual inspection utilities, the snapshot persistence architecture for reproducible execution, and a major refactor of the skill decomposition pipeline to use snapshot-grounded coordinate frames. His contributions include environment setup (UV package management), prompt externalization to YAML configs, CuRobo integration, numerous debugging utilities, and extensive documentation improvements.

### TJ Vitchutripop (tjvitchutripop) - 21 commits

TJ focused on the primitive execution layer and robot-perception synchronization, designing the simplified primitive system that translates high-level actions into robot commands and solving critical bugs around robot state synchronization with perception snapshots and RealSense depth data handling. His recent work includes ongoing improvements to the primitive executor and prompt refinements.
