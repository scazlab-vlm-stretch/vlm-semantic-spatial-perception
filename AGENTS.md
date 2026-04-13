# Agent Instructions

This repository implements a VLM-based Task and Motion Planning (TAMP) system that translates natural language commands into executable robot actions to complete tasks that are potneitally long horizon and complex.

## Repository Layout

This repository studies VLM-driven task and motion planning, with a focus on
turning open-ended language plus RGB-D perception into executable symbolic and
geometric robot behavior. The current planning stack is organized around a
staged representation builder so failures can be localized and repaired by
layer instead of regenerating everything at once.

Key paths:
- `src/planning/`: staged task analysis, representation building, PDDL serialization, solver orchestration, and task-state monitoring.
- `src/perception/`: continuous object tracking, predicate extraction, and registry/state management.
- `src/primitives/` and `src/task_motion_planner.py`: symbolic-to-primitive decomposition and execution.
- `scripts/`: capture, replay, and benchmarking entry points. `scripts/benchmark_saved_world.py` is the main no-robot integration replay.
- `config/`: prompt templates and runtime configuration.
- `tests/`: focused unit tests plus cached-world and perception fixtures.
- `agents/`: task-specific implementation notes such as migration plans and behavior journals.

## Style Guidelines
- Avoid unecessary scaffolding code and helpers. In general, only use more layers of abstraction if a block is reused and/or logically separated from the rest of the code.
- For one-time use scripts (e.g. plotting, debugging, etc.), make them as simple as possible. 
- Avoid rebuilding wheels. E.g. if a logic can be reused, create an abstraction instread of copying the code. When running experiments and tests, *never* rebuild wheels of already implemented functionality, and instead always choose to import and reuse previous code.
- Unless the user specifically asks, do not maintain backwards compatibility. Always remove the deprecated code and update affected parts of the codebase/tests. 
- Comment functions and classes in pytorch format, with a short description and usage examples. Some helpers or launcher functions may not need such detailed comments, and can instead just have a short description.
- Inline comments should be liberally made, but not when the code is self-explanatory. 

## Development Guidelines

1. **Use `uv` for environment management.**
   - `uv add <package>` instead of `pip install`.
   - `uv run script.py` instead of plain `python`.
   - `uvx` for interactive tools rather than `pipx`.
   - **Do not** use `pip` overrides like `uv pip install <package>` as uv
     does not keep track of the package versions.
   - Reference https://docs.astral.sh/uv/llms.txt for uv documentation.

2. **Consult `agents/` to determine if there are relevant instructions for the task.**
   - After performing the task, update the relevant instructions.
   - If the task is not in `agents/`, create a new instructions file in the appropriate subfolder.
   - If the subfolder does not exist, create it.
   - Aim for conciseness and avoid redundancy in tracking. Documentation should add information that is not obvious from the code (e.g. how it is used, the motivation, notations & mathematical formulae), and as such should not repeat information already present in the code.

3. **Notebook format** – Notebooks are marimo scripts; Jupyter is not used here. 
   - Reference https://docs.marimo.io/llms.txt for marimo documentation.
   - Marimo supports Agent Client Protocol (ACP), see https://docs.marimo.io/guides/editor_features/agents/ for more details and https://docs.marimo.io/CLAUDE.md for agent-facing instructions.
   - `check` can be used to verify the correctness of the notebook: `uvx marimo check my_notebook.py`.

4. **Configuration** – Scripts rely on **gin** for nearly all parameters. Only a
   few entry points use `argparse` for top-level flags and command line arguments. 
   Helpers should be defined and used in a consistent manner in `src/utils.py`:
   - Reference https://raw.githubusercontent.com/google/gin-config/refs/heads/master/docs/index.md 
     for gin documentation.

5. **Testing** – All pytest configuration is consolidated in `pyproject.toml`. 
   - `uv run -m pytest`

## Git Worktree Setup
Use worktree scripts for parallel development, e.g. for running experiments in parallel:

```bash
# Create worktree with a reusable script (as sibling directory)
./scripts/setup_worktree.sh my-experiment

# Remove when done
./scripts/cleanup_worktree.sh my-experiment
```

**Shared via symlinks:** A certain number of files/directories are ignored by git but should be shared across worktrees. E.g. `checkpoints/`, `results/`, `logs/`. Make sure the list of such files/directories is maintained and updated across documentation/worktree setup scripts.
