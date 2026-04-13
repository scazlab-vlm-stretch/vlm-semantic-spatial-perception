# Migration Journal: Staged Representation Builder

## Scope

This journal tracks concrete behavior changes made while implementing
`agents/MIGRATION_PLAN.md`.

## Pre-Migration Baseline Attempt

- Date: 2026-03-06
- Command:
  - `uv run scripts/benchmark_saved_world.py --captured-root outputs/captured_worlds --run-all --benchmark-output-root outputs/benchmarks/migration_baseline_pre`
- Result:
  - Blocked locally because `GEMINI_API_KEY` / `GOOGLE_API_KEY` was not set in the shell.
  - No live LLM baseline artifacts were produced in this environment.

## Behavioral Changes

### Planner Construction

- Replaced the one-shot task analysis contract with staged outputs:
  - `abstract_goal`
  - `predicate_inventory`
  - `action_schemas`
  - `grounding_summary`
  - `diagnostics`
- `LLMTaskAnalyzer` now exposes:
  - `analyze_goal(...)`
  - `analyze_predicates(...)`
  - `analyze_actions(...)`
  - `analyze_grounding(...)`

### Domain Maintenance

- Replaced domain-text refinement as the primary repair path.
- `PDDLDomainMaintainer` now owns:
  - `build_representation(...)`
  - `ground_representation(...)`
  - embedded validation
  - `repair_representation(...)`
- Repair behavior now follows reverse abstraction order:
  - actions
  - predicates
  - goal

### Orchestration

- `TaskOrchestrator.process_task_request()` now builds stages 1-3 explicitly.
- Detection callbacks now refresh only grounding.
- Planning retries now classify likely failure layer and request targeted repair.

### Diagnostics

- Representation validation now records:
  - per-layer validity
  - validation issues
  - repair history
  - LLM call counts / elapsed time
- `scripts/benchmark_saved_world.py` now emits replay summaries with:
  - staged validation status
  - grounding gaps
  - undefined-symbol error counts
  - LLM call metrics

## Test Evidence

### Deterministic Migration Tests

- Added `tests/test_representation_builder_migration.py`
- Coverage:
  - staged build + grounding on cached `vegetables` world
  - missing grounding on cached `markers` world
  - targeted action-layer repair without mutating goal/predicate layers

### Repository Test Suite

- Command:
  - `uv run -m pytest tests`
- Result:
  - `10 passed, 1 skipped`

### Live Cached-World Replays

- Shell mode:
  - login shell via `bash`
- Successful replay:
  - `uv run scripts/benchmark_saved_world.py --world-dir outputs/captured_worlds/vegetables --benchmark-output-root outputs/benchmarks/migration_post_login_retry2`
  - Result:
    - success
    - plan length `4`
    - refinements `0`
    - summary at `outputs/benchmarks/migration_post_login_retry2/summary_20260306_153245.json`
- Remaining failures:
  - `blocks`
    - goal layer still produces quantified / non-ground STRIPS-incompatible goals
    - goal repair returned an empty model response on the observed run
  - `markers`
    - goal/constraint shaping still leaks an unsupported `broken` predicate into the symbolic representation
    - targeted repair returned an empty model response on the observed run

### Replay-Driven Fixes Made After First Live Run

- Normalized stored action schemas into a solver-safe STRIPS subset by removing
  negative preconditions before validation and serialization.
- Added a pre-solve validation guard that skips early repair until the first
  wait-for-objects pass has had a chance to ground the replay.
- Hardened empty-response handling for repair calls so failures now report the
  real issue instead of crashing with `json.loads(None)`.

## Incidental Fixes Discovered During Validation

- `PrimitiveExecutor.prepare_plan()` now normalizes translated `target_position`
  outputs to plain rounded Python float lists.
- This fixed an existing test failure unrelated to the migration but surfaced by
  the full-suite run.
