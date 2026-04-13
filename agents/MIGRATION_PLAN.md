# In-Place Migration Plan: Representation Builder

## 1. Problem Context

We are building a system that turns natural language instructions and perception into symbolic task descriptions (PDDL domain + problem), then uses a classical planner to generate and execute robot plans.

Right now, the weakest link is how we construct and debug those symbolic representations:

- **Everything is generated at once.**  
  The current system asks an LLM to jointly produce goals, predicates, and actions (and sometimes grounding) in a single or near-single step. We only find out something is wrong when the planner fails.

- **Failures are opaque and entangled.**  
  When planning fails, we cannot tell whether the problem is:
  - the **goal** being mis-specified,
  - the **predicate set** being incomplete or misaligned with the goal,
  - the **actions** lacking the right preconditions/effects, or
  - the **grounding** between symbols and the actual world being wrong.  
  All of these errors are mixed together in one representation, so there is no principled way to localize or repair them.

This project replaces that behavior with a **deliberate abstraction order** and **layered repair procedure**: we construct goals, predicates, actions, and grounding in a fixed abstract→concrete sequence, and when planning or execution fails, we walk back up that sequence to identify and correct the *lowest* layer that is wrong, instead of regenerating everything blindly.

## 2. Target Behavior

Representation should be built in this order:

1. Task language -> abstract goal
2. Abstract goal -> minimal predicate set
3. Predicates + goal -> action schemas
4. Predicates + actions + observed world -> grounded planning model

When planning fails, repair should run in reverse abstraction order:

1. action repair
2. predicate repair
3. goal repair

Only fall back to full regeneration if targeted repairs fail.

## 3. Why a breaking change

- Existing one-shot contracts and field names encode the old assumptions.
- Keeping compatibility would preserve ambiguous interfaces and slow down migration.
- Hard cut makes behavior and ownership clear.

## 4. One Anchor Example

Task: “Get the jar out of the cabinet without breaking anything.”

Forward build:

1. Goal layer defines abstract success conditions: jar retrieved, fragile objects intact.
2. Predicate layer chooses only required predicates (containment/open state/holding/intact/reachability).
3. Action layer creates schemas whose effects can move those predicates toward the goal.
4. Grounding layer binds symbols to observed object IDs and executable skills.

Failure handling:

- If no action changes cabinet/jar containment relation, repair actions first.
- If actions are coherent but needed state concepts are missing, repair predicates.
- If both are coherent but intent is wrong, repair goal.

This example is representative of the intended behavior across tasks.

## 5. Scope and Constraints

- In-place replacement in existing files and paths.
- Validator and repair logic live inside orchestrator/representation builder code, not separate modules.
- Keep stage count minimal (4 forward stages).

## 6. Planned Architecture

## 6.1 `src/planning/llm_task_analyzer.py`

Use staged analysis methods as primary interface:

- `analyze_goal(...)`
- `analyze_predicates(...)`
- `analyze_actions(...)`
- `analyze_grounding(...)`

`analyze_task(...)` may remain only as an internal wrapper that calls the staged methods; it is not the conceptual contract anymore.

## 6.2 `src/planning/pddl_domain_maintainer.py`

Make this the representation-construction core with embedded checks and repair:

- `build_representation(task, scene_context, image)`
- `ground_representation(detected_objects, predicates)`
- `_validate_representation(...)`
- `repair_representation(failure_context, layer)`

Retire domain text replacement as the main repair mechanism.

## 6.3 `src/planning/task_orchestrator.py`

Wire orchestration to staged builder calls:

- `process_task_request()` runs stages 1-3 via `build_representation(...)`.
- `_on_detection_callback()` refreshes grounding (stage 4).
- `solve_and_plan_with_refinement()` classifies failure and triggers `repair_representation(...)` by layer.

## 6.4 `src/planning/utils/task_types.py`

Replace the analysis data contract in place with staged fields:

- `abstract_goal`
- `predicate_inventory`
- `action_schemas`
- `grounding_summary`
- `diagnostics`

No backward compatibility aliases.

## 6.5 Existing config paths

Use existing config files, replace prompt keys:

- `config/llm_task_analyzer_prompts.yaml`
  - `goal_prompt`
  - `predicates_prompt`
  - `actions_prompt`
  - `grounding_prompt`
- `config/pddl_domain_maintainer_prompts.yaml`
  - `repair_actions_prompt`
  - `repair_predicates_prompt`
  - `repair_goal_prompt`

## 6.6 `src/planning/pddl_representation.py`

Keep as serializer/holder. Its inputs must come from validated staged outputs, not ad hoc patching.

## 7. Validation and Repair Behavior

## 7.1 Embedded validation checks

Validation is implemented inside maintainer/orchestrator flow and includes:

- Goal expressibility by current predicates.
- Predicate minimality/relevance to goal and action effects.
- Action influence coverage: each goal-relevant predicate should be achievable or maintainable.
- Grounding completeness: symbolic references map to observed objects/skills.
- PDDL symbol consistency (defined predicates, object references, arity consistency).

## 7.2 Layered repair policy

On solver/validation failure:

1. classify likely failing layer
2. run targeted repair prompt for that layer
3. revalidate and retry planning
4. escalate to next layer only if needed

This avoids full regeneration loops by default.

## 8. Explicit Breaking Changes

1. One-shot analysis contract is removed as source-of-truth behavior.
2. Old `TaskAnalysis` field schema is removed.
3. Domain text replacement refinement path is removed.
4. Old prompt key names are removed.
5. Demos/scripts/tests are updated to staged outputs and layer-based diagnostics.

## 9. Migration Sequence

### Step 0: Freeze pre-migration baselines

Capture artifacts from current behavior for comparison:

- task text
- observed objects/predicates snapshot
- generated domain/problem PDDL
- solver result (success, plan length, error)
- number of refinement attempts
- LLM calls/tokens

Use existing replayable assets first:

- `tests/assets/continuous_pick_fixture`
- `tests/assets/continuous_pick_fixture_old`
- `scripts/replay_cached_demo.py`

### Step 1: Replace analyzer internals

Implement staged analysis methods in `llm_task_analyzer.py`.

### Step 2: Replace maintainer internals

Implement `build/ground/validate/repair` in `pddl_domain_maintainer.py`.

### Step 3: Rewire orchestrator repair loop

Use layer-targeted repair in `solve_and_plan_with_refinement()` path.

### Step 4: Update call sites in place

Update existing files, including:

- `examples/tamp_demo.py`
- `examples/orchestrator_demo.py`
- `examples/test_orchestrator_solver.py`
- `src/task_motion_planner.py`

## 10. Comparison-Based Testing (During Migration)

No runtime dual mode. Compare post-migration runs against frozen pre-migration artifacts.

Metrics:

- planner success rate
- non-empty plan rate
- repair iterations per failure
- undefined symbol errors in generated PDDL
- grounding coverage
- LLM call count and token count
- wall-clock planning pipeline time

Cutover gate:

- no planner success regression on migration scenario set
- zero undefined symbol/object errors
- repair iterations reduced or unchanged
- latency/token overhead controlled or justified by success gains

---

This document is the complete implementation-time reference for migrating to a staged, diagnosable representation builder using existing files and paths.
