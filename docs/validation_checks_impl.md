# GTAMP Validation Checks — Implementation Status

Cross-reference of every check against the actual implementation.

**Key:**
- ✅ Fully implemented and matches spec
- ⚠️ Partially implemented (behaviour deviates from spec)
- ❌ Not implemented

---

## Layer L1 — Goal Specification

All checks implemented in [`_validate_l1`](../src/planning/layered_domain_generator.py#L456).

| ID | Condition | Status | Implementation |
|----|-----------|--------|----------------|
| L1-V1 | Goal set is non-empty | ✅ | [layered_domain_generator.py:471](../src/planning/layered_domain_generator.py#L471) — REJECT with early return |
| L1-V2 | Goal predicates are syntactically well-formed (parentheses, grounded, naming pattern, ≥1 arg, no empty args) | ✅ | [layered_domain_generator.py:490–527](../src/planning/layered_domain_generator.py#L490) — primary regex at L490, token-level sub-checks through L527 |
| L1-V3 | No contradictory goals under uniqueness constraints | ✅ | [layered_domain_generator.py:536–554](../src/planning/layered_domain_generator.py#L536) — REJECT on `pred:subject` key collision per `UNIQUENESS_CONSTRAINTS` (line 93) |
| L1-V4 | All goal arguments reference entities in the scene | ⚠️ | [layered_domain_generator.py:556–564](../src/planning/layered_domain_generator.py#L556) — REJECT implemented; entity set includes object IDs, semantic labels (`object_type`, `class_name`, `surface_id`), and `"robot"`. **EXPLORATION flag not raised** (missing entity not recorded as an exploration target). |

---

## Layer L2 — Predicate Vocabulary

Checks implemented in [`_validate_l2`](../src/planning/layered_domain_generator.py#L612) and [`_repair_l2_auto`](../src/planning/layered_domain_generator.py#L771).

| ID | Condition | Status | Implementation |
|----|-----------|--------|----------------|
| L2-V1 | All goal predicate names are defined in vocabulary | ✅ | [layered_domain_generator.py:627–634](../src/planning/layered_domain_generator.py#L627) — REJECT |
| L2-V2 | All names follow PDDL naming convention | ✅ | [layered_domain_generator.py:636–652](../src/planning/layered_domain_generator.py#L636) — REJECT; checks predicate name and type names against `_PDDL_NAME_RE` (line 105) |
| L2-V3 | Every sensed/external predicate has a checked variant | ✅ | Type-mismatch REJECT at [layered_domain_generator.py:672–705](../src/planning/layered_domain_generator.py#L672); AUTO-REPAIR (generate missing `checked-*` variant) in `_repair_l2_auto` at [layered_domain_generator.py:771](../src/planning/layered_domain_generator.py#L771) |
| L2-V4 | All argument types are in the valid type set | ✅ | [layered_domain_generator.py:707–716](../src/planning/layered_domain_generator.py#L707) — REJECT per invalid type against `VALID_PDDL_TYPES` (line 87); also detects cross-predicate variable type conflicts |
| L2-V5 | No unused predicates (deferred to after L3) | ✅ | Deferred; runs in [`_prune_unused_predicates`](../src/planning/layered_domain_generator.py#L1213) after L3 validation passes — AUTO-REPAIR (remove unused) |
| L2-V6 | All type classifications are valid | ✅ | [layered_domain_generator.py:759–767](../src/planning/layered_domain_generator.py#L759) — validates `type_classifications` values against `VALID_TYPE_CLASSIFICATIONS` (line 100) |

---

## Layer L3 — Action Schemas

Checks implemented in [`_validate_l3`](../src/planning/layered_domain_generator.py#L868) and [`_repair_l3_auto`](../src/planning/layered_domain_generator.py#L1099).

| ID | Condition | Status | Implementation |
|----|-----------|--------|----------------|
| L3-V1 | All predicates in action formulas exist in vocabulary | ✅ | [layered_domain_generator.py:909–931](../src/planning/layered_domain_generator.py#L909) — REJECT with list of undefined predicates and available vocab |
| L3-V2 | All variables declared in parameter list with valid types | ⚠️ | [layered_domain_generator.py:933–955](../src/planning/layered_domain_generator.py#L933) — undeclared variable detection ✅; **type validity for declared parameters not checked** (spec L3-V2 second half) |
| L3-V3 | All checked-* in preconditions have a producing action | ✅ | Detection at [layered_domain_generator.py:957–1016](../src/planning/layered_domain_generator.py#L957) + AUTO-REPAIR (generate sensing action) in `_repair_l3_auto` at [layered_domain_generator.py:1099](../src/planning/layered_domain_generator.py#L1099) |
| L3-V4 | All goal predicates reachable via relaxed planning graph | ✅ | [layered_domain_generator.py:1018–1096](../src/planning/layered_domain_generator.py#L1018) — delete-relaxation BFS using `bfs_reachable_predicates` (line 197); REJECT with per-predicate diagnosis |

---

## Layer L4 — Grounding and Skill Specifications

Algorithmic pre-checks V1–V3 in [`_run_l4_precheck`](../src/planning/layered_domain_generator.py#L1267) are **non-blocking warnings** (spec calls for REJECT/EXPLORATION).
Skill-level checks V4–V7 are in the primitives layer. V8–V9 are not implemented.

| ID | Condition | Status | Implementation |
|----|-----------|--------|----------------|
| L4-V1 | Grounding rules exist for all PDDL types | ⚠️ | [layered_domain_generator.py:1326–1333](../src/planning/layered_domain_generator.py#L1326) — emits WARNING, does not REJECT |
| L4-V2 | Scene has elements for all required categories | ⚠️ | [layered_domain_generator.py:1335–1360](../src/planning/layered_domain_generator.py#L1335) — emits WARNING, does not trigger perception update + REJECT |
| L4-V3 | All goal entities present in scene | ⚠️ | [layered_domain_generator.py:1370–1377](../src/planning/layered_domain_generator.py#L1370) — emits WARNING; **EXPLORATION mechanism not triggered** |
| L4-V4 | All primitives in sequence exist in library | ✅ | [`SkillPlan.validate`](../src/primitives/skill_plan_types.py#L218) called from [`PrimitiveExecutor.prepare_plan`](../src/primitives/primitive_executor.py#L100) — REJECT (raises `ValueError`) |
| L4-V5 | All semantic parameters valid per schema | ✅ | [`PrimitiveSchema.validate`](../src/primitives/skill_plan_types.py#L75) checks required params, unknown params, and per-param validators — REJECT |
| L4-V6 | All symbolic labels resolve in scene | ⚠️ | [`PrimitiveExecutor.prepare_plan`](../src/primitives/primitive_executor.py#L176) falls back to `point_label` from registry but does **not REJECT if label is unresolvable** — silent fallback |
| L4-V7 | All required parameters present | ✅ | [`PrimitiveSchema.validate`](../src/primitives/skill_plan_types.py#L85) checks `required_params` — REJECT |
| L4-V8 | All constraints instantiable after grounding | ❌ | Not implemented — no constraint template system exists yet |
| L4-V9 | All target poses pass feasibility pre-checks (workspace, reach, aperture) | ❌ | Not implemented — PyBullet IK failure is the only runtime guard |

---

## Layer L5 — Initial State Construction

All checks embedded in [`_run_l5`](../src/planning/layered_domain_generator.py#L1385).

| ID | Condition | Status | Implementation |
|----|-----------|--------|----------------|
| L5-V1 | All predicate-entity combinations assigned | ⚠️ | State constructed algorithmically; no explicit enumeration-and-diff check. Missing combinations are implicitly FALSE under closed-world assumption but are not logged. |
| L5-V2 | Checked and sensed predicates initialized to FALSE | ✅ | [layered_domain_generator.py:1577–1593](../src/planning/layered_domain_generator.py#L1577) — AUTO-REPAIR (set to FALSE) |
| L5-V3 | Entity types in facts match predicate argument types | ⚠️ | [layered_domain_generator.py:1595–1640](../src/planning/layered_domain_generator.py#L1595) — detects unknown entity IDs and removes offending facts (WARNING + AUTO-REPAIR); does **not** verify declared PDDL type compatibility |
| L5-V4 | Spatial facts consistent with scene representation | ⚠️ | Spatial facts derived from position-based heuristics (dz, dx, dy thresholds) at [layered_domain_generator.py:1483–1551](../src/planning/layered_domain_generator.py#L1483); no separate post-construction validation pass |
| L5-V5 | Initial state does not already satisfy all goals | ✅ | [layered_domain_generator.py:1642–1657](../src/planning/layered_domain_generator.py#L1642) — WARNING (non-blocking), matches spec |

---

## PDDLDomainMaintainer Representation Validation

The maintainer adds a second validation pass that runs against the staged `TaskAnalysis` before solving. Implemented in [`_validate_representation`](../src/planning/pddl_domain_maintainer.py#L837), with failure routing via [`classify_failure_layer`](../src/planning/pddl_domain_maintainer.py#L749) and targeted repair via [`repair_representation`](../src/planning/pddl_domain_maintainer.py#L277).

| Check | Condition | Status | Implementation |
|-------|-----------|--------|----------------|
| MV-1 | Goal predicates expressible by predicate inventory | ✅ | [pddl_domain_maintainer.py:882–889](../src/planning/pddl_domain_maintainer.py#L882) — REJECT (issues `predicates` layer) |
| MV-2 | Action predicates all defined in inventory | ✅ | [pddl_domain_maintainer.py:914–922](../src/planning/pddl_domain_maintainer.py#L914) — REJECT (issues `predicates` layer) |
| MV-3 | No negative preconditions (STRIPS backend constraint) | ✅ | [pddl_domain_maintainer.py:900–906](../src/planning/pddl_domain_maintainer.py#L900) — REJECT (issues `actions` layer) |
| MV-4 | Each goal predicate satisfiable by an action effect or grounded state | ✅ | [pddl_domain_maintainer.py:926–933](../src/planning/pddl_domain_maintainer.py#L926) — REJECT (issues `actions` layer) |
| MV-5 | No unused predicates in inventory | ⚠️ | [pddl_domain_maintainer.py:935–937](../src/planning/pddl_domain_maintainer.py#L935) — WARNING only, not a blocking issue |
| MV-6 | All goal object references present in detected objects | ⚠️ | [pddl_domain_maintainer.py:950–954](../src/planning/pddl_domain_maintainer.py#L950) — WARNING only when objects have been observed |
| MV-7 | Missing goal object bindings (grounding complete) | ✅ | [pddl_domain_maintainer.py:939–948](../src/planning/pddl_domain_maintainer.py#L939) — REJECT (issues `grounding` layer) when objects are present but bindings are missing |

---

## Summary

| Layer | Checks | Fully ✅ | Partial ⚠️ | Missing ❌ |
|-------|--------|----------|-----------|-----------|
| L1 | 4 | 3 | 1 (L1-V4 exploration) | 0 |
| L2 | 6 | 6 | 0 | 0 |
| L3 | 4 | 3 | 1 (L3-V2 type validity) | 0 |
| L4 (grounding) | 3 | 0 | 3 (non-blocking) | 0 |
| L4 (skills) | 6 | 4 (V4, V5, V7) | 1 (V6 silent fallback) | 2 (V8, V9) |
| L5 | 5 | 2 (V2, V5) | 3 (V1, V3, V4) | 0 |
| Maintainer (MV) | 7 | 5 (MV-1–4, MV-7) | 2 (MV-5, MV-6) | 0 |
| **Total** | **35** | **23** | **10** | **2** |

### Remaining gaps

1. **L4-V1/V2/V3** — upgrade from WARNING to REJECT / perception-update / EXPLORATION as specified.
2. **L4-V6** — REJECT (or trigger perception update) when a symbolic label cannot be resolved, instead of silently falling back.
3. **L4-V8** — constraint instantiation check (requires a constraint template registry).
4. **L4-V9** — workspace/reachability/aperture feasibility pre-checks before motion planning.
5. **L1-V4** — record missing entities as exploration targets (EXPLORATION mechanism not implemented).
6. **L3-V2** — add type validity check for declared action parameters (currently only checks that variables are declared, not that their declared types are valid).
