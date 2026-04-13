# GTAMP Layer Validation Specification
## Developer Reference

Version: 1.1
Status: Implementation Specification

---

## Purpose

This document specifies every formatting requirement and validation check for each layer of the GTAMP domain generation pipeline. Each check is described in terms of what condition must hold, how to verify it, what to do when it fails, and what message to return. A developer reading this document should be able to implement the complete validation pipeline without ambiguity.

---

## Terminology

**REJECT** means the output fails validation and must be regenerated. The specific failure message is returned to the LLM or VLM that produced the output, along with any corrective context (such as the list of valid options), and the system requests a new output.

**AUTO-REPAIR** means the system fixes the problem deterministically without any LLM or VLM call. The repair procedure is constructive — it generates the missing artifact from existing validated information. Auto-repairs are always logged.

**WARNING** means the condition is flagged but does not block progress. The system logs the warning and continues.

**EXPLORATION** means the failure indicates a missing entity that may exist in the environment but has not been observed. The system enters the exploration mechanism to attempt to locate the entity before declaring failure.

**Predicate type classifications** used throughout this document:
- *Robot state:* Properties directly known from robot state feedback. Examples: robot-at, gripper-empty.
- *Object state:* Properties observable from the perception system. Examples: object-at, on.
- *Sensed:* Properties that require active sensing to verify. Examples: object-graspable, path-clear.
- *External:* Properties of the external world that are uncertain. Examples: door-locked, container-full.
- *Checked:* Predicates that track whether sensing has occurred for a corresponding sensed or external predicate. Always initialized to FALSE. Examples: checked-object-graspable, checked-path-clear.

**PDDL naming convention** referenced throughout: names must be lowercase, may contain hyphens between segments, must start with a letter, and may contain digits. Valid examples: object-at, checked-path-clear, robot-near. Invalid examples: Object_At, HOLDING, gripper empty.

---

## Layer L1: Goal Specification

### Expected Output

The LLM produces a set of goal predicates, where each goal predicate consists of a predicate name, one or more argument strings, and an optional negation flag. The goal set represents the desired end-state conditions that the planner must achieve.

---

### L1-V1: Non-Emptiness

**What must hold:** The goal set must contain at least one goal predicate.

**How to check:** Count the number of goal predicates. If the count is zero, the check fails.

**On failure:** REJECT. Return the message: "Goal set is empty. Re-examine the task description and extract at least one desired end-state condition."

---

### L1-V2: Well-Formedness

**What must hold:** Every goal predicate must have a predicate name that conforms to the PDDL naming convention, and must have at least one argument. Each argument must be a non-empty string.

**How to check:** For each goal predicate, verify that the predicate name matches the PDDL naming convention. Verify that the argument list is non-empty. Verify that every argument is a non-empty string.

**On failure:** REJECT. Return the specific goal predicates that failed, identifying whether the issue is the predicate name, missing arguments, or empty argument strings. Include a description of the expected naming convention.

---

### L1-V3: Internal Consistency

**What must hold:** The goal set must not contain contradictory assertions. A contradiction occurs when two goal predicates assert incompatible values for a uniqueness-constrained argument. For example, if the predicate "object-at" constrains an object to be at only one location, then the goals "(object-at mug counter)" and "(object-at mug table)" are contradictory because the mug cannot be at two locations simultaneously.

**How to check:** The system maintains a uniqueness constraint registry — a static configuration that lists which predicates have uniqueness constraints and which argument positions are constrained. For each predicate in the registry, group all non-negated goal predicates using that predicate by the constrained argument. If any constrained argument maps to more than one value of the varying argument, report a contradiction.

**On failure:** REJECT. Return the specific contradicting goal predicates, identify which uniqueness constraint they violate, and explain why they conflict (e.g., "entity 'mug' cannot be at both 'counter' and 'table' simultaneously because 'object-at' constrains an object to one location").

---

### L1-V4: Argument Plausibility

**What must hold:** Every argument string in every goal predicate should reference an entity that exists in the current scene representation.

**How to check:** Collect all entity identifiers and semantic labels from the scene representation — this includes all object IDs, all object semantic labels, all surface IDs, all surface semantic labels, and the robot identifier. For each argument in each goal predicate, check whether it appears in this combined set.

**On failure:** REJECT. Return the specific arguments that do not match any known entity, along with the complete list of known entities. Note that arguments failing this check are candidates for exploration — the entity may exist in the environment but has not been observed yet. The system should record these as potential exploration targets for use at L4.

---

### L1 Auto-Repair

None. All L1 failures require LLM re-generation with the specific failure message.

### L1 Locking

On successful validation, the goal set is frozen as an immutable artifact. No downstream layer may modify it. All downstream layers receive a read-only reference to the locked goal set.

---

## Layer L2: Predicate Vocabulary

### Expected Output

The LLM produces a set of predicate definitions. Each predicate definition consists of a name, a list of typed arguments (where each argument has a name and a type), and a type classification indicating how the predicate's truth value is determined (robot state, object state, sensed, external, or checked).

---

### L2-V1: Goal Coverage

**What must hold:** Every predicate name that appears in any goal predicate from L1 must have a corresponding definition in the predicate set.

**How to check:** Extract all predicate names from the locked L1 goal set. Check that each of these names exists as a key in the predicate set. Any goal predicate name not found in the predicate set is a missing predicate.

**On failure:** REJECT. Return the specific predicate names that appear in goals but have no definition. Instruct the LLM to add definitions for these predicates.

---

### L2-V2: Naming Compliance

**What must hold:** Every predicate name and every argument type name must conform to the PDDL naming convention.

**How to check:** For each predicate, verify the predicate name matches the naming convention. For each argument in each predicate, verify the type string matches the naming convention.

**On failure:** REJECT. Return each name or type that violates the convention, along with a description of the expected format.

---

### L2-V3: Checked-Variant Completeness

**What must hold:** For every predicate whose type classification is "sensed" or "external," a corresponding predicate must exist with the name "checked-" prepended to the original name, with type classification "checked," and with identical argument types in the same order.

**How to check:** Filter the predicate set for all predicates classified as "sensed" or "external." For each such predicate with name N, check whether a predicate named "checked-N" exists in the set. If it exists, verify that its type classification is "checked" and that its argument types match the original predicate's argument types exactly.

If the checked variant exists but has mismatched argument types or wrong classification, this is a type mismatch error. If the checked variant is entirely missing, this triggers auto-repair.

**On type mismatch:** REJECT. Return the specific mismatches for LLM correction.

**On missing variant:** AUTO-REPAIR. For each missing checked variant, the system creates a new predicate definition with name "checked-N," the same argument names and types as the base predicate, and type classification "checked." No LLM call is required.

---

### L2-V4: Type Validity

**What must hold:** Every argument type used in any predicate definition must be a member of the valid PDDL type set recognized by the system. The valid types are determined by the pre-defined grounding rules — specifically, every type must have a corresponding grounding rule that maps it to a scene element category. The standard valid types are: object, location, surface, robot, and region. Additional types may be added to the system configuration if corresponding grounding rules are provided.

**How to check:** Collect all type strings from all argument definitions across all predicates. Check each type string against the set of valid types. Any type not in the valid set is a violation.

**On failure:** REJECT. Return each invalid type, the predicate and argument it appears in, and the list of valid types.

---

### L2-V5: Minimality (Deferred Check)

**What must hold:** Every predicate in the vocabulary should be referenced by at least one action precondition, action effect, or goal predicate. Predicates that appear nowhere are dead weight that bloats the state space.

**When to check:** This check is NOT run during L2 validation. It is deferred until after L3 validation completes, because the action schemas needed to determine predicate usage do not exist yet at L2 time.

**How to check:** After L3 is validated, compute the union of all predicate names appearing in: the goal set (L1), any action precondition (L3), and any action effect (L3). Any predicate in the vocabulary that does not appear in this union is unused.

**On failure:** AUTO-REPAIR. Remove all unused predicates from the vocabulary. This is safe because a predicate that appears nowhere in goals or actions cannot affect planning. Log which predicates were removed.

---

### L2-V6: Type Classification Validity

**What must hold:** Every predicate's type classification must be one of the recognized classification values: "robot_state," "object_state," "sensed," "external," or "checked."

**How to check:** For each predicate, verify that its type classification string matches one of the five valid values.

**On failure:** REJECT. Return each invalid classification, the predicate it appears on, and the list of valid classifications.

---

### L2 Locking

On successful validation (including any auto-repairs), the predicate set is frozen as an immutable artifact. The auto-repair log (listing any checked variants or other modifications that were generated automatically) is attached to the artifact for traceability.

---

## Layer L3: Action Schema Generation

### Expected Output

The LLM produces a set of action schemas. Each action schema consists of a name, a list of typed parameters (where each parameter has a name and a type), a precondition formula expressed as a PDDL string, and an effect formula expressed as a PDDL string.

### Prerequisite Capability

The validation system must be able to parse PDDL formula strings to extract predicate names and variable references. Specifically:
- Predicate extraction must handle conjunction (and), negation (not), disjunction (or), and nested formulas, returning the set of all predicate names referenced.
- Positive effect extraction must return only predicate names that appear as positive effects (not wrapped in negation).
- Variable extraction must return all PDDL variable references (strings beginning with "?") from a formula.

---

### L3-V1: Symbolic Closure

**What must hold:** Every predicate name that appears anywhere in any action's precondition formula or effect formula must exist as a defined predicate in the locked L2 predicate vocabulary.

**How to check:** For each action schema, parse the precondition formula and effect formula to extract all referenced predicate names. Take the union of all predicate names across all actions. Check that every name in this union exists in the L2 predicate set. Any predicate name found in an action formula but not in the predicate set is a violation.

**On failure:** REJECT. For each violating action, return the action name, the undefined predicate names, and whether each undefined predicate appears in the preconditions, effects, or both. Include the complete list of available predicates from L2 so the LLM can correct its references.

---

### L3-V2: Parameter Consistency

**What must hold:** For every action schema, every variable referenced in its precondition or effect formula must be declared in that action's parameter list, and every declared parameter must have a type from the valid PDDL type set.

**How to check:** For each action schema, extract all variable references from the precondition and effect formulas. Collect the set of declared parameter names (with the "?" prefix). Any variable in the formulas that does not appear in the declared set is an undeclared variable. Additionally, verify that every declared parameter's type is in the valid PDDL type set (same set used in L2-V4).

**On failure:** REJECT. Return the specific undeclared variables per action (listing which formula they appear in), and any parameters with invalid types. Include the action's declared parameter list for context.

---

### L3-V3: Sensing Coverage

**What must hold:** For every predicate whose name starts with "checked-" that appears in any action's precondition, there must exist at least one action in the action set whose positive effects include setting that checked predicate to true. This ensures the planner has a way to verify every uncertain property it might need to rely on.

**How to check:** Collect all predicate names starting with "checked-" that appear in any action's preconditions. Separately, collect all predicate names starting with "checked-" that appear as positive effects in any action. The difference between these two sets (predicates required in preconditions but never produced by any effect) is the set of uncovered checked predicates.

**On failure (uncovered predicates exist):** AUTO-REPAIR. For each uncovered checked predicate "checked-N," the system generates a sensing action as follows: the action is named "check-N," its parameters match the argument types of the base predicate N (looked up in the L2 vocabulary), its precondition references an appropriate proximity predicate (such as "robot-near" applied to the first argument, if such a predicate exists in the vocabulary; otherwise the precondition is left trivially satisfiable), and its effect sets "checked-N" to true for the given arguments. This generation is deterministic and requires no LLM call.

If the base predicate N does not exist in the L2 vocabulary, the auto-repair cannot proceed and the system must REJECT, reporting that the checked predicate references a non-existent base predicate.

---

### L3-V4: Goal Achievability

**What must hold:** The action set must be structurally capable of reaching every goal predicate from some plausible initial state. At minimum, every goal predicate must be producible through some chain of action effects. This is a structural check, not a full planning attempt — it verifies the domain is not trivially unsolvable.

**How to check:** Construct a relaxed planning graph using delete-relaxation. This is a standard technique from classical planning:

Begin with an initial fact layer containing all predicate names that could plausibly be true in any initial state. This includes all predicates classified as "robot_state" or "object_state" (which can be grounded from perception) and all "checked" predicates (which can be set by sensing actions). In the relaxed version, ignore delete effects — only accumulate positive effects.

Iteratively expand: for each action whose precondition predicates are all present in the current fact layer, add all of that action's positive effect predicates to the fact layer. Repeat until either all goal predicates are present in the fact layer (pass) or no new predicates are added in an expansion step (fixpoint reached without covering goals — fail).

This algorithm is polynomial in the size of the domain (number of actions times number of predicates times maximum number of expansion layers). For typical manipulation domains it completes in milliseconds.

**On failure:** REJECT. Report which goal predicates are unreachable, and for each unreachable predicate, report whether any action produces it at all. If no action produces a goal predicate, the message should state: "No action in the domain produces the predicate [name]. Add an action whose effects include this predicate." If actions exist that produce the predicate but their preconditions are never satisfiable, the message should identify the broken precondition chain.

---

### L3 Post-Validation Step

After L3 passes all checks, immediately run the deferred L2-V5 minimality check using the now-available action schemas.

### L3 Locking

On successful validation (including any auto-repaired sensing actions), the action set is frozen. The auto-repair log is attached.

---

## Layer L4: Symbolic Grounding and Skill Specification Generation

L4 produces two artifacts and validates both. Artifact 4a (grounding mappings) is a system-level check against the scene. Artifact 4b (skill specifications) validates VLM-generated primitive decompositions against the primitive library schema and the scene representation.

### Validation Order

L4 validation must be executed in the following order because later checks depend on results of earlier checks:

1. L4-V1 through L4-V3 (grounding checks — must pass before skill specifications are generated)
2. L4-V4 through L4-V7 (structural checks on the VLM output — verify against library schema)
3. Grounding resolution (resolve symbolic labels to concrete poses using the scene representation)
4. L4-V8 through L4-V9 (post-grounding checks — require resolved poses)

---

### L4-V1: Grounding Rule Completeness

**What must hold:** Every PDDL type used anywhere in the domain (in predicate argument types and action parameter types) must have a corresponding entry in the pre-defined grounding rule table. The grounding rules are a static system configuration that maps PDDL types to scene element categories.

**How to check:** Collect all type strings from all predicate arguments (L2) and all action parameters (L3). Check each type against the grounding rule table. Any type without a grounding rule is a violation.

**On failure:** REJECT. This is a system configuration error — either the grounding rule table needs to be extended to cover the new type, or the L2 vocabulary introduced a type that should not exist. Return the missing types and the available grounding rules.

---

### L4-V2: Scene Element Availability

**What must hold:** For every scene element category required by the grounding rules used in the domain, the current scene representation must contain at least one element of that category. For example, if any predicate uses the type "object" (which grounds to the objects section of the scene representation), then the objects section must be non-empty.

**How to check:** Determine which scene element categories are needed by looking up the grounding rule for each PDDL type used in the domain. For each required category, check whether the scene representation contains at least one element of that category. The categories are: objects, surfaces, robot, and regions.

**On failure:** REJECT. Trigger a perception update and retry. If the category is still empty after a fresh perception update, report a perception failure identifying which element categories are missing and which PDDL types require them.

---

### L4-V3: Entity Coverage

**What must hold:** Every entity referenced by name in the goal set must be present in the current scene representation, either as an object ID, object semantic label, surface ID, surface semantic label, or the robot identifier.

**How to check:** Extract all argument strings from all goal predicates. Build the set of known entities from the scene representation (all object IDs, object semantic labels, surface IDs, surface semantic labels, and the robot ID). Check each goal argument against this set.

**On failure:** EXPLORATION. Do not reject outright. Instead, flag each missing entity as an exploration target. The system should generate exploration sub-problems to locate these entities in the environment before proceeding. If exploration is not feasible or all exploration attempts for an entity are exhausted, then reject with the explanation that the entity could not be found.

---

### L4-V4: Primitive Membership

**What must hold:** Every primitive named in the VLM-generated skill specification must exist in the primitive library. The primitive library defines the complete executable vocabulary of the system.

**How to check:** For each step in the skill specification's primitive sequence, check whether the primitive name exists as a key in the primitive library.

**On failure:** REJECT. Return the unknown primitive names and the complete list of available primitives in the library. Send this to the VLM for re-decomposition.

---

### L4-V5: Semantic Parameter Validity

**What must hold:** For each primitive in the skill specification, every semantic parameter provided must be a recognized parameter name for that primitive, and its value must be within the set of valid values defined by the primitive library's parameter schema. Specifically: enum-typed parameters must have a value from the enumerated valid set; float-typed parameters must be numeric; integer-typed parameters must be whole numbers.

**How to check:** For each step in the primitive sequence, look up the primitive's parameter schema from the library. For each semantic parameter provided in the step, verify: (a) the parameter name exists in the schema, (b) the value conforms to the type specified in the schema (enum membership, numeric type, etc.).

**On failure:** REJECT. For each invalid parameter, return the step index, primitive name, parameter name, the provided value, and the valid options (for enum types) or expected type (for numeric types). Send to the VLM for re-decomposition.

---

### L4-V6: Symbolic Reference Resolution

**What must hold:** Every symbolic label used in situational parameters must resolve to a concrete element in the current scene representation. Specifically:

Labels in parameters typed as "interaction_point" must match a named interaction point on the target object in the scene representation. Labels in parameters typed as "surface" must match a surface ID or surface semantic label. Labels in parameters typed as "location" must match either a surface or an object identifier. Labels in parameters typed as "surface_boundary" must match a boundary identifier on the referenced surface. Labels in parameters typed as "region" must match a surface or region identifier.

**How to check:** For each step in the primitive sequence, for each situational parameter, look up the parameter's grounding type from the primitive library schema. Based on the grounding type, check whether the provided label exists in the appropriate section of the scene representation. For interaction points, this requires looking up the target object first and then checking its interaction point dictionary.

**On failure:** The response depends on the type of resolution failure:

If the label references an interaction point that does not exist on the target object: first trigger a targeted perception update for that object (the interaction point may exist but not have been detected yet). If the label still does not resolve after the perception update, REJECT and return the available interaction point labels for that object to the VLM for re-decomposition.

If the label references a surface, location, or region that does not exist: REJECT and return the available labels of the appropriate type to the VLM for re-decomposition.

---

### L4-V7: Required Parameter Completeness

**What must hold:** For each primitive in the skill specification, every parameter marked as "required" in the primitive library schema must have a value assigned, either as a semantic parameter or a situational parameter. Additionally, conditionally required parameters must be present when their condition is met. For example, the "hinge_boundary" parameter of the push_pull primitive is required whenever "articulation_mode" is set to "revolute."

**How to check:** For each step in the primitive sequence, look up the primitive's parameter schema. Identify all required parameters (those marked as required, plus any conditionally required parameters whose conditions are satisfied by the provided semantic parameters). Compute the set of all parameter names that have been assigned a value (union of semantic and situational parameter keys). Any required parameter not in the assigned set is missing.

**On failure:** REJECT. Return the step index, primitive name, and each missing required parameter with its specification (type, valid values if applicable). Send to the VLM for re-decomposition.

---

### L4-V8: Constraint Instantiation

**What must hold:** After all symbolic labels have been resolved to concrete geometric values via grounding rules, every geometric constraint defined by the primitive library for each primitive must be fully instantiable — meaning all pose, geometry, and spatial arguments in the constraint are concrete numeric values, not null or symbolic.

This check runs AFTER grounding resolution, not before.

**How to check:** For each step in the primitive sequence, look up the constraint templates defined for that primitive in the library (e.g., "reachable(target)" or "collision_free(path_to(target))"). Substitute the grounded values into each constraint template. Verify that no argument in the instantiated constraint is null, unresolved, or still symbolic.

This check does NOT evaluate whether the constraints are satisfied (that happens during geometric feasibility verification by the motion planner). It only verifies that the constraints CAN be evaluated — that all inputs are concrete.

**On failure:** REJECT. Return which constraints could not be instantiated and which arguments remain unresolved. This typically indicates a grounding resolution failure (a label resolved to a null pose, which may mean the perception system detected the entity but could not compute its geometry). Trigger a perception update for the relevant entity and retry. If still failing, send to VLM for re-decomposition.

---

### L4-V9: Scene Feasibility Pre-Check

**What must hold:** Every grounded target pose must pass lightweight geometric feasibility checks that can rule out obviously impossible configurations before invoking the expensive motion planner. These are necessary-but-not-sufficient conditions: passing them does not guarantee feasibility, but failing them guarantees infeasibility.

Three checks are performed:

**Workspace containment:** For every primitive that moves the gripper or base to a target pose, the target position must be within the robot's pre-computed workspace bounding volume. This is a single point-in-volume test.

**Gross reachability:** For every gripper target pose, the Euclidean distance from the robot base to the target position must not exceed the robot's maximum reach. This is a single distance comparison.

**Grasp aperture compatibility:** For any primitive sequence where a close_gripper follows a move_gripper_to_pose (indicating a grasp), the target object's bounding box dimensions must be compatible with the selected grasp mode and the gripper's maximum aperture. For a top-down grasp, the object's minimum horizontal dimension must not exceed the gripper aperture. For a side grasp, the object's minimum cross-sectional dimension (perpendicular to the approach) must not exceed the gripper aperture. This is a comparison of two scalar values.

**How to check:** For each step referencing a target pose: resolve the pose to coordinates, perform the workspace containment check, perform the reachability distance check, and (if a grasp is indicated by the sequence context) perform the aperture compatibility check.

**On failure:** REJECT. Return each failing check with the specific geometric values (distance vs. max reach, dimension vs. aperture, etc.) and the explanation of why it is infeasible. Send to the VLM for re-decomposition, excluding the failing options. For example, if a top-down grasp fails the aperture check, tell the VLM: "Top-down grasp is infeasible for this object (minimum dimension exceeds gripper aperture). Consider a side grasp or different interaction point."

---

### L4 Locking

On successful validation of both artifacts, the grounding mappings and skill specifications are frozen. Skill specifications are also cached in the Domain Knowledge Base indexed by action name and target object, with the object's class annotation for cross-instance reuse.

---

## Layer L5: Initial State Construction

L5 is fully automated — no LLM or VLM is involved. The system deterministically constructs the initial PDDL state by applying grounding rules to the current scene representation, following initialization rules that depend on each predicate's type classification.

### Construction Rules

The initial state is a set of facts, where each fact is a predicate applied to specific entity arguments with a boolean truth value.

The system enumerates all applicable predicate-entity combinations by computing, for each predicate, the set of all entity tuples matching the predicate's argument types (using the grounding rules to determine which entities match each type). For each such tuple, a truth value is assigned according to the predicate's type classification:

- **Robot state predicates:** The truth value is queried directly from the robot's state interface. These are always deterministically known.
- **Object state predicates:** The truth value is determined by evaluating the corresponding spatial relationship in the scene representation's spatial relations data.
- **Checked predicates:** The truth value is always FALSE. This is invariant — checked predicates are never initialized to TRUE regardless of any prior knowledge, because the sensing coverage guarantee requires runtime verification.
- **Sensed and external predicates:** The truth value is always FALSE (conservative default). This forces the planner to include sensing actions before relying on these properties, maintaining the sensing coverage guarantee.

---

### L5-V1: State Completeness

**What must hold:** Every applicable predicate-entity combination must have a truth value assigned in the initial state. No combinations may be missing.

**How to check:** Compute the set of all expected predicate-entity combinations (the same enumeration used during construction). Verify that the initial state contains an entry for every expected combination. If using the closed-world assumption, missing entries default to FALSE; otherwise, missing entries are flagged.

**On failure:** AUTO-REPAIR. For any missing combinations, add them to the initial state with value FALSE (conservative default). Since L5 is automated, missing entries indicate a gap in the enumeration logic rather than an LLM error.

---

### L5-V2: Initialization Rule Compliance

**What must hold:** Every fact in the initial state must follow the correct initialization rule for its predicate's type classification. Specifically: all checked predicates must be FALSE, and all sensed and external predicates must be FALSE (conservative default).

**How to check:** For each fact in the initial state, look up the predicate's type classification from the L2 vocabulary. If the classification is "checked" and the value is not FALSE, the fact violates the rule. If the classification is "sensed" or "external" and the value is not FALSE, the fact violates the rule.

**On failure:** AUTO-REPAIR. Reset any violating facts to FALSE. Log each correction. Since L5 is automated, violations indicate a bug in the construction procedure.

---

### L5-V3: Type Match

**What must hold:** For every fact in the initial state, the entity constants in the argument positions must match the types declared in the predicate definition. An entity identified as an object in the scene representation must only appear in argument positions typed as "object." An entity identified as a surface must only appear in positions typed as "surface" or "location." The robot identifier must only appear in positions typed as "robot."

**How to check:** For each fact, look up the predicate definition from L2. For each argument in the fact, determine the entity's type by checking which section of the scene representation contains it (objects, surfaces, or robot). Verify that this type matches or is compatible with the type declared for that argument position in the predicate definition.

**On failure:** REJECT. This indicates a system bug in the state construction procedure (since L5 is automated). Return the specific type mismatches. Trigger a re-examination of the entity type assignments and re-run state construction.

---

### L5-V4: Consistency with Scene Representation

**What must hold:** Every spatial predicate (such as object-at, on, in, attached-to) that is asserted as TRUE in the initial state must correspond to a spatial relationship actually present in the scene representation's spatial relations data. The initial state must not assert spatial facts that contradict what the perception system observes.

**How to check:** Identify all facts in the initial state whose predicate is classified as a spatial predicate (maintained as a static configuration list). For each such fact that has value TRUE, verify that a matching spatial relation exists in the scene representation. A match requires the same predicate name, the same first entity, and the same second entity.

**On failure:** REJECT. This indicates a bug in the construction procedure. Return the specific inconsistent facts. Trigger a fresh perception update and re-run state construction.

---

### L5-V5: Goal-State Distinctness

**What must hold:** The initial state should not already satisfy all goal predicates. If it does, the task is trivially complete — the planner will produce an empty plan.

**How to check:** For each goal predicate from L1, evaluate it against the initial state. A non-negated goal is satisfied if the corresponding fact is TRUE. A negated goal is satisfied if the corresponding fact is FALSE. If all goals are satisfied, the check triggers.

**On failure:** WARNING (not rejection). Log the warning: "Initial state already satisfies all goals. The task may be trivially complete. Verify that the goal specification and perception are correct." Continue execution — the planner will correctly produce an empty plan if the task is indeed already complete.

---

### L5 Locking

On successful validation (including any auto-repairs), the initial state is frozen. The complete domain (predicate vocabulary from L2, action schemas from L3, initial state from L5, and goal set from L1) is passed to the planner.

---

## Validation Check Summary

**Layer L1 — Goal Specification (4 checks):**

| ID | Condition | Response |
|----|-----------|----------|
| L1-V1 | Goal set is non-empty | REJECT |
| L1-V2 | All goal predicates are syntactically well-formed | REJECT |
| L1-V3 | No contradictory goals under uniqueness constraints | REJECT |
| L1-V4 | All goal arguments reference entities in the scene | REJECT (flag for exploration) |

**Layer L2 — Predicate Vocabulary (6 checks):**

| ID | Condition | Response |
|----|-----------|----------|
| L2-V1 | All goal predicate names are defined | REJECT |
| L2-V2 | All names follow PDDL naming convention | REJECT |
| L2-V3 | Every sensed/external predicate has a checked variant | AUTO-REPAIR |
| L2-V4 | All argument types are in the valid type set | REJECT |
| L2-V5 | No unused predicates (deferred to after L3) | AUTO-REPAIR |
| L2-V6 | All type classifications are valid | REJECT |

**Layer L3 — Action Schemas (4 checks):**

| ID | Condition | Response |
|----|-----------|----------|
| L3-V1 | All predicates in actions exist in vocabulary | REJECT |
| L3-V2 | All variables declared with valid types | REJECT |
| L3-V3 | All checked predicates in preconditions have producing actions | AUTO-REPAIR |
| L3-V4 | All goal predicates reachable via relaxed planning graph | REJECT |

**Layer L4 — Grounding and Skill Specifications (9 checks):**

| ID | Condition | Response |
|----|-----------|----------|
| L4-V1 | Grounding rules exist for all PDDL types | REJECT |
| L4-V2 | Scene has elements for all required categories | REJECT (perception update) |
| L4-V3 | All goal entities present in scene | EXPLORATION |
| L4-V4 | All primitives in sequence exist in library | REJECT (VLM re-decompose) |
| L4-V5 | All semantic parameters valid per schema | REJECT (VLM re-decompose) |
| L4-V6 | All symbolic labels resolve in scene | REJECT (perception update or VLM re-decompose) |
| L4-V7 | All required parameters present | REJECT (VLM re-decompose) |
| L4-V8 | All constraints instantiable after grounding | REJECT |
| L4-V9 | All target poses pass feasibility pre-checks | REJECT (VLM re-decompose with exclusions) |

**Layer L5 — Initial State (5 checks):**

| ID | Condition | Response |
|----|-----------|----------|
| L5-V1 | All predicate-entity combinations assigned | AUTO-REPAIR |
| L5-V2 | Checked and sensed predicates initialized correctly | AUTO-REPAIR |
| L5-V3 | Entity types match predicate argument types | REJECT (system bug) |
| L5-V4 | Spatial facts consistent with scene | REJECT (system bug) |
| L5-V5 | Initial state does not already satisfy all goals | WARNING |

**Auto-Repair Summary (5 procedures, all deterministic, no LLM/VLM required):**

| Triggered By | What Is Repaired |
|-------------|-----------------|
| L2-V3 | Missing checked-variant predicates are generated with matching argument types and "checked" classification |
| L2-V5 | Unused predicates are removed from the vocabulary |
| L3-V3 | Missing sensing actions are generated from predicate templates with appropriate parameters and effects |
| L5-V1 | Missing state facts are added with FALSE as the conservative default value |
| L5-V2 | Checked and sensed predicates incorrectly set to TRUE are reset to FALSE |