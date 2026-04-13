# GTAMP Layer Validation Specification
## Developer Reference: Formatting Requirements and Validation Checks

Version: 1.0
Status: Implementation Specification

---

## Document Purpose

This document specifies every formatting requirement and validation check for each layer of the GTAMP domain generation pipeline. Each check includes: the exact condition tested, the algorithm for testing it, the data structures involved, the failure response (reject vs. auto-repair vs. warn), and the feedback message returned to the LLM on failure.

This is the implementation specification. A developer reading this document should be able to build the complete validation pipeline without referring to any other document.

---

## Prerequisites and Shared Data Structures

### Primitive Library Schema

The primitive library is loaded at system initialization and does not change at runtime.

```python
AMP_LIBRARY = {
    "move_gripper_to_pose": {
        "params": {
            "target": {
                "type": "interaction_point",
                "grounding_type": "interaction_point",
                "required": True
            },
            "grasp_mode": {
                "type": "enum",
                "valid_values": ["top_down", "side"],
                "required": True
            },
            "approach_offset": {
                "type": "float",
                "unit": "meters",
                "required": False,
                "default": 0.05
            }
        },
        "constraints": ["reachable(target)", "collision_free(path_to(target))"]
    },
    "close_gripper": {
        "params": {},
        "constraints": []
    },
    "open_gripper": {
        "params": {},
        "constraints": []
    },
    "push_pull": {
        "params": {
            "surface": {
                "type": "surface_label",
                "grounding_type": "surface",
                "required": True
            },
            "force_direction": {
                "type": "enum",
                "valid_values": ["perpendicular", "parallel"],
                "required": True
            },
            "interaction_type": {
                "type": "enum",
                "valid_values": ["press", "sustained"],
                "required": True
            },
            "articulation_mode": {
                "type": "enum",
                "valid_values": ["linear", "revolute"],
                "required": True
            },
            "hinge_boundary": {
                "type": "surface_boundary_label",
                "grounding_type": "surface_boundary",
                "required": False,
                "required_if": {"articulation_mode": "revolute"}
            }
        },
        "constraints": ["reachable(surface)", "collision_free(path_to(surface))"]
    },
    "twist": {
        "params": {
            "direction": {
                "type": "enum",
                "valid_values": ["cw", "ccw"],
                "required": True
            },
            "angle": {
                "type": "float",
                "unit": "radians",
                "required": False
            }
        },
        "constraints": []
    },
    "retract_gripper": {
        "params": {},
        "constraints": []
    },
    "turn_base": {
        "params": {
            "direction": {
                "type": "enum",
                "valid_values": ["left", "right"],
                "required": True
            },
            "angle": {
                "type": "float",
                "unit": "radians",
                "required": False
            }
        },
        "constraints": []
    },
    "scan_region": {
        "params": {
            "n_observations": {
                "type": "int",
                "required": False,
                "default": 4
            }
        },
        "constraints": [],
        "triggers_perception_update": True
    },
    "observe": {
        "params": {
            "target_region": {
                "type": "region_label",
                "grounding_type": "surface",
                "required": True
            }
        },
        "constraints": [],
        "triggers_perception_update": True
    },
    "move_base_to_pose": {
        "params": {
            "target": {
                "type": "location_label",
                "grounding_type": "location",
                "required": True
            }
        },
        "constraints": ["navigable(target)"],
        "requires_mobile_base": True
    }
}
```

### Scene Representation Schema (Ψ)

```python
Ψ = {
    "objects": {
        "<object_id>": {
            "geometry": {"mesh": ..., "bbox": BBox3D, "point_cloud": ...},
            "pose": SE3,
            "semantic_label": str,
            "class": str,
            "interaction_points": {
                "<label>": {"point": Vec3, "pose": SE3, "confidence": float, "source": str}
            },
            "affordance_regions": {
                "<label>": {"mask": Mask, "confidence": float}
            },
            "part_segments": {
                "<label>": {"mask": Mask, "confidence": float}
            },
            "articulation": {
                "type": str | None,  # "twist", "pull", "revolute", None
                "axis": Vec3 | None,
                "limits": (float, float) | None
            } | None
        }
    },
    "surfaces": {
        "<surface_id>": {
            "geometry": {"plane_eq": Vec4, "boundary_polygon": [Vec3], "normal": Vec3},
            "semantic_label": str,
            "support_region": Mask,
            "spatial_context": [str]  # neighboring object IDs
        }
    },
    "spatial_relations": [
        {"entity1": str, "relation": str, "entity2": str}
    ],
    "robot": {
        "id": str,
        "base_position": Vec3,
        "max_reach": float,
        "workspace_bounds": ConvexHull | Sphere,
        "gripper": {
            "max_aperture": float,
            "current_state": "open" | "closed"
        }
    },
    "last_update": datetime,
    "confidence_scores": {"<element_id>": float}
}
```

### Pre-Defined Grounding Rules

```python
GROUNDING_RULES = {
    "object":   {"target_category": "objects",   "lookup": "Ψ.objects[id]"},
    "location": {"target_category": "surfaces",  "lookup": "Ψ.surfaces[id]"},
    "surface":  {"target_category": "surfaces",  "lookup": "Ψ.surfaces[id]"},
    "robot":    {"target_category": "robot",     "lookup": "Ψ.robot"},
    "region":   {"target_category": "surfaces",  "lookup": "Ψ.surfaces[id]"}
}

VALID_PDDL_TYPES = set(GROUNDING_RULES.keys())
```

### Uniqueness Constraint Registry (for L₁ consistency checking)

```python
UNIQUENESS_CONSTRAINTS = [
    {"predicate": "object-at", "unique_arg_index": 0, "varying_arg_index": 1},
    {"predicate": "robot-at",  "unique_arg_index": 0, "varying_arg_index": 1},
    {"predicate": "holding",   "unique_arg_index": 0, "varying_arg_index": None},
    # Add domain-specific entries as needed
]
```

### PDDL Naming Pattern

```python
PDDL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
PDDL_VARIABLE_PATTERN = re.compile(r"\?[a-z][a-z0-9_-]*")
PDDL_GOAL_PATTERN = re.compile(
    r"^\((?:not\s+)?\([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9_-]*)+\)\)$|"
    r"^\([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9_-]*)+\)$"
)
```

---

## Layer L₁: Goal Specification

### Expected LLM Output Format

```json
{
    "goals": [
        {
            "predicate": "object-at",
            "arguments": ["milk", "counter"],
            "negated": false
        },
        {
            "predicate": "closed",
            "arguments": ["fridge"],
            "negated": false
        }
    ]
}
```

Alternatively accepted as PDDL goal strings:
```
(and (object-at milk counter) (closed fridge))
```

### Internal Representation After Parsing

```python
@dataclass
class GoalPredicate:
    predicate_name: str
    arguments: list[str]
    negated: bool

G: list[GoalPredicate]
```

### Validation Checks

---

#### L1-V1: Non-Emptiness

**Condition:** `len(G) > 0`

**Algorithm:**
```python
def check_non_emptiness(G):
    if len(G) == 0:
        return Failure(
            check="L1-V1",
            message="Goal set is empty. Re-examine the task description "
                    "and extract at least one desired end-state condition."
        )
    return Pass()
```

**Failure response:** REJECT. Return message to LLM for re-generation.

---

#### L1-V2: Well-Formedness

**Condition:** Each goal predicate has a valid predicate name and at least one argument.

**Algorithm:**
```python
def check_well_formedness(G):
    failures = []
    for i, g in enumerate(G):
        # Predicate name check
        if not PDDL_NAME_PATTERN.match(g.predicate_name):
            failures.append(f"Goal {i}: predicate name '{g.predicate_name}' "
                          f"does not match PDDL naming pattern "
                          f"(lowercase-hyphenated, e.g., 'object-at').")
        
        # Argument check
        if len(g.arguments) == 0:
            failures.append(f"Goal {i}: predicate '{g.predicate_name}' "
                          f"has no arguments.")
        
        for j, arg in enumerate(g.arguments):
            if not isinstance(arg, str) or len(arg.strip()) == 0:
                failures.append(f"Goal {i}, argument {j}: "
                              f"empty or non-string argument.")
    
    if failures:
        return Failure(check="L1-V2", messages=failures)
    return Pass()
```

**Failure response:** REJECT. Return specific malformed goals with correction guidance.

---

#### L1-V3: Internal Consistency

**Condition:** No two goal predicates assert contradictory values for a uniqueness-constrained argument.

**Algorithm:**
```python
def check_internal_consistency(G):
    failures = []
    
    for constraint in UNIQUENESS_CONSTRAINTS:
        pred_name = constraint["predicate"]
        unique_idx = constraint["unique_arg_index"]
        varying_idx = constraint["varying_arg_index"]
        
        # Collect all non-negated goals using this predicate
        relevant = [g for g in G 
                    if g.predicate_name == pred_name and not g.negated]
        
        if varying_idx is None:
            # Uniqueness constraint on the predicate itself
            # (e.g., can only hold one object)
            if len(relevant) > 1:
                unique_vals = set(g.arguments[unique_idx] for g in relevant)
                if len(unique_vals) > 1:
                    failures.append(
                        f"Contradiction: predicate '{pred_name}' asserts "
                        f"multiple values for uniqueness-constrained "
                        f"argument: {unique_vals}"
                    )
        else:
            # Group by the unique argument
            groups = {}
            for g in relevant:
                key = g.arguments[unique_idx]
                val = g.arguments[varying_idx]
                if key not in groups:
                    groups[key] = set()
                groups[key].add(val)
            
            for key, vals in groups.items():
                if len(vals) > 1:
                    failures.append(
                        f"Contradiction: '{pred_name}' asserts "
                        f"'{key}' maps to multiple values: {vals}. "
                        f"An entity can only be in one location."
                    )
    
    if failures:
        return Failure(check="L1-V3", messages=failures)
    return Pass()
```

**Failure response:** REJECT. Return specific contradictions to LLM.

---

#### L1-V4: Argument Plausibility

**Condition:** All argument strings in G reference entities that exist in the current scene representation Ψ.

**Algorithm:**
```python
def check_argument_plausibility(G, Ψ):
    known_entities = (
        set(Ψ["objects"].keys()) | 
        set(Ψ["surfaces"].keys()) | 
        {Ψ["robot"]["id"]}
    )
    
    # Also accept semantic labels (not just IDs)
    known_labels = set()
    for obj in Ψ["objects"].values():
        known_labels.add(obj["semantic_label"])
    for surf in Ψ["surfaces"].values():
        known_labels.add(surf["semantic_label"])
    
    all_known = known_entities | known_labels
    
    flagged = []
    for i, g in enumerate(G):
        for j, arg in enumerate(g.arguments):
            if arg not in all_known:
                flagged.append({
                    "goal_index": i,
                    "argument_index": j,
                    "argument_value": arg,
                    "message": f"Entity '{arg}' not found in current scene."
                })
    
    if flagged:
        return Failure(
            check="L1-V4",
            flagged_arguments=flagged,
            known_entities=sorted(all_known),
            message="The following goal arguments reference entities not "
                    "currently observed in the scene. Either the entity "
                    "names are incorrect or the entities need to be "
                    "discovered through exploration. Known entities: "
                    f"{sorted(all_known)}"
        )
    return Pass()
```

**Failure response:** REJECT. Return flagged arguments with list of known entities. The LLM can either correct the entity names or the system can flag these entities for exploration at L₄.

**Note:** Entities that fail this check are candidates for exploration actions. The system should record which goal arguments failed L1-V4 — these become exploration targets at L₄ if the LLM confirms the entity names are correct.

---

### L₁ Auto-Repair

None. All L₁ failures require LLM re-generation with the specific failure message.

### L₁ Locking

On success: G is frozen as an immutable artifact. Downstream layers receive a read-only reference.

```python
L1_artifact = {
    "goals": G,
    "locked": True,
    "validation_timestamp": datetime.now(),
    "exploration_candidates": [args that failed L1-V4 but were confirmed by LLM]
}
```

---

## Layer L₂: Predicate Vocabulary

### Expected LLM Output Format

```json
{
    "predicates": [
        {
            "name": "object-at",
            "arguments": [
                {"name": "obj", "type": "object"},
                {"name": "loc", "type": "location"}
            ],
            "type_classification": "object_state"
        },
        {
            "name": "object-graspable",
            "arguments": [
                {"name": "obj", "type": "object"}
            ],
            "type_classification": "sensed"
        },
        {
            "name": "checked-object-graspable",
            "arguments": [
                {"name": "obj", "type": "object"}
            ],
            "type_classification": "checked"
        }
    ]
}
```

### Internal Representation

```python
@dataclass
class PredicateDefinition:
    name: str
    arguments: list[dict]  # [{"name": str, "type": str}]
    type_classification: str  # "robot_state" | "object_state" | "sensed" | "external" | "checked"

P: dict[str, PredicateDefinition]  # keyed by name
```

### Valid Type Classifications

```python
VALID_TYPE_CLASSIFICATIONS = {
    "robot_state",    # Directly known from robot state (e.g., robot-at, gripper-empty)
    "object_state",   # Observable from perception (e.g., object-at, on)
    "sensed",         # Requires active sensing to verify (e.g., object-graspable)
    "external",       # External world property, uncertain (e.g., door-locked)
    "checked"         # Tracks whether sensing has occurred (e.g., checked-object-graspable)
}
```

### Validation Checks

---

#### L2-V1: Goal Coverage

**Condition:** Every predicate name used in G exists in P.

**Algorithm:**
```python
def check_goal_coverage(G, P):
    goal_pred_names = {g.predicate_name for g in G}
    P_names = set(P.keys())
    missing = goal_pred_names - P_names
    
    if missing:
        return Failure(
            check="L2-V1",
            message=f"The following predicates appear in the goal but are "
                    f"not defined in the predicate vocabulary: {missing}. "
                    f"Add definitions for these predicates.",
            missing_predicates=sorted(missing)
        )
    return Pass()
```

**Failure response:** REJECT. Return missing predicate names.

---

#### L2-V2: Naming Compliance

**Condition:** Every predicate name and argument type name matches the PDDL naming pattern.

**Algorithm:**
```python
def check_naming_compliance(P):
    failures = []
    
    for pred_name, pred_def in P.items():
        if not PDDL_NAME_PATTERN.match(pred_name):
            failures.append(
                f"Predicate name '{pred_name}' does not match PDDL "
                f"naming pattern (lowercase-hyphenated)."
            )
        
        for arg in pred_def.arguments:
            if not PDDL_NAME_PATTERN.match(arg["type"]):
                failures.append(
                    f"Predicate '{pred_name}', argument '{arg['name']}': "
                    f"type '{arg['type']}' does not match naming pattern."
                )
    
    if failures:
        return Failure(check="L2-V2", messages=failures)
    return Pass()
```

**Failure response:** REJECT. Return specific naming violations.

---

#### L2-V3: Checked-Variant Completeness

**Condition:** For every predicate with type_classification in {"sensed", "external"}, a corresponding predicate named `checked-{name}` exists with type_classification "checked" and identical argument types.

**Algorithm:**
```python
def check_checked_variant_completeness(P):
    sensed_preds = {
        name: pred for name, pred in P.items()
        if pred.type_classification in ("sensed", "external")
    }
    
    missing = []
    type_mismatches = []
    
    for name, pred in sensed_preds.items():
        checked_name = f"checked-{name}"
        
        if checked_name not in P:
            missing.append({
                "base_predicate": name,
                "expected_checked": checked_name,
                "arg_types": [a["type"] for a in pred.arguments]
            })
        else:
            checked_pred = P[checked_name]
            # Verify argument types match
            base_types = [a["type"] for a in pred.arguments]
            checked_types = [a["type"] for a in checked_pred.arguments]
            if base_types != checked_types:
                type_mismatches.append({
                    "base": name,
                    "checked": checked_name,
                    "base_types": base_types,
                    "checked_types": checked_types
                })
            # Verify classification
            if checked_pred.type_classification != "checked":
                type_mismatches.append({
                    "predicate": checked_name,
                    "expected_classification": "checked",
                    "actual_classification": checked_pred.type_classification
                })
    
    if type_mismatches:
        return Failure(check="L2-V3", 
                      message="Type mismatches in checked variants",
                      mismatches=type_mismatches)
    
    if missing:
        return AutoRepair(check="L2-V3", repairs=missing)
    
    return Pass()
```

**Failure response (type mismatch):** REJECT. Return mismatches for LLM correction.

**Failure response (missing):** AUTO-REPAIR. Generate missing checked variants:

```python
def auto_repair_checked_variants(missing, P):
    for entry in missing:
        checked_pred = PredicateDefinition(
            name=entry["expected_checked"],
            arguments=[
                {"name": a["name"], "type": a["type"]}
                for a in P[entry["base_predicate"]].arguments
            ],
            type_classification="checked"
        )
        P[entry["expected_checked"]] = checked_pred
    return P
```

---

#### L2-V4: Type Consistency

**Condition:** No argument name is assigned conflicting types across different predicates, and all argument types are in VALID_PDDL_TYPES.

**Algorithm:**
```python
def check_type_consistency(P):
    # Check all types are valid
    invalid_types = []
    for name, pred in P.items():
        for arg in pred.arguments:
            if arg["type"] not in VALID_PDDL_TYPES:
                invalid_types.append({
                    "predicate": name,
                    "argument": arg["name"],
                    "type": arg["type"],
                    "valid_types": sorted(VALID_PDDL_TYPES)
                })
    
    if invalid_types:
        return Failure(
            check="L2-V4",
            message="The following argument types are not recognized. "
                    f"Valid types: {sorted(VALID_PDDL_TYPES)}",
            invalid_types=invalid_types
        )
    
    return Pass()
```

**Failure response:** REJECT. Return invalid types with valid type list.

---

#### L2-V5: Minimality (Deferred)

**Condition:** Every predicate in P is referenced by at least one action or goal.

**THIS CHECK IS NOT RUN AT L₂.** It is deferred until after L₃ validation completes.

**Algorithm (run after L₃):**
```python
def check_minimality_deferred(P, G, A):
    used_in_goals = {g.predicate_name for g in G}
    used_in_actions = set()
    for action in A.values():
        used_in_actions |= extract_predicates(action.preconditions)
        used_in_actions |= extract_predicates(action.effects)
    
    all_used = used_in_goals | used_in_actions
    unused = set(P.keys()) - all_used
    
    if unused:
        # AUTO-REPAIR: Remove unused predicates
        for name in unused:
            del P[name]
        return AutoRepair(
            check="L2-V5",
            message=f"Removed {len(unused)} unused predicates: {sorted(unused)}",
            removed=sorted(unused)
        )
    return Pass()
```

**Failure response:** AUTO-REPAIR. Remove unused predicates silently. This cannot affect planning since unused predicates appear nowhere.

---

#### L2-V6: Type Classification Validity

**Condition:** Every predicate has a type_classification from the valid set.

**Algorithm:**
```python
def check_type_classification_validity(P):
    invalid = []
    for name, pred in P.items():
        if pred.type_classification not in VALID_TYPE_CLASSIFICATIONS:
            invalid.append({
                "predicate": name,
                "classification": pred.type_classification,
                "valid": sorted(VALID_TYPE_CLASSIFICATIONS)
            })
    
    if invalid:
        return Failure(check="L2-V6", invalid=invalid)
    return Pass()
```

**Failure response:** REJECT. Return invalid classifications with valid options.

---

### L₂ Locking

```python
L2_artifact = {
    "predicates": P,  # frozen dict
    "locked": True,
    "validation_timestamp": datetime.now(),
    "auto_repairs_applied": [list of auto-repair records]
}
```

---

## Layer L₃: Action Schema Generation

### Expected LLM Output Format

```json
{
    "actions": [
        {
            "name": "pick",
            "parameters": [
                {"name": "obj", "type": "object"},
                {"name": "loc", "type": "location"}
            ],
            "preconditions": "(and (object-at ?obj ?loc) (gripper-empty) (checked-object-graspable ?obj))",
            "effects": "(and (holding ?obj) (not (object-at ?obj ?loc)) (not (gripper-empty)))"
        }
    ]
}
```

### Internal Representation

```python
@dataclass
class ActionSchema:
    name: str
    parameters: list[dict]  # [{"name": str, "type": str}]
    preconditions: str       # PDDL formula string
    effects: str             # PDDL formula string

A: dict[str, ActionSchema]  # keyed by name
```

### Helper Functions Required

```python
def extract_predicates(pddl_formula: str) -> set[str]:
    """Extract all predicate names from a PDDL formula string.
    Returns set of predicate name strings."""
    # Parse PDDL formula, return all predicate names
    # Must handle: (and ...), (not ...), (or ...), nested formulas
    ...

def extract_positive_effects(pddl_formula: str) -> set[str]:
    """Extract predicate names that appear as positive effects 
    (not wrapped in 'not')."""
    ...

def extract_variables(pddl_formula: str) -> set[str]:
    """Extract all PDDL variable references (?var-name) from formula."""
    return set(PDDL_VARIABLE_PATTERN.findall(pddl_formula))
```

### Validation Checks

---

#### L3-V1: Symbolic Closure

**Condition:** All predicates referenced in any precondition or effect exist in P.

**Algorithm:**
```python
def check_symbolic_closure(A, P):
    P_names = set(P.keys())
    violations = []
    
    for action_name, action in A.items():
        preds_in_pre = extract_predicates(action.preconditions)
        preds_in_eff = extract_predicates(action.effects)
        preds_used = preds_in_pre | preds_in_eff
        
        undefined = preds_used - P_names
        if undefined:
            violations.append({
                "action": action_name,
                "undefined_predicates": sorted(undefined),
                "used_in": {
                    p: ("precondition" if p in preds_in_pre else "") + 
                       (" effect" if p in preds_in_eff else "")
                    for p in undefined
                }
            })
    
    if violations:
        return Failure(
            check="L3-V1",
            message="Actions reference predicates not defined in the "
                    "predicate vocabulary. Either add these predicates "
                    "to L₂ or correct the action definitions.",
            violations=violations,
            available_predicates=sorted(P_names)
        )
    return Pass()
```

**Failure response:** REJECT. Return undefined predicates per action with available predicate list.

---

#### L3-V2: Parameter Consistency

**Condition:** Every variable in preconditions/effects is declared in the action's parameter list with a valid type.

**Algorithm:**
```python
def check_parameter_consistency(A):
    failures = []
    
    for action_name, action in A.items():
        declared = {f"?{p['name']}" for p in action.parameters}
        declared_types = {p["name"]: p["type"] for p in action.parameters}
        
        referenced = (
            extract_variables(action.preconditions) | 
            extract_variables(action.effects)
        )
        
        # Check for undeclared variables
        undeclared = referenced - declared
        if undeclared:
            failures.append({
                "action": action_name,
                "issue": "undeclared_variables",
                "variables": sorted(undeclared),
                "declared_parameters": sorted(declared)
            })
        
        # Check parameter types are valid
        for param in action.parameters:
            if param["type"] not in VALID_PDDL_TYPES:
                failures.append({
                    "action": action_name,
                    "issue": "invalid_parameter_type",
                    "parameter": param["name"],
                    "type": param["type"],
                    "valid_types": sorted(VALID_PDDL_TYPES)
                })
    
    if failures:
        return Failure(check="L3-V2", failures=failures)
    return Pass()
```

**Failure response:** REJECT. Return specific undeclared variables and invalid types per action.

---

#### L3-V3: Sensing Coverage

**Condition:** For every `checked-*` predicate in any precondition, some action in A produces it as a positive effect.

**Algorithm:**
```python
def check_sensing_coverage(A):
    # Collect all checked-* predicates used in preconditions
    checked_in_pre = set()
    for action in A.values():
        preds = extract_predicates(action.preconditions)
        checked_in_pre |= {p for p in preds if p.startswith("checked-")}
    
    # Collect all checked-* predicates produced as positive effects
    produced = set()
    for action in A.values():
        pos_effects = extract_positive_effects(action.effects)
        produced |= {p for p in pos_effects if p.startswith("checked-")}
    
    uncovered = checked_in_pre - produced
    
    if uncovered:
        return AutoRepair(
            check="L3-V3",
            uncovered_predicates=sorted(uncovered),
            message=f"No sensing action produces: {sorted(uncovered)}"
        )
    return Pass()
```

**Failure response:** AUTO-REPAIR. Generate sensing actions for uncovered predicates:

```python
def auto_repair_sensing_actions(uncovered, P, A):
    for checked_pred_name in uncovered:
        # Derive base predicate name
        base_name = checked_pred_name.replace("checked-", "", 1)
        
        if base_name not in P:
            # Cannot auto-repair if base predicate doesn't exist
            raise ValidationError(
                f"Cannot generate sensing action for '{checked_pred_name}': "
                f"base predicate '{base_name}' not found in P."
            )
        
        base_pred = P[base_name]
        
        # Generate sensing action
        sensing_action = ActionSchema(
            name=f"check-{base_name}",
            parameters=[
                {"name": a["name"], "type": a["type"]}
                for a in base_pred.arguments
            ],
            preconditions="(robot-near ?{})".format(
                base_pred.arguments[0]["name"]
            ),
            effects="({} ?{})".format(
                checked_pred_name,
                " ?".join(a["name"] for a in base_pred.arguments)
            )
        )
        
        A[sensing_action.name] = sensing_action
    
    return A
```

**Note:** If `robot-near` is not in P, substitute with an appropriate proximity predicate or use `(true)` as a trivially satisfiable precondition. Log a warning that the auto-generated sensing action has a trivial precondition.

---

#### L3-V4: Goal Achievability

**Condition:** Every goal predicate is producible by some chain of actions (relaxed planning graph reachability).

**Algorithm:**
```python
def check_goal_achievability(G, A, P):
    # Build initial relaxed fact layer
    # Include all predicates that could be true in any initial state
    F = set()
    for pred_name, pred in P.items():
        if pred.type_classification in ("robot_state", "object_state"):
            F.add(pred_name)
        # In relaxed version, add both checked=T and checked=F
        if pred.type_classification == "checked":
            F.add(pred_name)
    
    goal_preds = {g.predicate_name for g in G}
    
    # Expand until fixpoint or all goals covered
    changed = True
    iterations = 0
    max_iterations = len(P) * 100  # Safety bound
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        for action in A.values():
            # Check if relaxed preconditions satisfied
            pre_preds = extract_predicates(action.preconditions)
            if pre_preds.issubset(F):
                # Add all positive effects
                pos_effects = extract_positive_effects(action.effects)
                new_facts = pos_effects - F
                if new_facts:
                    F |= new_facts
                    changed = True
        
        # Check if goals covered
        if goal_preds.issubset(F):
            return Pass()
    
    # Fixpoint reached without covering all goals
    unreachable = goal_preds - F
    
    # Identify which actions could produce each unreachable goal
    # (to give useful feedback)
    no_producer = set()
    for pred in unreachable:
        producers = [
            a.name for a in A.values()
            if pred in extract_positive_effects(a.effects)
        ]
        if not producers:
            no_producer.add(pred)
    
    return Failure(
        check="L3-V4",
        unreachable_goals=sorted(unreachable),
        goals_with_no_producing_action=sorted(no_producer),
        message=f"Goal predicates unreachable: {sorted(unreachable)}. "
                f"Predicates with no producing action at all: "
                f"{sorted(no_producer)}. "
                f"Add actions that produce these predicates or revise "
                f"the goal specification."
    )
```

**Failure response:** REJECT. Return unreachable goals and the specific gap (no producing action, or precondition chain broken).

---

### L₃ Post-Validation: Deferred Minimality Check

After L₃ passes, run L2-V5 (minimality) on P with the now-available A.

### L₃ Locking

```python
L3_artifact = {
    "actions": A,  # frozen dict
    "locked": True,
    "validation_timestamp": datetime.now(),
    "auto_repairs_applied": [list of auto-generated sensing actions]
}
```

---

## Layer L₄: Symbolic Grounding and Skill Specification Generation

L₄ produces two artifacts. Artifact 4a (grounding mappings) is validated by the system. Artifact 4b (skill specifications) is generated by the VLM and validated against the primitive library schema and scene.

### Artifact 4a: Grounding Mappings

Grounding rules are pre-defined (see Prerequisites). Runtime validation checks that grounding can be *applied* to the current scene.

### Artifact 4b: Skill Specification — Expected VLM Output Format

```json
{
    "action_name": "open",
    "target_object": "bottle_1",
    "primitive_sequence": [
        {
            "primitive": "move_gripper_to_pose",
            "semantic_params": {"grasp_mode": "side"},
            "situational_params": {"target": "bottle_cap"}
        },
        {
            "primitive": "close_gripper",
            "semantic_params": {},
            "situational_params": {}
        },
        {
            "primitive": "twist",
            "semantic_params": {"direction": "ccw"},
            "situational_params": {}
        },
        {
            "primitive": "open_gripper",
            "semantic_params": {},
            "situational_params": {}
        },
        {
            "primitive": "retract_gripper",
            "semantic_params": {},
            "situational_params": {}
        }
    ],
    "reasoning": "The bottle has a twist cap. Approach from the side, grasp the cap, twist counter-clockwise to open, then release."
}
```

### Internal Representation

```python
@dataclass
class PrimitiveStep:
    primitive: str
    semantic_params: dict[str, Any]
    situational_params: dict[str, str]  # label references into Ψ

@dataclass
class SkillSpecification:
    action_name: str
    target_object: str
    primitive_sequence: list[PrimitiveStep]
    reasoning: str
    grounded_params: dict | None  # populated after grounding resolution
```

### Validation Checks

---

#### L4-V1: Grounding Completeness

**Condition:** Every PDDL type in the domain has a grounding rule.

**Algorithm:**
```python
def check_grounding_completeness(P, A):
    all_types = set()
    for pred in P.values():
        for arg in pred.arguments:
            all_types.add(arg["type"])
    for action in A.values():
        for param in action.parameters:
            all_types.add(param["type"])
    
    missing = all_types - set(GROUNDING_RULES.keys())
    
    if missing:
        return Failure(
            check="L4-V1",
            message=f"No grounding rule for PDDL types: {sorted(missing)}. "
                    f"Add grounding rules for these types or correct the "
                    f"type definitions in L₂.",
            missing_types=sorted(missing),
            available_rules=sorted(GROUNDING_RULES.keys())
        )
    return Pass()
```

**Failure response:** REJECT (system configuration error — likely requires L₂ correction).

---

#### L4-V2: Scene Element Availability

**Condition:** Ψ contains at least one element of each type required by the grounding rules used in the domain.

**Algorithm:**
```python
def check_scene_element_availability(P, A, Ψ):
    required_categories = set()
    all_types = set()
    for pred in P.values():
        for arg in pred.arguments:
            all_types.add(arg["type"])
    
    missing_categories = []
    for pddl_type in all_types:
        if pddl_type in GROUNDING_RULES:
            category = GROUNDING_RULES[pddl_type]["target_category"]
            
            if category == "objects" and len(Ψ["objects"]) == 0:
                missing_categories.append(("objects", pddl_type))
            elif category == "surfaces" and len(Ψ["surfaces"]) == 0:
                missing_categories.append(("surfaces", pddl_type))
            elif category == "robot" and "robot" not in Ψ:
                missing_categories.append(("robot", pddl_type))
    
    if missing_categories:
        return Failure(
            check="L4-V2",
            message="Scene representation Ψ is missing required elements.",
            missing=missing_categories
        )
    return Pass()
```

**Failure response:** REJECT. Trigger perception update and retry. If still missing, report perception failure.

---

#### L4-V3: Entity Coverage

**Condition:** Every entity referenced in the goal G exists in Ψ.

**Algorithm:**
```python
def check_entity_coverage(G, Ψ):
    known_entities = (
        set(Ψ["objects"].keys()) |
        set(Ψ["surfaces"].keys()) |
        {Ψ["robot"]["id"]}
    )
    # Also check semantic labels
    known_labels = set()
    for obj in Ψ["objects"].values():
        known_labels.add(obj["semantic_label"])
    for surf in Ψ["surfaces"].values():
        known_labels.add(surf["semantic_label"])
    
    all_known = known_entities | known_labels
    
    goal_entities = set()
    for g in G:
        for arg in g.arguments:
            goal_entities.add(arg)
    
    missing = goal_entities - all_known
    
    if missing:
        return ExplorationNeeded(
            check="L4-V3",
            missing_entities=sorted(missing),
            message=f"Goal references entities not in scene: "
                    f"{sorted(missing)}. Exploration required."
        )
    return Pass()
```

**Failure response:** EXPLORATION. Flag missing entities as exploration targets. Generate exploration sub-problem (see exploration mechanism in framework document).

---

#### L4-V4: Primitive Membership

**Condition:** Every primitive in the VLM-generated sequence exists in the primitive library.

**Algorithm:**
```python
def check_primitive_membership(skill_spec):
    unknown = []
    for i, step in enumerate(skill_spec.primitive_sequence):
        if step.primitive not in AMP_LIBRARY:
            unknown.append({
                "index": i,
                "primitive": step.primitive,
                "available": sorted(AMP_LIBRARY.keys())
            })
    
    if unknown:
        return Failure(
            check="L4-V4",
            message="Skill specification contains unknown primitives.",
            unknown_primitives=unknown
        )
    return Pass()
```

**Failure response:** REJECT. Return unknown primitives with available primitive list to VLM for re-decomposition.

---

#### L4-V5: Semantic Parameter Validity

**Condition:** For each primitive, all semantic parameters are valid names with valid values per the library schema.

**Algorithm:**
```python
def check_semantic_parameter_validity(skill_spec):
    failures = []
    
    for i, step in enumerate(skill_spec.primitive_sequence):
        if step.primitive not in AMP_LIBRARY:
            continue  # Caught by L4-V4
        
        schema = AMP_LIBRARY[step.primitive]["params"]
        
        for param_name, param_value in step.semantic_params.items():
            if param_name not in schema:
                failures.append({
                    "step": i,
                    "primitive": step.primitive,
                    "issue": "unknown_parameter",
                    "parameter": param_name,
                    "valid_params": sorted(schema.keys())
                })
                continue
            
            param_spec = schema[param_name]
            
            if param_spec["type"] == "enum":
                if param_value not in param_spec["valid_values"]:
                    failures.append({
                        "step": i,
                        "primitive": step.primitive,
                        "issue": "invalid_value",
                        "parameter": param_name,
                        "value": param_value,
                        "valid_values": param_spec["valid_values"]
                    })
            
            elif param_spec["type"] == "float":
                if not isinstance(param_value, (int, float)):
                    failures.append({
                        "step": i,
                        "primitive": step.primitive,
                        "issue": "wrong_type",
                        "parameter": param_name,
                        "expected": "float",
                        "got": type(param_value).__name__
                    })
            
            elif param_spec["type"] == "int":
                if not isinstance(param_value, int):
                    failures.append({
                        "step": i,
                        "primitive": step.primitive,
                        "issue": "wrong_type",
                        "parameter": param_name,
                        "expected": "int",
                        "got": type(param_value).__name__
                    })
    
    if failures:
        return Failure(check="L4-V5", failures=failures)
    return Pass()
```

**Failure response:** REJECT. Return specific invalid parameters with valid options to VLM for re-decomposition.

---

#### L4-V6: Symbolic Reference Resolution

**Condition:** Every symbolic label in situational parameters resolves to a concrete element in Ψ.

**Algorithm:**
```python
def check_symbolic_reference_resolution(skill_spec, Ψ):
    failures = []
    
    for i, step in enumerate(skill_spec.primitive_sequence):
        if step.primitive not in AMP_LIBRARY:
            continue
        
        schema = AMP_LIBRARY[step.primitive]["params"]
        
        for param_name, label in step.situational_params.items():
            if param_name not in schema:
                failures.append({
                    "step": i, "primitive": step.primitive,
                    "issue": "unknown_situational_param",
                    "parameter": param_name
                })
                continue
            
            param_spec = schema[param_name]
            grounding_type = param_spec.get("grounding_type")
            
            if grounding_type == "interaction_point":
                # Look up in target object's interaction points
                target_obj_id = skill_spec.target_object
                if target_obj_id not in Ψ["objects"]:
                    failures.append({
                        "step": i, "primitive": step.primitive,
                        "issue": "target_object_not_in_scene",
                        "object": target_obj_id
                    })
                    continue
                
                obj = Ψ["objects"][target_obj_id]
                if label not in obj["interaction_points"]:
                    available = sorted(obj["interaction_points"].keys())
                    failures.append({
                        "step": i, "primitive": step.primitive,
                        "issue": "interaction_point_not_found",
                        "label": label,
                        "object": target_obj_id,
                        "available_points": available
                    })
            
            elif grounding_type == "surface":
                if label not in Ψ["surfaces"]:
                    # Also check by semantic label
                    matching = [
                        sid for sid, s in Ψ["surfaces"].items()
                        if s["semantic_label"] == label
                    ]
                    if not matching:
                        available = sorted(Ψ["surfaces"].keys())
                        failures.append({
                            "step": i, "primitive": step.primitive,
                            "issue": "surface_not_found",
                            "label": label,
                            "available_surfaces": available
                        })
            
            elif grounding_type == "location":
                # Check surfaces and named locations
                if (label not in Ψ["surfaces"] and 
                    label not in Ψ["objects"]):
                    failures.append({
                        "step": i, "primitive": step.primitive,
                        "issue": "location_not_found",
                        "label": label
                    })
    
    if failures:
        # Determine repair strategy
        perception_retry = [
            f for f in failures 
            if f["issue"] == "interaction_point_not_found"
        ]
        vlm_retry = [
            f for f in failures 
            if f["issue"] != "interaction_point_not_found"
        ]
        
        return Failure(
            check="L4-V6",
            failures=failures,
            suggested_repair=(
                "perception_update" if perception_retry and not vlm_retry
                else "vlm_redecomposition"
            )
        )
    return Pass()
```

**Failure response (interaction point not found):** Trigger perception update (T2) for the target object. If still missing after update, fall through to VLM re-decomposition with available labels.

**Failure response (surface/location not found):** REJECT. Return available labels to VLM for re-decomposition.

---

#### L4-V7: Required Parameter Completeness

**Condition:** Every required parameter for each primitive has a value assigned.

**Algorithm:**
```python
def check_required_parameter_completeness(skill_spec):
    failures = []
    
    for i, step in enumerate(skill_spec.primitive_sequence):
        if step.primitive not in AMP_LIBRARY:
            continue
        
        schema = AMP_LIBRARY[step.primitive]["params"]
        provided = set(step.semantic_params.keys()) | set(step.situational_params.keys())
        
        for param_name, param_spec in schema.items():
            is_required = param_spec.get("required", False)
            
            # Handle conditional requirements
            if "required_if" in param_spec:
                condition = param_spec["required_if"]
                for cond_param, cond_value in condition.items():
                    actual_value = step.semantic_params.get(cond_param)
                    if actual_value == cond_value:
                        is_required = True
            
            if is_required and param_name not in provided:
                failures.append({
                    "step": i,
                    "primitive": step.primitive,
                    "missing_parameter": param_name,
                    "parameter_spec": param_spec
                })
    
    if failures:
        return Failure(check="L4-V7", failures=failures)
    return Pass()
```

**Failure response:** REJECT. Return missing required parameters per primitive to VLM for re-decomposition.

---

#### L4-V8: Constraint Instantiation

**Condition:** After grounding resolution, all geometric constraints can be instantiated with concrete values.

**Algorithm:**
```python
def check_constraint_instantiation(skill_spec, Ψ):
    failures = []
    
    for i, step in enumerate(skill_spec.primitive_sequence):
        if step.primitive not in AMP_LIBRARY:
            continue
        
        constraint_templates = AMP_LIBRARY[step.primitive].get("constraints", [])
        
        for ct in constraint_templates:
            # Attempt to instantiate with grounded values
            try:
                grounded = instantiate_constraint(
                    ct, step.situational_params, skill_spec.target_object, Ψ
                )
                if grounded is None or any(v is None for v in grounded.values()):
                    failures.append({
                        "step": i,
                        "primitive": step.primitive,
                        "constraint": ct,
                        "issue": "null_values_after_grounding"
                    })
            except GroundingError as e:
                failures.append({
                    "step": i,
                    "primitive": step.primitive,
                    "constraint": ct,
                    "issue": str(e)
                })
    
    if failures:
        return Failure(check="L4-V8", failures=failures)
    return Pass()
```

**Failure response:** REJECT. Return uninstantiable constraints. May indicate perception gap (trigger update) or invalid symbolic references (VLM re-decomposition).

---

#### L4-V9: Scene Feasibility Pre-Check

**Condition:** Lightweight geometric checks pass for all grounded poses.

**Algorithm:**
```python
def check_scene_feasibility(skill_spec, Ψ):
    failures = []
    robot = Ψ["robot"]
    
    for i, step in enumerate(skill_spec.primitive_sequence):
        # Only check primitives that reference target poses
        if step.primitive not in ("move_gripper_to_pose", "move_base_to_pose"):
            continue
        
        target_label = step.situational_params.get("target")
        if target_label is None:
            continue
        
        # Resolve to pose
        target_pose = resolve_to_pose(target_label, skill_spec.target_object, Ψ)
        if target_pose is None:
            continue  # Caught by L4-V6
        
        # Check 1: Workspace containment
        if not robot["workspace_bounds"].contains(target_pose.position):
            failures.append({
                "step": i,
                "primitive": step.primitive,
                "check": "workspace_containment",
                "target": target_label,
                "message": f"Target pose at {target_pose.position} is "
                          f"outside robot workspace bounds."
            })
        
        # Check 2: Gross reachability
        dist = np.linalg.norm(
            target_pose.position - robot["base_position"]
        )
        if dist > robot["max_reach"]:
            failures.append({
                "step": i,
                "primitive": step.primitive,
                "check": "gross_reachability",
                "target": target_label,
                "distance": float(dist),
                "max_reach": robot["max_reach"],
                "message": f"Target at distance {dist:.3f}m exceeds "
                          f"max reach {robot['max_reach']:.3f}m."
            })
        
        # Check 3: Grasp aperture (if this is a grasp approach)
        grasp_mode = step.semantic_params.get("grasp_mode")
        if grasp_mode and skill_spec.target_object in Ψ["objects"]:
            obj = Ψ["objects"][skill_spec.target_object]
            bbox = obj["geometry"]["bbox"]
            max_aperture = robot["gripper"]["max_aperture"]
            
            if grasp_mode == "top_down":
                min_dim = min(bbox.width, bbox.depth)
            elif grasp_mode == "side":
                min_dim = min(bbox.width, bbox.height)
            else:
                min_dim = 0
            
            if min_dim > max_aperture:
                failures.append({
                    "step": i,
                    "primitive": step.primitive,
                    "check": "grasp_aperture",
                    "grasp_mode": grasp_mode,
                    "object_dimension": float(min_dim),
                    "gripper_aperture": max_aperture,
                    "message": f"Object dimension {min_dim:.3f}m exceeds "
                              f"gripper aperture {max_aperture:.3f}m "
                              f"for {grasp_mode} grasp."
                })
    
    if failures:
        return Failure(
            check="L4-V9",
            failures=failures,
            message="Scene feasibility pre-check failed. These are "
                    "guaranteed infeasible — do not attempt motion planning.",
            suggested_repair="vlm_redecomposition"
        )
    return Pass()
```

**Failure response:** REJECT. Return specific geometric violations with explanation. Feed to VLM with the failing constraint excluded (e.g., "top_down grasp is infeasible for this object, try side grasp").

---

### L₄ Validation Execution Order

```python
def validate_L4(G, P, A, Ψ, skill_specs):
    # Artifact 4a: Grounding
    run_check(L4-V1, check_grounding_completeness, P, A)
    run_check(L4-V2, check_scene_element_availability, P, A, Ψ)
    run_check(L4-V3, check_entity_coverage, G, Ψ)
    # L4-V3 may return ExplorationNeeded — handle separately
    
    # Artifact 4b: Skill specifications (per action)
    for spec in skill_specs:
        run_check(L4-V4, check_primitive_membership, spec)
        run_check(L4-V5, check_semantic_parameter_validity, spec)
        run_check(L4-V6, check_symbolic_reference_resolution, spec, Ψ)
        run_check(L4-V7, check_required_parameter_completeness, spec)
        
        # Grounding resolution (resolve labels to poses)
        resolve_grounding(spec, Ψ)
        
        # Post-grounding checks
        run_check(L4-V8, check_constraint_instantiation, spec, Ψ)
        run_check(L4-V9, check_scene_feasibility, spec, Ψ)
```

### L₄ Locking

```python
L4_artifact = {
    "grounding_rules": GROUNDING_RULES,  # static reference
    "skill_specifications": {
        (spec.action_name, spec.target_object): spec
        for spec in skill_specs
    },
    "locked": True,
    "validation_timestamp": datetime.now()
}
```

---

## Layer L₅: Initial State Construction

L₅ is fully automated. No LLM involvement. The system applies grounding rules to Ψ to produce truth values for all predicate instances.

### Output Format

```python
@dataclass
class StateFact:
    predicate: str
    arguments: list[str]  # concrete entity IDs
    value: bool

P_init: list[StateFact]
```

### Construction Procedure

```python
def construct_initial_state(P, Ψ, GROUNDING_RULES):
    P_init = []
    
    for pred_name, pred_def in P.items():
        # Get all entity tuples matching argument types
        entity_tuples = get_entity_tuples(pred_def.arguments, Ψ, GROUNDING_RULES)
        
        for entities in entity_tuples:
            if pred_def.type_classification == "checked":
                value = False  # ALWAYS FALSE
            
            elif pred_def.type_classification in ("sensed", "external"):
                value = False  # Conservative default
            
            elif pred_def.type_classification == "robot_state":
                value = query_robot_state(pred_name, entities, Ψ)
            
            elif pred_def.type_classification == "object_state":
                value = evaluate_spatial_relation(pred_name, entities, Ψ)
            
            else:
                value = False  # Unknown classification → conservative
            
            P_init.append(StateFact(
                predicate=pred_name,
                arguments=list(entities),
                value=value
            ))
    
    return P_init
```

### Validation Checks

---

#### L5-V1: State Completeness

**Condition:** Every applicable predicate-entity combination has a truth value.

**Algorithm:**
```python
def check_state_completeness(P_init, P, Ψ, GROUNDING_RULES):
    expected = set()
    for pred_name, pred_def in P.items():
        entity_tuples = get_entity_tuples(pred_def.arguments, Ψ, GROUNDING_RULES)
        for entities in entity_tuples:
            expected.add((pred_name, tuple(entities)))
    
    assigned = {(f.predicate, tuple(f.arguments)) for f in P_init}
    missing = expected - assigned
    
    if missing:
        return AutoRepair(
            check="L5-V1",
            message=f"Missing {len(missing)} state facts.",
            missing_count=len(missing)
        )
    return Pass()
```

**Failure response:** AUTO-REPAIR. Re-run construction for missing entries using conservative defaults (FALSE).

---

#### L5-V2: Grounding Strategy Compliance

**Condition:** Truth values follow the correct initialization rule per predicate type.

**Algorithm:**
```python
def check_grounding_strategy_compliance(P_init, P):
    violations = []
    
    for fact in P_init:
        pred_def = P.get(fact.predicate)
        if pred_def is None:
            violations.append({
                "fact": fact,
                "issue": "predicate_not_in_vocabulary"
            })
            continue
        
        if pred_def.type_classification == "checked" and fact.value != False:
            violations.append({
                "fact": fact,
                "issue": "checked_predicate_not_false",
                "expected": False,
                "actual": fact.value
            })
        
        if pred_def.type_classification in ("sensed", "external") and fact.value != False:
            violations.append({
                "fact": fact,
                "issue": "sensed_predicate_not_conservative",
                "expected": False,
                "actual": fact.value
            })
    
    if violations:
        return AutoRepair(check="L5-V2", violations=violations)
    return Pass()
```

**Failure response:** AUTO-REPAIR. Reset checked predicates to FALSE, sensed predicates to FALSE.

---

#### L5-V3: Type Match Verification

**Condition:** Entity constants in each fact match the expected types from the predicate definition.

**Algorithm:**
```python
def check_type_match(P_init, P, Ψ):
    violations = []
    
    for fact in P_init:
        pred_def = P.get(fact.predicate)
        if pred_def is None:
            continue
        
        for i, arg in enumerate(fact.arguments):
            expected_type = pred_def.arguments[i]["type"]
            actual_type = get_entity_type(arg, Ψ)
            
            if actual_type is None:
                violations.append({
                    "fact": fact,
                    "argument_index": i,
                    "entity": arg,
                    "issue": "entity_not_found_in_scene"
                })
            elif actual_type != expected_type:
                violations.append({
                    "fact": fact,
                    "argument_index": i,
                    "entity": arg,
                    "expected_type": expected_type,
                    "actual_type": actual_type
                })
    
    if violations:
        return Failure(check="L5-V3", violations=violations)
    return Pass()

def get_entity_type(entity_id, Ψ):
    if entity_id in Ψ["objects"]:
        return "object"
    if entity_id in Ψ["surfaces"]:
        return "surface"
    if entity_id == Ψ["robot"]["id"]:
        return "robot"
    return None
```

**Failure response:** REJECT (system bug). Re-run state construction with corrected entity types.

---

#### L5-V4: Consistency with Ψ

**Condition:** Spatial predicates in P_init are consistent with Ψ.spatial_relations.

**Algorithm:**
```python
SPATIAL_PREDICATES = {"object-at", "on", "in", "attached-to", "above", "below"}

def check_psi_consistency(P_init, Ψ):
    violations = []
    
    for fact in P_init:
        if fact.predicate not in SPATIAL_PREDICATES:
            continue
        
        if fact.value == True:
            # Verify this spatial relation exists in Ψ
            found = any(
                r["entity1"] == fact.arguments[0] and
                r["relation"] == fact.predicate and
                r["entity2"] == fact.arguments[1]
                for r in Ψ["spatial_relations"]
            )
            if not found:
                violations.append({
                    "fact": fact,
                    "issue": "true_fact_not_in_psi",
                    "message": f"P_init asserts ({fact.predicate} "
                              f"{fact.arguments[0]} {fact.arguments[1]}) "
                              f"but this relation is not in Ψ."
                })
    
    if violations:
        return Failure(check="L5-V4", violations=violations)
    return Pass()
```

**Failure response:** REJECT (system bug). Trigger perception update and re-construct state.

---

#### L5-V5: Goal-State Distinctness

**Condition:** The initial state does not already satisfy all goals.

**Algorithm:**
```python
def check_goal_state_distinctness(P_init, G):
    state_lookup = {}
    for fact in P_init:
        key = (fact.predicate, tuple(fact.arguments))
        state_lookup[key] = fact.value
    
    all_satisfied = True
    for g in G:
        key = (g.predicate_name, tuple(g.arguments))
        current_value = state_lookup.get(key, False)
        
        if g.negated:
            if current_value != False:
                all_satisfied = False
                break
        else:
            if current_value != True:
                all_satisfied = False
                break
    
    if all_satisfied:
        return Warning(
            check="L5-V5",
            message="Initial state already satisfies all goals. "
                    "The task may be trivially complete. Verify "
                    "goal specification and perception accuracy."
        )
    return Pass()
```

**Failure response:** WARNING (not rejection). Log and continue. The planner will produce an empty plan, which is correct if the task is indeed already complete.

---

### L₅ Validation Execution Order

```python
def validate_L5(P_init, P, G, Ψ):
    run_check(L5-V1, check_state_completeness, P_init, P, Ψ, GROUNDING_RULES)
    run_check(L5-V2, check_grounding_strategy_compliance, P_init, P)
    run_check(L5-V3, check_type_match, P_init, P, Ψ)
    run_check(L5-V4, check_psi_consistency, P_init, Ψ)
    run_check(L5-V5, check_goal_state_distinctness, P_init, G)
```

### L₅ Locking

```python
L5_artifact = {
    "initial_state": P_init,  # frozen list
    "locked": True,
    "validation_timestamp": datetime.now(),
    "auto_repairs_applied": [list of auto-repair records]
}
```

---

## Validation Summary

### Check Index

| Check | Layer | Condition | Failure Response |
|-------|-------|-----------|-----------------|
| L1-V1 | L₁ | Goal set non-empty | REJECT |
| L1-V2 | L₁ | Goal predicates well-formed | REJECT |
| L1-V3 | L₁ | No goal contradictions | REJECT |
| L1-V4 | L₁ | Goal arguments exist in Ψ | REJECT (flag for exploration) |
| L2-V1 | L₂ | Goal predicates defined in P | REJECT |
| L2-V2 | L₂ | Names match PDDL pattern | REJECT |
| L2-V3 | L₂ | Checked variants complete | AUTO-REPAIR |
| L2-V4 | L₂ | Types in valid set | REJECT |
| L2-V5 | L₂ (deferred) | No unused predicates | AUTO-REPAIR |
| L2-V6 | L₂ | Valid type classifications | REJECT |
| L3-V1 | L₃ | Symbolic closure | REJECT |
| L3-V2 | L₃ | Parameters declared and typed | REJECT |
| L3-V3 | L₃ | Sensing actions exist for all checked predicates | AUTO-REPAIR |
| L3-V4 | L₃ | Goal achievable (relaxed planning graph) | REJECT |
| L4-V1 | L₄ | Grounding rules exist for all types | REJECT |
| L4-V2 | L₄ | Ψ has elements for all required types | REJECT (perception update) |
| L4-V3 | L₄ | Goal entities exist in Ψ | EXPLORATION |
| L4-V4 | L₄ | All primitives in library | REJECT (VLM re-decompose) |
| L4-V5 | L₄ | Semantic params valid per schema | REJECT (VLM re-decompose) |
| L4-V6 | L₄ | Symbolic labels resolve in Ψ | REJECT (perception update or VLM re-decompose) |
| L4-V7 | L₄ | Required params present | REJECT (VLM re-decompose) |
| L4-V8 | L₄ | Constraints instantiable | REJECT |
| L4-V9 | L₄ | Scene feasibility pre-check | REJECT (VLM re-decompose with exclusions) |
| L5-V1 | L₅ | All predicate instances assigned | AUTO-REPAIR |
| L5-V2 | L₅ | Initialization rules followed | AUTO-REPAIR |
| L5-V3 | L₅ | Entity types match | REJECT (system bug) |
| L5-V4 | L₅ | Spatial facts consistent with Ψ | REJECT (system bug) |
| L5-V5 | L₅ | Initial state ≠ goal state | WARNING |

### Auto-Repair Summary

| Check | What is Auto-Repaired | LLM Required? |
|-------|----------------------|---------------|
| L2-V3 | Generate missing `checked-*` predicate variants | No |
| L2-V5 | Remove unused predicates | No |
| L3-V3 | Generate sensing actions from templates | No |
| L5-V1 | Fill missing state facts with conservative defaults | No |
| L5-V2 | Reset checked/sensed predicates to FALSE | No |

All auto-repairs are deterministic and constructive. No LLM or VLM call is required for any auto-repair.

---

## Execution History Log Schema

Every action attempt is recorded regardless of outcome:

```python
@dataclass
class ExecutionHistoryEntry:
    action_name: str
    target_object: str
    skill_specification: SkillSpecification
    outcome: str  # "success" | "failure"
    failure_type: str | None  # "F1" | "F2" | "F3" | "F4" | "F5" | None
    failure_detail: str | None
    psi_before: dict  # Ψ snapshot hash or summary
    psi_after: dict
    new_objects_discovered: list[str]  # entities that appeared in Ψ_after but not Ψ_before
    timestamp: datetime

execution_history: list[ExecutionHistoryEntry]
```

### History Filtering for VLM Prompts

When providing history to the VLM during decomposition or re-decomposition:

```python
def get_relevant_history(action_name, target_object, history, max_entries=10):
    """Filter history entries relevant to current decomposition."""
    relevant = []
    
    # Priority 1: Same action + same object
    for entry in reversed(history):
        if entry.action_name == action_name and entry.target_object == target_object:
            relevant.append(entry)
    
    # Priority 2: Same action + same object class
    target_class = Ψ["objects"].get(target_object, {}).get("class")
    if target_class:
        for entry in reversed(history):
            obj_class = Ψ["objects"].get(entry.target_object, {}).get("class")
            if entry.action_name == action_name and obj_class == target_class:
                if entry not in relevant:
                    relevant.append(entry)
    
    # Priority 3: Same object (different action)
    for entry in reversed(history):
        if entry.target_object == target_object:
            if entry not in relevant:
                relevant.append(entry)
    
    return relevant[:max_entries]
```