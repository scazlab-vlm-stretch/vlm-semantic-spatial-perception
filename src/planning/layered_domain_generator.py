"""
Layered Domain Generator

Implements the 5-layer gated domain generation pipeline described in
gtamp-update-plan.md. Each layer is validated before the next layer runs;
failures trigger targeted regeneration of only that layer.

Layers:
  L1: Goal Specification       — extract grounded goal predicates from NL task
  L2: Predicate Vocabulary     — minimal predicate set covering L1 goals
  L3: Action Schemas           — pure PDDL actions using only L2 predicates
  L4: Grounding Pre-check      — algorithmic scene feasibility (no LLM)
  L5: Initial State            — algorithmic state construction from scene (no LLM)

Usage:
    generator = LayeredDomainGenerator(api_key=os.getenv("GEMINI_API_KEY"))
    artifact = await generator.generate_domain(
        "Put the red block on the blue block",
        observed_objects=[
            {"object_id": "red_block_1", "object_type": "block",
             "affordances": ["graspable", "stackable"], "position_3d": [0.3, 0.0, 0.1]},
            ...
        ]
    )
    task_analysis = artifact.to_task_analysis()
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml
from google import genai
from google.genai import types

from ..utils.prompt_utils import render_prompt_template
from .utils.task_types import (
    L1GoalArtifact,
    L2PredicateArtifact,
    L3ActionArtifact,
    L4GroundingArtifact,
    L5InitialStateArtifact,
    LayeredDomainArtifact,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Any:
    """
    Parse JSON from raw LLM output, tolerating markdown code fences and
    prose before/after the JSON block.
    """
    # 1. Try bare parse first (clean output)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Extract from ```json ... ``` or ``` ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 3. Find the first {...} block in the text (prefer object over array)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No valid JSON found in model output", text, 0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Recognised PDDL object/parameter types — must match the grounding rule type set.
# These are the only types for which grounding rules exist.
VALID_PDDL_TYPES: Set[str] = {
    "object", "location", "surface", "robot", "region",
}

# Uniqueness constraints: predicate_name -> (constrained_arg_index, varying_arg_index)
# If the constrained arg maps to more than one value of the varying arg, it's a contradiction.
UNIQUENESS_CONSTRAINTS: Dict[str, Tuple[int, int]] = {
    "on": (0, 1),
    "object-at": (0, 1),
    "at": (0, 1),
}

# Valid type_classification values for L2 predicate entries
VALID_TYPE_CLASSIFICATIONS: Set[str] = {
    "sensed", "checked", "derived", "action", "robot_state", "object_state", "external",
}

# Pattern for lowercase-hyphenated PDDL names
_PDDL_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")

# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, reusable in tests)
# ---------------------------------------------------------------------------

_LOGICAL_OPS = {"and", "or", "not", "forall", "exists", "when", "imply"}


def extract_predicate_name_from_literal(literal: str) -> Optional[str]:
    """Return the predicate name from a PDDL literal like '(on red_block_1 blue_block_1)'."""
    m = re.match(r"^\(\s*([a-zA-Z][a-zA-Z0-9_-]*)", literal.strip())
    return m.group(1).lower() if m else None


def extract_predicates_from_formula(formula: str) -> Set[str]:
    """Extract all predicate names used in a PDDL formula string."""
    names: Set[str] = set()
    for match in re.finditer(r"\(([a-zA-Z][a-zA-Z0-9_-]*)", formula):
        name = match.group(1).lower()
        if name not in _LOGICAL_OPS:
            names.add(name)
    return names


def extract_variables_from_formula(formula: str) -> Set[str]:
    """Extract all ?variable names used in a PDDL formula."""
    return set(re.findall(r"\?[a-zA-Z][a-zA-Z0-9_-]*", formula))


def extract_predicate_usages_from_formula(formula: str) -> List[Tuple[str, int]]:
    """
    Extract (predicate_name, arg_count) pairs for every atom in a PDDL formula.

    Uses balanced-paren scanning so nested expressions like (and (not (on ?a ?b)))
    are handled correctly.  Logical operators (and, or, not, when, forall, exists,
    imply) are skipped.  Returns one entry per unique (name, arg_count) combination.
    """
    results: List[Tuple[str, int]] = []

    def _scan(text: str) -> None:
        i = 0
        while i < len(text):
            if text[i] == "(":
                # Find matching close-paren
                j = i + 1
                depth = 1
                while j < len(text) and depth > 0:
                    if text[j] == "(":
                        depth += 1
                    elif text[j] == ")":
                        depth -= 1
                    j += 1
                inner = text[i + 1:j - 1].strip()
                tokens = inner.split()
                if tokens:
                    head = tokens[0].lower()
                    if head not in _LOGICAL_OPS:
                        # Count arguments: tokens after the head that are not sub-expressions
                        arg_count = sum(1 for t in tokens[1:] if not t.startswith("("))
                        results.append((head, arg_count))
                    # Recurse into sub-expressions contained in inner
                    _scan(inner)
                i = j
            else:
                i += 1

    _scan(formula)
    return results


def extract_positive_predicates_from_formula(formula: str) -> Set[str]:
    """
    Extract predicate names that appear as POSITIVE effects (not inside `not`).
    Used for the relaxed planning graph — delete effects are ignored.
    """
    # Remove all (not (...)) sub-expressions before extracting
    cleaned = re.sub(r"\(\s*not\s*\([^()]*\)\s*\)", "", formula)
    return extract_predicates_from_formula(cleaned)


def extract_positive_precondition_predicates(formula: str) -> Set[str]:
    """
    Extract predicate names that appear as POSITIVE preconditions (not inside `not`).
    Negated preconditions do not need to be established — they only need to be absent,
    which is trivially satisfied in the relaxed (delete-free) graph.
    """
    cleaned = re.sub(r"\(\s*not\s*\([^()]*\)\s*\)", "", formula)
    return extract_predicates_from_formula(cleaned)


def bfs_reachable_predicates(
    actions: List[Dict],
    initial_predicates: Set[str],
) -> Set[str]:
    """
    Relaxed planning graph BFS (delete-relaxation).

    Correctly models which predicates can become true starting from initial_predicates:
    - An action fires when all POSITIVE precondition predicates are reachable
      (negated preconditions are ignored under delete-relaxation)
    - Only POSITIVE effects are added to the reachable set
      (delete effects are ignored under delete-relaxation)
    - checked-* predicates are NOT in the initial set — they must be produced by
      sensing actions whose own preconditions are satisfied first

    Returns the set of all reachable predicate names.
    """
    reachable = set(initial_predicates)
    changed = True
    while changed:
        changed = False
        for action in actions:
            pre = action.get("precondition", "")
            eff = action.get("effect", "")
            # Only fire if all positive precondition predicates are already reachable
            required = extract_positive_precondition_predicates(pre)
            if required <= reachable:
                # Only add positive effects (delete-relaxation ignores negations)
                new = extract_positive_predicates_from_formula(eff) - reachable
                if new:
                    reachable |= new
                    changed = True
    return reachable


# ---------------------------------------------------------------------------
# LayeredDomainGenerator
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS_PATH = (
    Path(__file__).parent.parent.parent / "config" / "layered_domain_generator_prompts.yaml"
)

_ROBOT_DESCRIPTION = (
    "The robot is a 7-DOF robot arm, with a two-finger rigid gripper as end effector. "
    "The end effector cannot grasp objects <1cm or >15cm. "
    "The robot has a Realsense RGBD camera mounted at the end effector."
)


class LayerValidationError(Exception):
    """Raised when a layer exceeds max retries without passing validation."""
    def __init__(self, layer: str, errors: List[str]) -> None:
        self.layer = layer
        self.errors = errors
        super().__init__(f"Layer {layer} validation failed after max retries: {errors}")


class LayeredDomainGenerator:
    """
    Generates a PDDL domain by running five ordered, validated layers.

    Each layer that relies on LLM output is automatically retried (up to
    `max_layer_retries`) with the validation errors appended to the prompt.
    L4 and L5 are fully algorithmic (no LLM call).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        prompts_config_path: Optional[Path] = None,
        max_layer_retries: int = 2,
        dkb: Optional[Any] = None,
        robot_description: Optional[str] = None,
        llm_client: Optional[Any] = None,  # src.llm_interface.LLMClient
    ) -> None:
        self._llm_client = llm_client
        if llm_client is None:
            self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_layer_retries = max_layer_retries
        self.dkb = dkb
        self.robot_description = robot_description or _ROBOT_DESCRIPTION

        path = Path(prompts_config_path) if prompts_config_path else _DEFAULT_PROMPTS_PATH
        with open(path, "r", encoding="utf-8") as f:
            self._prompts: Dict[str, str] = yaml.safe_load(f) or {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def generate_domain(
        self,
        task: str,
        observed_objects: Optional[List[Dict]] = None,
        image: Optional[Any] = None,
    ) -> LayeredDomainArtifact:
        """
        Run the full L1→L5 pipeline and return a LayeredDomainArtifact.

        Args:
            task: Natural language task description.
            observed_objects: List of object dicts from perception, each with at minimum
                              {"object_id": str, "object_type": str, "affordances": list,
                               "position_3d": list[float]}.
            image: Optional image (currently unused, reserved for future L4 VLM calls).

        Returns:
            LayeredDomainArtifact — bridge to legacy TaskAnalysis via .to_task_analysis().
        """
        scene_objects: List[Dict] = observed_objects or []

        print(f"\n[LayeredDomainGenerator] Generating domain for: '{task}'")
        print(f"  Scene objects: {[o.get('object_id') for o in scene_objects]}")

        # L1 — Goal Specification
        l1 = await self._run_layer_with_retry(
            layer_name="L1",
            run_fn=lambda errs: self._run_l1(task, scene_objects, validation_errors=errs),
            validate_fn=lambda art: self._validate_l1(art, scene_objects),
            repair_fn=None,
        )
        print(f"  [L1] Goals: {l1.goal_predicates}")

        # L2 — Predicate Vocabulary
        l2 = await self._run_layer_with_retry(
            layer_name="L2",
            run_fn=lambda errs: self._run_l2(task, l1, scene_objects, validation_errors=errs),
            validate_fn=lambda art: self._validate_l2(art, l1),
            repair_fn=lambda art: self._repair_l2_auto(art),
        )
        # Always ensure checked-* variants are generated before L3 sees the vocab.
        # _repair_l2_auto is idempotent — safe to call even when L2 passed cleanly.
        l2 = self._repair_l2_auto(l2)
        print(f"  [L2] Predicates: {l2.predicate_signatures}")
        if l2.checked_variants:
            print(f"  [L2] Auto-generated checked variants: {l2.checked_variants}")

        # L3 — Action Schemas
        l3 = await self._run_layer_with_retry(
            layer_name="L3",
            run_fn=lambda errs: self._run_l3(task, l1, l2, validation_errors=errs),
            validate_fn=lambda art: self._validate_l3(art, l2, l1),
            repair_fn=lambda art: self._repair_l3_auto(art, l2),
        )
        print(f"  [L3] Actions: {[a.get('name') for a in l3.actions]}")

        # L2-V5 (deferred): remove predicates unused by any action or goal
        l2 = self._prune_unused_predicates(l2, l3, l1)

        # L4 — Grounding Pre-check (algorithmic)
        l4 = self._run_l4_precheck(l1, l3, scene_objects)
        if l4.warnings:
            for w in l4.warnings:
                print(f"  [L4] Warning: {w}")

        # L5 — Initial State Construction (algorithmic)
        l5 = self._run_l5(l2, l3, scene_objects, l1)
        print(f"  [L5] Initial true literals: {len(l5.true_literals)}, false: {len(l5.false_literals)}")

        artifact = LayeredDomainArtifact(
            l1=l1,
            l2=l2,
            l3=l3,
            l4=l4,
            l5=l5,
            task_description=task,
            scene_objects=scene_objects,
        )

        # Record to DKB if available (non-blocking)
        if self.dkb is not None:
            try:
                self.dkb.record_execution(task, artifact, solver_result=None)
            except Exception:
                pass

        return artifact

    # ------------------------------------------------------------------
    # Generic retry loop
    # ------------------------------------------------------------------

    async def _run_layer_with_retry(
        self,
        layer_name: str,
        run_fn: Callable[[List[str]], Any],
        validate_fn: Callable[[Any], List[str]],
        repair_fn: Optional[Callable[[Any], Any]],
    ) -> Any:
        """
        Run a layer, validate, optionally auto-repair, and retry on validation failure.

        On retry, the validation error list is passed to run_fn so it can inject
        the errors into the <<VALIDATION_ERRORS>> placeholder.
        """
        errors: List[str] = []
        artifact = None

        for attempt in range(self.max_layer_retries + 1):
            artifact = await run_fn(errors)
            # Merge any pre-set errors (e.g. parse failures) with validation errors
            pre_errors = list(getattr(artifact, "validation_errors", None) or [])
            errors = pre_errors + validate_fn(artifact)

            if not errors:
                artifact.validation_errors = []
                return artifact

            # Try auto-repair before next LLM retry
            if repair_fn is not None:
                artifact = repair_fn(artifact)
                errors = validate_fn(artifact)
                if not errors:
                    artifact.validation_errors = []
                    return artifact

            print(f"  [{layer_name}] Attempt {attempt + 1}/{self.max_layer_retries + 1} — "
                  f"validation errors: {errors}")
            artifact.generation_attempts = attempt + 1

        # Record errors on the artifact and return it (don't raise — allow partial results)
        artifact.validation_errors = errors
        print(f"  [{layer_name}] Max retries reached with errors: {errors}")
        return artifact

    # ------------------------------------------------------------------
    # L1: Goal Specification
    # ------------------------------------------------------------------

    async def _run_l1(
        self,
        task: str,
        scene_objects: List[Dict],
        validation_errors: Optional[List[str]] = None,
    ) -> L1GoalArtifact:
        template = self._prompts.get("l1_goal_spec_prompt", "")
        prompt = render_prompt_template(template, {
            "TASK": task,
            "ROBOT_DESCRIPTION": self.robot_description,
            "OBJECTS_JSON": json.dumps(scene_objects, indent=2),
            "VALIDATION_ERRORS": self._format_errors(validation_errors),
        })
        text = ""
        try:
            text = await self._call_llm(prompt)
            data = _extract_json(text)
            return L1GoalArtifact(
                goal_predicates=data.get("goal_predicates", []),
                goal_objects=data.get("goal_objects", []),
                global_predicates=data.get("global_predicates", []),
            )
        except Exception as e:
            print(f"  [L1] Parse error: {e}\n  raw output: {text!r}")
            return L1GoalArtifact(
                validation_errors=[f"JSON parse failure: {e}. Respond with valid JSON only."]
            )

    def _validate_l1(
        self, artifact: L1GoalArtifact, scene_objects: List[Dict]
    ) -> List[str]:
        errors: List[str] = []
        # Build full entity set: object IDs, semantic labels, surface IDs and "robot"
        known_ids: Set[str] = {o.get("object_id", "") for o in scene_objects}
        for o in scene_objects:
            for label_key in ("object_type", "semantic_label", "class_name", "surface_id"):
                lbl = o.get(label_key, "")
                if lbl:
                    known_ids.add(lbl)
        known_ids.discard("")
        known_ids.add("robot")

        # L1-V1: non-empty goal set
        if not artifact.goal_predicates:
            errors.append(
                "goal_predicates is empty. Re-examine the task description and "
                "extract at least one desired end-state condition."
            )
            return errors

        # Full structural regex for a grounded literal: (pred-name arg1 arg2 ...)
        _GROUNDED_LITERAL_RE = re.compile(
            r"^\([a-z][a-z0-9]*(-[a-z0-9]+)*( [a-zA-Z][a-zA-Z0-9_-]*)+\)$"
        )

        # Track uniqueness constraints for contradiction detection (L1-V3)
        # Maps (pred_name, constrained_arg_value) -> varying_arg_value
        uniqueness_seen: Dict[Tuple[str, str], str] = {}

        for i, gp in enumerate(artifact.goal_predicates):
            gp = gp.strip()

            # L1-V2: full structural regex match (primary check)
            if not _GROUNDED_LITERAL_RE.match(gp):
                # Fall through to token checks for better error messages
                if not (gp.startswith("(") and gp.endswith(")")):
                    errors.append(
                        f"Goal {i}: predicate '{gp}' is not a valid PDDL literal (missing parentheses). "
                        f"Pattern must be: (predicate-name arg1 arg2 ...) with lowercase-hyphenated names."
                    )
                    continue

                if "?" in gp:
                    errors.append(
                        f"Goal {i}: predicate contains a variable — goals must be grounded: {gp}"
                    )
                    continue

                tokens = re.findall(r"[^\s()]+", gp)
                if not tokens:
                    errors.append(f"Goal {i}: empty literal")
                    continue

                pred_name = tokens[0]
                args = tokens[1:]

                if not _PDDL_NAME_RE.match(pred_name):
                    errors.append(
                        f"Goal {i}: predicate name '{pred_name}' does not match "
                        f"PDDL naming pattern (lowercase-hyphenated, e.g. 'on', 'hand-empty')"
                    )

                if not args:
                    errors.append(
                        f"Goal {i}: predicate '{pred_name}' has no arguments — "
                        f"grounded goal predicates must reference at least one object. "
                        f"Pattern: (predicate-name arg1 arg2 ...)"
                    )
                    continue

                for j, arg in enumerate(args):
                    if not arg:
                        errors.append(f"Goal {i}, argument {j}: empty argument in '{gp}'")
            else:
                tokens = re.findall(r"[^\s()]+", gp)
                pred_name = tokens[0]
                args = tokens[1:]

            # L1-V3: contradiction detection using UNIQUENESS_CONSTRAINTS registry
            if pred_name in UNIQUENESS_CONSTRAINTS:
                constrained_idx, varying_idx = UNIQUENESS_CONSTRAINTS[pred_name]
                if len(args) > constrained_idx and len(args) > varying_idx:
                    constrained_val = args[constrained_idx]
                    varying_val = args[varying_idx]
                    key = (pred_name, constrained_val)
                    if key in uniqueness_seen:
                        prev_val = uniqueness_seen[key]
                        if prev_val != varying_val:
                            errors.append(
                                f"Contradiction: predicate '{pred_name}' asserts "
                                f"'{constrained_val}' maps to both '{prev_val}' and "
                                f"'{varying_val}' simultaneously. "
                                f"'{pred_name}' uniqueness constraint: argument at position "
                                f"{constrained_idx} can only have one value at position {varying_idx}."
                            )
                    else:
                        uniqueness_seen[key] = varying_val

            # L1-V4: all arguments reference known scene entities.
            # Skip entirely when scene is empty (no objects yet observed).
            if known_ids:
                for arg in args:
                    if arg and arg not in known_ids:
                        errors.append(
                            f"Goal {i}: argument '{arg}' not found in observed scene. "
                            f"Known entities: {sorted(known_ids)}"
                        )

        return errors

    # ------------------------------------------------------------------
    # L2: Predicate Vocabulary
    # ------------------------------------------------------------------

    async def _run_l2(
        self,
        task: str,
        l1: L1GoalArtifact,
        scene_objects: List[Dict],
        validation_errors: Optional[List[str]] = None,
    ) -> L2PredicateArtifact:
        dkb_hints = self._get_dkb_predicate_hints(task)
        template = self._prompts.get("l2_predicate_vocab_prompt", "")
        prompt = render_prompt_template(template, {
            "TASK": task,
            "GOAL_PREDICATES_JSON": json.dumps(l1.goal_predicates, indent=2),
            "OBJECTS_JSON": json.dumps(scene_objects, indent=2),
            "DKB_HINTS": dkb_hints,
            "VALIDATION_ERRORS": self._format_errors(validation_errors),
        })
        text = ""
        try:
            text = await self._call_llm(prompt)
            data = _extract_json(text)
            type_classifications: Dict[str, str] = data.get("type_classifications", {})
            # Derive sensed_predicates from type_classifications if the LLM omitted the list
            explicit_sensed: List[str] = data.get("sensed_predicates", [])
            derived_sensed: List[str] = [
                name for name, cls in type_classifications.items()
                if cls in ("sensed", "external") and name not in explicit_sensed
            ]
            sensed_predicates = explicit_sensed + derived_sensed
            return L2PredicateArtifact(
                predicate_signatures=data.get("predicate_signatures", []),
                sensed_predicates=sensed_predicates,
                checked_variants=[],
                type_classifications=type_classifications,
            )
        except Exception as e:
            print(f"  [L2] Parse error: {e}\n  raw output: {text!r}")
            return L2PredicateArtifact(
                validation_errors=[f"JSON parse failure: {e}. Respond with valid JSON only."]
            )

    def _validate_l2(
        self, artifact: L2PredicateArtifact, l1: L1GoalArtifact
    ) -> List[str]:
        errors: List[str] = []

        # Build name → parameter list mapping from signatures
        sig_params: Dict[str, List[str]] = {}
        for sig in artifact.predicate_signatures:
            name = extract_predicate_name_from_literal(sig)
            if name:
                params = re.findall(r"\?[a-zA-Z][a-zA-Z0-9_-]*(?:\s+-\s+[a-zA-Z][a-zA-Z0-9_-]*)?", sig)
                sig_params[name] = params

        defined_names: Set[str] = set(sig_params.keys())

        # L2-V1: all goal predicate names must be defined in P
        for gp in l1.goal_predicates:
            name = extract_predicate_name_from_literal(gp)
            if name and name not in defined_names:
                errors.append(
                    f"Goal predicate '{name}' from L1 is not defined in predicate vocabulary. "
                    f"Add a definition for it."
                )

        # L2-V2: predicate names AND type names must match lowercase-hyphen pattern
        # Use the raw name from the signature (before lowercasing) to catch uppercase letters
        for sig in artifact.predicate_signatures:
            raw_name_match = re.match(r"^\(\s*([^\s()]+)", sig.strip())
            raw_name = raw_name_match.group(1) if raw_name_match else None
            name = extract_predicate_name_from_literal(sig)  # lowercased version
            if raw_name and not _PDDL_NAME_RE.match(raw_name):
                errors.append(
                    f"Predicate name '{raw_name}' does not match PDDL naming pattern (lowercase-hyphenated)"
                )
            # Extract inline type annotations (?var - type_name) and validate type names
            for type_name in re.findall(r"\?\S+\s+-\s+([a-zA-Z][a-zA-Z0-9_-]*)", sig):
                if not _PDDL_NAME_RE.match(type_name):
                    errors.append(
                        f"Predicate '{name}': type name '{type_name}' does not match "
                        f"PDDL naming pattern (lowercase-hyphenated)"
                    )

        # L2-NEW: reject manually-defined checked-* predicates — they are auto-generated only
        for name in defined_names:
            if name.startswith("checked-"):
                errors.append(
                    f"Predicate '{name}' starts with 'checked-' and appears to be manually defined. "
                    f"checked-* predicates are auto-generated by the system — do NOT include them in "
                    f"predicate_signatures. Remove it and list only the base sensed predicate."
                )

        # L2-NEW: predicates classified as 'checked' must start with 'checked-'
        classifications = getattr(artifact, "type_classifications", {}) or {}
        for pred_name, cls in classifications.items():
            if cls == "checked" and not pred_name.startswith("checked-"):
                errors.append(
                    f"Predicate '{pred_name}' is classified as 'checked' but does not start with "
                    f"'checked-'. Checked predicates must be named 'checked-<base>' (auto-generated)."
                )

        # L2-V3 (type mismatch part): if checked-X exists, its params must match X's params
        # exactly — same arity AND same type annotations in the same order.
        def _extract_typed_params(sig: str) -> List[Tuple[str, str]]:
            """Return list of (?varname, type) pairs from a predicate signature string.
            Variables without a type annotation get type 'object' as the default."""
            pairs: List[Tuple[str, str]] = []
            for m in re.finditer(r"(\?[a-zA-Z][a-zA-Z0-9_-]*)(?:\s+-\s+([a-zA-Z][a-zA-Z0-9_-]*))?", sig):
                var = m.group(1)
                typ = (m.group(2) or "object").lower()
                pairs.append((var, typ))
            return pairs

        for sensed_name in artifact.sensed_predicates:
            checked_name = f"checked-{sensed_name}"
            if checked_name in defined_names and sensed_name in defined_names:
                sensed_sig = next((s for s in artifact.predicate_signatures
                                   if extract_predicate_name_from_literal(s) == sensed_name), "")
                checked_sig = next((s for s in artifact.predicate_signatures
                                    if extract_predicate_name_from_literal(s) == checked_name), "")
                sensed_typed = _extract_typed_params(sensed_sig)
                checked_typed = _extract_typed_params(checked_sig)
                if len(sensed_typed) != len(checked_typed):
                    errors.append(
                        f"L2-V3: Checked variant '{checked_name}' has {len(checked_typed)} parameter(s) "
                        f"but base predicate '{sensed_name}' has {len(sensed_typed)} — arity must match exactly."
                    )
                else:
                    for pos, ((_, st), (_, ct)) in enumerate(zip(sensed_typed, checked_typed)):
                        if st != ct:
                            errors.append(
                                f"L2-V3: Checked variant '{checked_name}' argument {pos} has type "
                                f"'{ct}' but base predicate '{sensed_name}' has type '{st}' — "
                                f"argument types must be identical in the same order."
                            )

        # L2-V4: every argument type must be in the valid PDDL type set
        for sig in artifact.predicate_signatures:
            pred_label = extract_predicate_name_from_literal(sig) or sig[:30]
            for var, type_name in re.findall(r"(\?\S+)\s+-\s+([a-zA-Z][a-zA-Z0-9_-]*)", sig):
                t = type_name.lower()
                if t not in VALID_PDDL_TYPES:
                    errors.append(
                        f"L2-V4: Predicate '{pred_label}' argument '{var}' has invalid type '{type_name}'. "
                        f"Valid types: {sorted(VALID_PDDL_TYPES)}"
                    )

        # Type consistency: check for conflicting type assignments for the same variable name
        var_types: Dict[str, Set[str]] = {}
        for sig in artifact.predicate_signatures:
            for var, type_name in re.findall(r"(\?\S+)\s+-\s+([a-zA-Z][a-zA-Z0-9_-]*)", sig):
                var_types.setdefault(var, set()).add(type_name.lower())
        for var, types_seen in var_types.items():
            if len(types_seen) > 1:
                errors.append(
                    f"Type consistency: argument variable '{var}' has conflicting type assignments "
                    f"across predicates: {sorted(types_seen)}. "
                    f"Each argument name should map to exactly one type."
                )

        # L2-V3b: cross-check — predicates classified as sensed/external must be in sensed_predicates
        classifications = getattr(artifact, "type_classifications", {}) or {}
        for pred_name, cls in classifications.items():
            if cls in ("sensed", "external") and pred_name not in (artifact.sensed_predicates or []):
                errors.append(
                    f"Predicate '{pred_name}' is classified as '{cls}' in type_classifications "
                    f"but is missing from sensed_predicates. Add it to sensed_predicates."
                )

        # L2-V3c: any predicate named 'graspable', 'reachable', 'accessible', 'door-open',
        # 'path-clear' etc. that is NOT classified as sensed/external should be flagged —
        # these are inherently sensed properties
        INHERENTLY_SENSED = {
            "graspable", "reachable", "accessible",
            "object-graspable", "object-reachable", "object-accessible",
            "path-clear", "door-open", "door-locked",
            "container-full", "switch-on", "object-present",
        }
        for pred_name in defined_names:
            if pred_name in INHERENTLY_SENSED:
                pred_cls = classifications.get(pred_name, "")
                if pred_cls not in ("sensed", "external", "checked"):
                    errors.append(
                        f"Predicate '{pred_name}' represents a property that requires active "
                        f"sensing — it must have type_classification 'sensed', not '{pred_cls}'. "
                        f"Also add it to sensed_predicates."
                    )

        # L2-V6: every predicate entry must have a valid type_classification (if provided)
        # type_classifications is an optional dict in the artifact; skip if absent
        classifications = getattr(artifact, "type_classifications", {}) or {}
        for pred_name, classification in classifications.items():
            if classification not in VALID_TYPE_CLASSIFICATIONS:
                errors.append(
                    f"Predicate '{pred_name}' has invalid type_classification '{classification}'. "
                    f"Valid options: {sorted(VALID_TYPE_CLASSIFICATIONS)}"
                )

        return errors

    def _repair_l2_auto(self, artifact: L2PredicateArtifact) -> L2PredicateArtifact:
        """Auto-generate checked-X variants for all sensed predicates.
        Also strips any manually-defined checked-* sigs (they are auto-generated only).
        """
        # Remove any manually-defined checked-* sigs from the LLM output
        manually_checked = [
            sig for sig in artifact.predicate_signatures
            if (extract_predicate_name_from_literal(sig) or "").startswith("checked-")
        ]
        if manually_checked:
            artifact.predicate_signatures = [
                sig for sig in artifact.predicate_signatures if sig not in manually_checked
            ]
            for sig in manually_checked:
                name = extract_predicate_name_from_literal(sig) or sig
                artifact.type_classifications.pop(name, None)
                print(f"  [L2 repair] Removed manually-defined checked variant: {sig}")

        existing_names = {
            extract_predicate_name_from_literal(sig) or ""
            for sig in artifact.predicate_signatures
        }
        new_checked: List[str] = list(artifact.checked_variants)

        for sensed_name in artifact.sensed_predicates:
            checked_name = f"checked-{sensed_name}"
            if checked_name not in existing_names:
                # Find the arity of the base sensed predicate
                base_sig = next(
                    (s for s in artifact.predicate_signatures
                     if (extract_predicate_name_from_literal(s) or "") == sensed_name),
                    None
                )
                if base_sig:
                    # Build checked variant with same typed parameters as the base
                    typed_params = re.findall(
                        r"\?[a-zA-Z][a-zA-Z0-9_-]*(?:\s+-\s+[a-zA-Z][a-zA-Z0-9_-]*)?", base_sig
                    )
                    if typed_params:
                        checked_sig = f"({checked_name} {' '.join(typed_params)})"
                    else:
                        checked_sig = f"({checked_name})"
                else:
                    checked_sig = f"({checked_name} ?obj - object)"

                if checked_sig not in artifact.predicate_signatures:
                    artifact.predicate_signatures.append(checked_sig)
                new_checked.append(checked_sig)

                # Update type_classifications for the new checked variant
                artifact.type_classifications[checked_name] = "checked"
                print(f"  [L2 repair] Auto-generated checked variant: {checked_sig}")

        artifact.checked_variants = new_checked
        return artifact

    # ------------------------------------------------------------------
    # L3: Action Schemas
    # ------------------------------------------------------------------

    async def _run_l3(
        self,
        task: str,
        l1: L1GoalArtifact,
        l2: L2PredicateArtifact,
        validation_errors: Optional[List[str]] = None,
    ) -> L3ActionArtifact:
        dkb_hints = self._get_dkb_action_hints(task)
        template = self._prompts.get("l3_action_schema_prompt", "")
        prompt = render_prompt_template(template, {
            "TASK": task,
            "ROBOT_DESCRIPTION": self.robot_description,
            "GOAL_PREDICATES_JSON": json.dumps(l1.goal_predicates, indent=2),
            "PREDICATE_VOCAB_JSON": json.dumps(l2.predicate_signatures, indent=2),
            "TYPE_CLASSIFICATIONS_JSON": json.dumps(
                getattr(l2, "type_classifications", {}) or {}, indent=2
            ),
            "SENSED_PREDICATES_JSON": json.dumps(
                getattr(l2, "sensed_predicates", []) or [], indent=2
            ),
            "DKB_HINTS": dkb_hints,
            "VALIDATION_ERRORS": self._format_errors(validation_errors),
        })
        text = ""
        try:
            text = await self._call_llm(prompt)
            data = _extract_json(text)
            return L3ActionArtifact(
                actions=data.get("actions", []),
                sensing_actions=data.get("sensing_actions", []),
            )
        except Exception as e:
            print(f"  [L3] Parse error: {e}\n  raw output: {text!r}")
            return L3ActionArtifact(
                validation_errors=[f"JSON parse failure: {e}. Respond with valid JSON only."]
            )

    def _validate_l3(
        self,
        artifact: L3ActionArtifact,
        l2: L2PredicateArtifact,
        l1: L1GoalArtifact,
    ) -> List[str]:
        errors: List[str] = []
        all_actions = artifact.actions + artifact.sensing_actions

        # Empty action set check — must have at least one action
        if not artifact.actions and not artifact.sensing_actions:
            errors.append(
                "No actions were generated. Provide at least one action schema."
            )
            return errors

        # Build vocabulary of defined predicate names + arity map
        vocab: Set[str] = set()
        vocab_arity: Dict[str, int] = {}   # predicate name → number of ?-params
        for sig in l2.predicate_signatures:
            name = extract_predicate_name_from_literal(sig)
            if name:
                vocab.add(name)
                # Count typed parameters (?var or ?var - type)
                params = re.findall(r"\?[a-zA-Z][a-zA-Z0-9_-]*", sig)
                vocab_arity[name] = len(params)

        for action in all_actions:
            aname = action.get("name", "<unnamed>")
            raw_params = action.get("parameters", [])
            pre = action.get("precondition", "")
            eff = action.get("effect", "")

            # Declared variable names — strip PDDL type annotations (?var - type → ?var)
            declared_vars: Set[str] = set()
            for p in raw_params:
                p = str(p)
                var = p.split("-")[0].strip() if "-" in p else p.strip()
                if var.startswith("?"):
                    declared_vars.add(var)

            # L3-V1: symbolic closure — all predicates in pre/eff must be in vocabulary
            # Also check arity: each predicate atom must be used with the correct
            # number of arguments as declared in L2.
            for formula_name, formula in [("precondition", pre), ("effect", eff)]:
                used = extract_predicates_from_formula(formula)
                undefined = used - vocab
                if undefined:
                    errors.append(
                        f"Action '{aname}' {formula_name} uses undefined predicates: "
                        f"{sorted(undefined)}. Available predicates: {sorted(vocab)}"
                    )
                # Arity check for every defined predicate appearing in the formula
                seen_arity_errors: Set[str] = set()
                for pred_name, arg_count in extract_predicate_usages_from_formula(formula):
                    if pred_name in vocab_arity and pred_name not in seen_arity_errors:
                        expected = vocab_arity[pred_name]
                        if arg_count != expected:
                            seen_arity_errors.add(pred_name)
                            errors.append(
                                f"Action '{aname}' {formula_name}: predicate '{pred_name}' "
                                f"used with {arg_count} argument(s) but L2 defines it with "
                                f"{expected}. Fix the formula to match the L2 signature."
                            )

            # L3-V2: parameter consistency — all ?vars in pre/eff must be declared,
            # and each declared parameter must have a type from VALID_PDDL_TYPES.
            for formula_name, formula in [("precondition", pre), ("effect", eff)]:
                used_vars = extract_variables_from_formula(formula)
                undeclared = used_vars - declared_vars
                if undeclared:
                    errors.append(
                        f"Action '{aname}' {formula_name} uses undeclared variables: "
                        f"{sorted(undeclared)}"
                    )

            # L3-V2 (type validity extension): check each declared parameter's type annotation
            for p in raw_params:
                p_str = str(p).strip()
                # Extract type annotation: "?var - type"
                type_match = re.search(r"\?\S+\s+-\s+([a-zA-Z][a-zA-Z0-9_-]*)", p_str)
                if type_match:
                    param_type = type_match.group(1).lower()
                    if param_type not in VALID_PDDL_TYPES:
                        errors.append(
                            f"Action '{aname}' parameter '{p_str}' has invalid type '{param_type}'. "
                            f"Valid types: {sorted(VALID_PDDL_TYPES)}"
                        )

        # L3-NEW: regular (non-sensing) actions must NOT set checked-* predicates in effects.
        # Only check-X sensing actions are permitted to set checked-* predicates.
        sensing_names: Set[str] = {a.get("name", "") for a in artifact.sensing_actions}
        for action in artifact.actions:  # only regular actions
            aname = action.get("name", "")
            eff_preds = extract_predicates_from_formula(action.get("effect", ""))
            bad_checked = {p for p in eff_preds if p.startswith("checked-")}
            if bad_checked:
                errors.append(
                    f"Regular action '{aname}' sets checked-* predicates in its effects: "
                    f"{sorted(bad_checked)}. Only sensing actions (check-X) may set checked-* "
                    f"predicates. Remove these from '{aname}' effects."
                )

        # L3-V3: sensing coverage — every checked-* in any precondition must have a producer.
        # Also validate: actions named check-X must set checked-X in effects (not just X).
        for action in all_actions:
            aname = action.get("name", "")
            eff = action.get("effect", "")
            if aname.startswith("check-"):
                expected_checked = f"checked-{aname[len('check-'):]}"
                effect_preds = extract_predicates_from_formula(eff)
                if expected_checked not in effect_preds:
                    errors.append(
                        f"Sensing action '{aname}' must set '{expected_checked}' to TRUE in its "
                        f"effects (found effects: {sorted(effect_preds)}). "
                        f"Sensing actions track verification via checked-* predicates, "
                        f"not the base predicate. "
                        f"Example: check-graspable effect must be '(checked-graspable ?obj)', "
                        f"not '(graspable ?obj)'."
                    )

        produced: Set[str] = set()
        for action in all_actions:
            produced |= extract_predicates_from_formula(action.get("effect", ""))

        # Collect all checked-* predicates that must exist in vocab (from sensed_predicates)
        required_checked: Set[str] = {
            f"checked-{s}" for s in (l2.sensed_predicates or [])
        }
        # Also collect checked-* from vocab itself
        required_checked |= {n for n in vocab if n.startswith("checked-")}

        uncovered_checked: Set[str] = set()
        for pred in required_checked:
            if pred not in produced:
                uncovered_checked.add(pred)
        # Also check any checked-* in preconditions
        for action in all_actions:
            for pred in extract_predicates_from_formula(action.get("precondition", "")):
                if pred.startswith("checked-") and pred not in produced:
                    uncovered_checked.add(pred)

        if uncovered_checked:
            errors.append(
                f"No action produces the following checked predicates: {sorted(uncovered_checked)}. "
                f"For each 'checked-X', add a sensing action 'check-X' with "
                f"effect '(checked-X ?obj)'. "
                f"Sensed predicates requiring coverage: {sorted(l2.sensed_predicates or [])}"
            )

        # L3-V4: goal achievability via relaxed planning graph (delete-relaxation BFS).
        #
        # Initial fact layer F0 contains ONLY predicates that can be true before any action:
        #   - robot_state predicates  (known from robot kinematics/sensors directly)
        #   - object_state predicates (observable from perception — on, at, above, etc.)
        #   - zero-param predicates   (hand-empty, arm-at-home from global_predicates)
        #
        # NOT in F0:
        #   - sensed/external predicates  (conservative FALSE until sensing action runs)
        #   - checked-* predicates        (always FALSE until check-X action produces them)
        #
        # The BFS then propagates: sensing actions fire when their preconditions are met,
        # producing checked-* predicates, which enable actions that require them.
        classifications = getattr(l2, "type_classifications", {}) or {}

        initial_fact_layer: Set[str] = set()
        # robot_state and object_state predicates (grounded from perception/kinematics)
        for pname, cls in classifications.items():
            if cls in ("robot_state", "object_state"):
                initial_fact_layer.add(pname)
        # Zero-param predicates declared in global_predicates (always true at start)
        for p in l1.global_predicates:
            pname = p.strip("() ")
            if pname:
                initial_fact_layer.add(pname)
        # Zero-param predicates in vocabulary with no parameters (e.g. hand-empty)
        for sig in l2.predicate_signatures:
            if "?" not in sig:
                pname = extract_predicate_name_from_literal(sig)
                if pname and not pname.startswith("checked-"):
                    cls = classifications.get(pname, "")
                    if cls not in ("sensed", "external", "checked"):
                        initial_fact_layer.add(pname)

        reachable = bfs_reachable_predicates(all_actions, initial_fact_layer)

        # Diagnostic: report what the BFS found
        unreachable_vocab = vocab - reachable - initial_fact_layer
        if unreachable_vocab:
            print(f"  [L3-V4] BFS: reachable={sorted(reachable)}, "
                  f"unreachable_from_vocab={sorted(unreachable_vocab)}")

        unreachable_goals: List[str] = []
        no_producer: List[str] = []
        broken_chain: List[str] = []  # has a producer but its preconditions are never satisfiable

        for gp in l1.goal_predicates:
            name = extract_predicate_name_from_literal(gp)
            if name and name not in reachable:
                unreachable_goals.append(name)
                # Find all actions that produce this predicate
                producers = [
                    a for a in all_actions
                    if name in extract_positive_predicates_from_formula(a.get("effect", ""))
                ]
                if not producers:
                    no_producer.append(name)
                else:
                    # Producer exists but its preconditions are never satisfiable
                    for prod in producers:
                        req = extract_positive_precondition_predicates(prod.get("precondition", ""))
                        missing_pre = req - reachable
                        if missing_pre:
                            broken_chain.append(
                                f"action '{prod.get('name')}' produces '{name}' but requires "
                                f"{sorted(missing_pre)} which are never reachable"
                            )

        if unreachable_goals:
            msg = (
                f"Goal predicates unreachable via relaxed planning graph: {sorted(unreachable_goals)}. "
            )
            if no_producer:
                msg += f"No action produces: {sorted(no_producer)}. Add an action with these in its effects. "
            if broken_chain:
                msg += f"Broken precondition chains: {broken_chain}. "
            msg += f"Initial reachable set: {sorted(initial_fact_layer)}."
            errors.append(msg)

        return errors

    def _repair_l3_auto(
        self, artifact: L3ActionArtifact, l2: L2PredicateArtifact
    ) -> L3ActionArtifact:
        """
        Auto-generate sensing actions for any uncovered checked-* predicates.
        Covers two cases:
          1. checked-* in preconditions with no producing action
          2. sensed predicates in l2.sensed_predicates with no check-X action
        Also fixes existing check-X actions whose effect sets X instead of checked-X.
        """
        vocab_names: Set[str] = {
            extract_predicate_name_from_literal(sig) or "" for sig in l2.predicate_signatures
        }
        existing_action_names: Set[str] = {
            a.get("name", "") for a in artifact.actions + artifact.sensing_actions
        }

        # Fix broken check-X actions: effect must set checked-X, not X
        for action in artifact.sensing_actions:
            aname = action.get("name", "")
            if aname.startswith("check-"):
                base = aname[len("check-"):]
                checked_pred = f"checked-{base}"
                eff = action.get("effect", "")
                effect_preds = extract_predicates_from_formula(eff)
                if checked_pred not in effect_preds:
                    # Find arity from vocab
                    base_sig = next(
                        (s for s in l2.predicate_signatures
                         if (extract_predicate_name_from_literal(s) or "") == base),
                        None
                    )
                    params = re.findall(r"\?[a-zA-Z][a-zA-Z0-9_-]*", base_sig) if base_sig else ["?obj"]
                    param_str = " ".join(params)
                    action["effect"] = f"({checked_pred} {param_str})" if params else f"({checked_pred})"
                    print(f"  [L3-repair] Fixed '{aname}' effect to set '{checked_pred}' (was: {eff!r})")

        all_actions = artifact.actions + artifact.sensing_actions
        produced: Set[str] = set()
        for action in all_actions:
            produced |= extract_predicates_from_formula(action.get("effect", ""))

        # Collect all checked-* predicates that need coverage
        needs_coverage: Set[str] = set()
        # From sensed_predicates list
        for sname in (l2.sensed_predicates or []):
            needs_coverage.add(f"checked-{sname}")
        # From vocab
        for n in vocab_names:
            if n.startswith("checked-"):
                needs_coverage.add(n)
        # From preconditions
        for action in all_actions:
            for pred in extract_predicates_from_formula(action.get("precondition", "")):
                if pred.startswith("checked-"):
                    needs_coverage.add(pred)

        for checked_pred in needs_coverage:
            if checked_pred in produced:
                continue
            base = checked_pred[len("checked-"):]
            action_name = f"check-{base}"
            if action_name in existing_action_names:
                continue
            # Find typed params from vocabulary — base predicate MUST exist (spec L3-V3)
            base_sig = next(
                (s for s in l2.predicate_signatures
                 if (extract_predicate_name_from_literal(s) or "") == base),
                None
            )
            if not base_sig:
                # Spec: if base predicate N does not exist in L2 vocabulary, REJECT
                print(
                    f"  [L3-V3 ERROR] Cannot auto-generate sensing action for '{checked_pred}': "
                    f"base predicate '{base}' is not defined in L2 vocabulary. "
                    f"This checked predicate references a non-existent base predicate."
                )
                # Inject this as a validation error so the retry loop catches it
                artifact.validation_errors = list(getattr(artifact, "validation_errors", None) or [])
                artifact.validation_errors.append(
                    f"L3-V3: Cannot generate sensing action 'check-{base}' — base predicate "
                    f"'{base}' is not defined in the L2 predicate vocabulary. "
                    f"Either define '{base}' in predicate_signatures or remove references to "
                    f"'checked-{base}' from all action preconditions."
                )
                continue
            typed_params = re.findall(
                r"\?[a-zA-Z][a-zA-Z0-9_-]*(?:\s+-\s+[a-zA-Z][a-zA-Z0-9_-]*)?", base_sig
            )
            param_names = [p.split("-")[0].strip() if "-" in p else p.strip() for p in typed_params]
            param_str = " ".join(param_names)
            sensing = {
                "name": action_name,
                "parameters": typed_params,
                "precondition": "(and)",
                "effect": f"({checked_pred} {param_str})" if param_names else f"({checked_pred})",
            }
            artifact.sensing_actions.append(sensing)
            existing_action_names.add(action_name)
            produced.add(checked_pred)
            # CRITICAL: ensure checked-* predicate is in l2 vocabulary so it appears in :predicates
            existing_l2_names = {
                extract_predicate_name_from_literal(s) or "" for s in l2.predicate_signatures
            }
            if checked_pred not in existing_l2_names:
                checked_sig = f"({checked_pred} {' '.join(typed_params)})" if typed_params else f"({checked_pred})"
                l2.predicate_signatures.append(checked_sig)
                l2.type_classifications[checked_pred] = "checked"
                if checked_pred not in l2.checked_variants:
                    l2.checked_variants.append(checked_sig)
                print(f"  [L3-repair] Added '{checked_pred}' to L2 vocabulary: {checked_sig}")
            print(f"  [L3-repair] Auto-generated sensing action '{action_name}' with effect '({checked_pred} {param_str})'")
        return artifact

    def _prune_unused_predicates(
        self,
        l2: L2PredicateArtifact,
        l3: L3ActionArtifact,
        l1: L1GoalArtifact,
    ) -> L2PredicateArtifact:
        """
        L2-V5 (deferred): remove predicates from L2 that are not referenced by
        any action formula or goal predicate. Runs after L3 is valid.
        Also prunes type_classifications, sensed_predicates, and checked_variants.
        """
        used: Set[str] = set()

        # Predicates referenced in goals
        for gp in l1.goal_predicates:
            name = extract_predicate_name_from_literal(gp)
            if name:
                used.add(name)

        # Predicates referenced in action pre/effects
        for action in l3.actions + l3.sensing_actions:
            used |= extract_predicates_from_formula(action.get("precondition", ""))
            used |= extract_predicates_from_formula(action.get("effect", ""))

        unused = [
            sig for sig in l2.predicate_signatures
            if (extract_predicate_name_from_literal(sig) or "") not in used
        ]
        if unused:
            unused_names = [extract_predicate_name_from_literal(s) for s in unused]
            print(f"  [L2-V5] Removed {len(unused)} unused predicates: {sorted(unused_names)}")
            l2.predicate_signatures = [
                sig for sig in l2.predicate_signatures if sig not in unused
            ]
            # Prune type_classifications to remove entries for pruned predicates
            l2.type_classifications = {
                k: v for k, v in l2.type_classifications.items() if k in used
            }
            # Prune sensed_predicates list
            l2.sensed_predicates = [
                sp for sp in l2.sensed_predicates if sp in used
            ]
            # Prune checked_variants list
            l2.checked_variants = [
                cv for cv in l2.checked_variants
                if (extract_predicate_name_from_literal(cv) or "") in used
            ]

        return l2

    # ------------------------------------------------------------------
    # L4: Grounding Pre-check (algorithmic, no LLM)
    # ------------------------------------------------------------------

    def _run_l4_precheck(
        self,
        l1: L1GoalArtifact,
        l3: L3ActionArtifact,
        scene_objects: List[Dict],
    ) -> L4GroundingArtifact:
        """
        Algorithmic feasibility checks (non-blocking — produces warnings, not errors).

        L4-V1: every PDDL type referenced in action parameters has at least one matching
               object in the scene (by affordance mapping).
        L4-V2: the scene contains at least one element for each type required by the domain.
        L4-V3: every entity in goal_objects exists in the scene.
        """
        warnings: List[str] = []
        bindings: Dict[str, str] = {}

        all_actions = l3.actions + l3.sensing_actions
        obj_ids = [o.get("object_id", "") for o in scene_objects]
        affordance_index: Dict[str, List[str]] = {}
        for obj in scene_objects:
            for aff in obj.get("affordances", []):
                affordance_index.setdefault(aff, []).append(obj.get("object_id", ""))

        graspable_ids = affordance_index.get("graspable", [])

        # Collect all PDDL types used in action parameters (from ?var - type annotations)
        domain_types: Set[str] = set()
        for action in all_actions:
            for param in action.get("parameters", []):
                if isinstance(param, str) and "-" in param:
                    type_part = param.split("-", 1)[1].strip()
                    domain_types.add(type_part.lower())

        # Grounding rule table: maps PDDL type → which scene element category satisfies it.
        # "object" and "location" are satisfied by any object.
        # "surface" is satisfied by objects with object_type="surface" or affordance "support_surface".
        # "robot" is satisfied by the implicit robot entity.
        # "region" is satisfied by objects with object_type containing "region".
        def _type_has_scene_element(dtype: str, objects: List[Dict]) -> bool:
            if dtype in ("object", "location"):
                return bool(objects)
            if dtype == "robot":
                return True  # robot is always present
            if dtype == "surface":
                return any(
                    o.get("object_type", "") == "surface" or
                    "support_surface" in o.get("affordances", [])
                    for o in objects
                )
            if dtype == "region":
                return any("region" in o.get("object_type", "").lower() for o in objects)
            # Fallback: check object_type substring or affordance match
            return any(
                dtype in o.get("object_type", "").lower() or
                dtype in [a.lower() for a in o.get("affordances", [])]
                for o in objects
            )

        # L4-V1: every domain type must have at least one matching scene element
        for dtype in domain_types:
            if not _type_has_scene_element(dtype, scene_objects):
                warnings.append(
                    f"L4-V1: No scene element found for PDDL type '{dtype}'. "
                    f"Grounding rule requires at least one object satisfying this type. "
                    f"Available object types: {sorted({o.get('object_type','') for o in scene_objects})}"
                )

        # L4-V2: required scene element categories must be non-empty.
        # Check each category that is actually referenced by the domain types.
        required_categories: Set[str] = set()
        for dtype in domain_types:
            if dtype in ("object", "location"):
                required_categories.add("objects")
            elif dtype == "surface":
                required_categories.add("surfaces")
            elif dtype == "robot":
                required_categories.add("robot")
            elif dtype == "region":
                required_categories.add("regions")

        if "objects" in required_categories and not obj_ids:
            warnings.append(
                "L4-V2: Domain requires objects but scene has none. "
                "Trigger a perception update and retry."
            )
        if "surfaces" in required_categories and not any(
            o.get("object_type") == "surface" or "support_surface" in o.get("affordances", [])
            for o in scene_objects
        ):
            warnings.append(
                "L4-V2: Domain uses 'surface' type but no surface objects detected in scene. "
                "Ensure a support surface is visible and trigger a perception update."
            )

        # Parameterized actions need at least one candidate object to bind to
        for action in all_actions:
            aname = action.get("name", "")
            params = action.get("parameters", [])
            if params and obj_ids:
                candidates = graspable_ids or obj_ids
                bindings[aname] = candidates[0]

        # L4-V3: every goal entity must exist in scene (exploration flag, non-blocking)
        for goal_obj in l1.goal_objects:
            if goal_obj not in obj_ids and obj_ids:
                warnings.append(
                    f"L4-V3: Goal references entity '{goal_obj}' not found in observed scene. "
                    f"Exploration may be required to locate this object. "
                    f"Known objects: {sorted(obj_ids)}"
                )

        return L4GroundingArtifact(object_bindings=bindings, warnings=warnings)

    # ------------------------------------------------------------------
    # L5: Initial State Construction (algorithmic, no LLM)
    # ------------------------------------------------------------------

    def _run_l5(
        self,
        l2: L2PredicateArtifact,
        l3: L3ActionArtifact,
        scene_objects: List[Dict],
        l1: L1GoalArtifact,
    ) -> L5InitialStateArtifact:
        """
        Construct initial PDDL state from scene observation.

        Rules:
        - checked-* predicates: always FALSE
        - sensed/external predicates: always FALSE (conservative default)
        - graspable: TRUE if "graspable" in object affordances
        - on/above/atop: TRUE if z_distance between two objects < threshold
        - hand-empty, arm-at-home, etc. (global_predicates): TRUE
        - Zero-param state predicates in global_predicates: TRUE

        L5-V1: After construction, add FALSE entries for any 1-arg predicate that
               has no entry for some object (completeness guarantee).
        L5-V2: Verify all checked/sensed/external predicates are FALSE; reset any
               that are accidentally set to TRUE.
        L5-V4: Log WARNING for spatial TRUE facts where objects have no position_3d.
        """
        true_lits: List[Tuple[str, List[str]]] = []
        false_lits: List[Tuple[str, List[str]]] = []

        classifications = getattr(l2, "type_classifications", {}) or {}

        # Determine which predicate names are checked-* (always FALSE)
        checked_names: Set[str] = set()
        for sig in l2.predicate_signatures:
            name = extract_predicate_name_from_literal(sig)
            if name and name.startswith("checked-"):
                checked_names.add(name)
        for cv in l2.checked_variants:
            name = extract_predicate_name_from_literal(cv)
            if name:
                checked_names.add(name)
        # Also collect from type_classifications
        for pname, cls in classifications.items():
            if cls == "checked":
                checked_names.add(pname)

        # Determine sensed/external predicate names (always FALSE)
        sensed_names: Set[str] = set()
        for sname in (l2.sensed_predicates or []):
            sensed_names.add(sname)
        for pname, cls in classifications.items():
            if cls in ("sensed", "external"):
                sensed_names.add(pname)

        # Get all defined predicate names with their arities from signatures
        pred_arities: Dict[str, int] = {}
        for sig in l2.predicate_signatures:
            name = extract_predicate_name_from_literal(sig)
            if name:
                params = re.findall(r"\?[a-zA-Z][a-zA-Z0-9_-]*", sig)
                pred_arities[name] = len(params)

        obj_ids = [o.get("object_id", "") for o in scene_objects if o.get("object_id")]

        # Add checked-* predicates as FALSE for each object
        for cname in checked_names:
            arity = pred_arities.get(cname, 1)
            if arity == 0:
                false_lits.append((cname, []))
            else:
                for obj in scene_objects:
                    oid = obj.get("object_id", "")
                    if oid and arity == 1:
                        false_lits.append((cname, [oid]))
                    # arity >= 2: handled by L5-V1 completeness pass below

        # Add sensed/external predicates as FALSE for each object
        for sname in sensed_names:
            if sname in checked_names:
                continue  # already handled
            arity = pred_arities.get(sname, 1)
            if arity == 0:
                false_lits.append((sname, []))
            else:
                for obj in scene_objects:
                    oid = obj.get("object_id", "")
                    if oid and arity == 1:
                        false_lits.append((sname, [oid]))

        # Add zero-parameter global predicates as TRUE
        for gp in l1.global_predicates:
            gp_clean = gp.strip("() ")
            true_lits.append((gp_clean, []))

        # Add graspable / object-graspable (obj) for graspable objects.
        # These are sensed predicates so they appear as FALSE in L5-V2 reset,
        # but the affordance-derived pass in task_orchestrator.py re-adds them at
        # PDDL generation time. Nothing to do here for sensed predicates.
        # (Keeping this block as a no-op comment for documentation clarity.)

        # Spatial predicates to track for L5-V4
        _SPATIAL_PREDS = {"on", "above", "in", "atop", "attached-to"}

        # Add on(obj, surface) where obj is above an explicit support surface.
        # Surface detection: an object is treated as a support surface if it has
        # "support_surface" OR if it has "placeable_on"/"fixed" without "graspable"
        # (the LLM often tags tables as placeable_on but omits support_surface).
        def _is_surface(o: dict) -> bool:
            affs = set(o.get("affordances", []))
            if "support_surface" in affs:
                return True
            # Non-graspable objects with placeable_on or fixed are support surfaces
            if "graspable" not in affs and ("placeable_on" in affs or "fixed" in affs):
                return True
            return False

        if "on" in pred_arities and pred_arities["on"] == 2:
            surfaces = [o for o in scene_objects if _is_surface(o)]
            manipulables = [o for o in scene_objects if not _is_surface(o)]
            objects_with_on: Set[str] = set()
            for obj in manipulables:
                obj_pos = obj.get("position_3d")
                obj_bbox = obj.get("bounding_box_2d")   # [y1, x1, y2, x2] in 0-1000 scale
                obj_pos2d = obj.get("position_2d")       # [y, x] in 0-1000 scale
                if not obj_pos and not obj_pos2d:
                    print(
                        f"  [L5-V4] WARNING: object '{obj.get('object_id')}' has no position "
                        f"data — spatial facts for 'on' cannot be derived reliably."
                    )
                    continue
                for surf in surfaces:
                    surf_pos = surf.get("position_3d")
                    surf_bbox = surf.get("bounding_box_2d")
                    matched = False

                    # 3D proximity check (reliable when positions are world-frame)
                    if obj_pos and surf_pos:
                        dz = obj_pos[2] - surf_pos[2]
                        dx = abs(obj_pos[0] - surf_pos[0])
                        dy = abs(obj_pos[1] - surf_pos[1]) if len(obj_pos) > 1 else 0
                        if 0 <= dz <= 0.3 and dx < 0.5 and dy < 0.5:
                            matched = True

                    # 2D bounding box containment fallback: the object's centroid
                    # lies inside the surface's 2D bbox. Reliable regardless of
                    # whether positions are world-frame or camera-frame.
                    if not matched and obj_pos2d and surf_bbox and len(surf_bbox) == 4:
                        sy1, sx1, sy2, sx2 = surf_bbox
                        oy, ox = obj_pos2d[0], obj_pos2d[1]
                        if sx1 <= ox <= sx2 and sy1 <= oy <= sy2:
                            matched = True

                    if matched:
                        oid = obj.get("object_id", "")
                        if oid not in objects_with_on:
                            true_lits.append(("on", [oid, surf.get("object_id", "")]))
                            objects_with_on.add(oid)

            # Add on(obj1, obj2) for clearly stacked objects (tight thresholds to avoid
            # false positives from noisy depth perception).
            # Require: obj1 is strictly above obj2 (dz > 0.03m), within 0.04m horizontally.
            non_surfaces = [o for o in scene_objects if "support_surface" not in o.get("affordances", [])]
            for obj1 in non_surfaces:
                for obj2 in non_surfaces:
                    if obj1 is obj2:
                        continue
                    pos1 = obj1.get("position_3d")
                    pos2 = obj2.get("position_3d")
                    if not pos1 or not pos2:
                        continue
                    dz = pos1[2] - pos2[2]
                    dx = abs(pos1[0] - pos2[0])
                    dy = abs(pos1[1] - pos2[1]) if len(pos1) > 1 else 0
                    if 0.03 < dz <= 0.12 and dx < 0.04 and dy < 0.04:
                        oid1 = obj1.get("object_id", "")
                        true_lits.append(("on", [oid1, obj2.get("object_id", "")]))
                        objects_with_on.add(oid1)

            # Fallback: if a graspable object still has no 'on' fact, it must be
            # resting on the table/floor. Find the lowest-z object as the default
            # support proxy, or skip if no support can be inferred.
            # This prevents pick actions from being permanently inapplicable.
            if manipulables:
                # Only consider objects with support_surface affordance as the table proxy.
                # Never use a graspable object as a support surface.
                _surfaces = [o for o in scene_objects if _is_surface(o)]
                # Prefer a surface with position data for z-ordering; fall back to first surface
                _surfaces_with_pos = [o for o in _surfaces if o.get("position_3d")]
                if _surfaces_with_pos:
                    table_proxy = min(_surfaces_with_pos, key=lambda o: o["position_3d"][2])
                elif _surfaces:
                    table_proxy = _surfaces[0]
                else:
                    table_proxy = None
                for obj in manipulables:
                    oid = obj.get("object_id", "")
                    if oid and oid not in objects_with_on and table_proxy and table_proxy.get("object_id") != oid:
                        proxy_id = table_proxy.get("object_id", "")
                        true_lits.append(("on", [oid, proxy_id]))
                        print(
                            f"  [L5] Fallback: '{oid}' has no 'on' fact — assuming on '{proxy_id}' "
                            f"(lowest-z object, z={table_proxy['position_3d'][2]:.3f})"
                        )
                        objects_with_on.add(oid)

        # L5-V1 (AUTO-REPAIR): completeness — for every 1-arg predicate in vocabulary
        # that has no entry for some object, add a FALSE entry.
        covered_true: Dict[str, Set[str]] = {}  # pred_name -> set of object_ids with TRUE entry
        covered_false: Dict[str, Set[str]] = {}  # pred_name -> set of object_ids with FALSE entry
        for pname, args in true_lits:
            if len(args) == 1:
                covered_true.setdefault(pname, set()).add(args[0])
        for pname, args in false_lits:
            if len(args) == 1:
                covered_false.setdefault(pname, set()).add(args[0])

        completeness_repairs = 0
        for pname, arity in pred_arities.items():
            if arity != 1:
                continue
            all_covered = covered_true.get(pname, set()) | covered_false.get(pname, set())
            for oid in obj_ids:
                if oid and oid not in all_covered:
                    false_lits.append((pname, [oid]))
                    covered_false.setdefault(pname, set()).add(oid)
                    completeness_repairs += 1
        if completeness_repairs > 0:
            print(f"  [L5-V1] Added {completeness_repairs} FALSE entries for completeness.")

        # L5-V2 (AUTO-REPAIR): check every true_lit — if predicate is checked/sensed/external,
        # reset to FALSE and log.
        must_be_false: Set[str] = checked_names | sensed_names
        # Also include predicates classified as external
        for pname, cls in classifications.items():
            if cls in ("sensed", "external", "checked"):
                must_be_false.add(pname)

        v2_violations = [(pname, args) for pname, args in true_lits if pname in must_be_false]
        if v2_violations:
            for pname, args in v2_violations:
                print(
                    f"  [L5-V2] AUTO-REPAIR: predicate '{pname}' is classified as "
                    f"checked/sensed/external but was initialized to TRUE. Resetting to FALSE."
                )
                true_lits.remove((pname, args))
                false_lits.append((pname, args))

        # L5-V3a: strip facts whose predicate name is not in the L2 vocabulary.
        # This prevents the planner from seeing predicates in :init/:goal that have
        # no declaration in :predicates — the direct cause of "unknown predicate" errors.
        vocab_names: Set[str] = set(pred_arities.keys())
        # Also allow zero-param global predicates (hand-empty etc.) — they may not be
        # in pred_arities if L2 didn't define them, but are still valid if in global_predicates.
        for gp in l1.global_predicates:
            vocab_names.add(gp.strip("() "))

        def _strip_undefined_preds(lits: List[Tuple[str, List[str]]], label: str) -> List[Tuple[str, List[str]]]:
            kept, removed = [], []
            for pred_name, args in lits:
                if pred_name not in vocab_names:
                    removed.append((pred_name, args))
                else:
                    kept.append((pred_name, args))
            for pred_name, args in removed:
                print(
                    f"  [L5-V3a] Removing {label} fact ({pred_name} {' '.join(args)}): "
                    f"predicate '{pred_name}' is not defined in the L2 vocabulary. "
                    f"Defined predicates: {sorted(vocab_names)}"
                )
            return kept

        true_lits = _strip_undefined_preds(true_lits, "TRUE")
        false_lits = _strip_undefined_preds(false_lits, "FALSE")

        # L5-V3b: entity args in facts must reference known scene entities.
        # Per spec: type mismatches indicate a system bug in state construction.
        # We log each violation as a system error and remove the offending literal
        # (rather than passing malformed facts to the planner).
        all_lits = list(true_lits) + list(false_lits)
        for pred_name, args in all_lits:
            for arg in args:
                if arg and arg not in obj_ids:
                    print(
                        f"  [L5-V3b] SYSTEM BUG: fact ({pred_name} {' '.join(args)}) references "
                        f"unknown entity '{arg}' — not in scene objects {obj_ids}. "
                        f"This indicates a bug in the state construction procedure. "
                        f"Removing fact to prevent planner errors."
                    )
                    if (pred_name, args) in true_lits:
                        true_lits.remove((pred_name, args))
                    elif (pred_name, args) in false_lits:
                        false_lits.remove((pred_name, args))
                    break

        # L5-V5 (WARN): check if initial state already satisfies all goals
        true_facts: Set[str] = {
            f"({name} {' '.join(args)})" if args else f"({name})"
            for name, args in true_lits
        }
        goal_already_satisfied = all(
            gp.strip() in true_facts
            for gp in l1.goal_predicates
            if gp.strip()
        )
        if goal_already_satisfied and l1.goal_predicates:
            print(
                "  [L5-V5] WARNING: Initial state already satisfies all goals. "
                "The task may be trivially complete. "
                "Verify goal specification and perception accuracy."
            )

        return L5InitialStateArtifact(true_literals=true_lits, false_literals=false_lits)

    # ------------------------------------------------------------------
    # LLM call helper
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.1,
        response_mime_type: str = "application/json",
    ) -> str:
        if self._llm_client is not None:
            from src.llm_interface.base import GenerateConfig
            cfg = GenerateConfig(
                temperature=temperature,
                top_p=0.9,
                max_output_tokens=8192,
                response_mime_type=response_mime_type,
                thinking_budget=0,  # disable thinking — structured JSON needs output tokens, not reasoning
            )
            response = await self._llm_client.generate_async(prompt, config=cfg)
            return response.text

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.9,
            max_output_tokens=8192,
            response_mime_type=response_mime_type,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=config,
        )
        text = getattr(response, "text", None)
        if text is None:
            raise ValueError("LLM response missing text payload")
        return text if isinstance(text, str) else str(text)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_errors(errors: Optional[List[str]]) -> str:
        if not errors:
            return ""
        lines = "\n".join(f"  - {e}" for e in errors)
        return (
            f"\n**PREVIOUS ATTEMPT ERRORS (fix all of these):**\n{lines}\n"
            f"\nFix every error listed above before responding.\n"
        )

    def _get_dkb_predicate_hints(self, task: str) -> str:
        if self.dkb is None:
            return "(none)"
        try:
            hints = self.dkb.get_predicate_hints(task)
            return json.dumps(hints, indent=2) if hints else "(none)"
        except Exception:
            return "(none)"

    def _get_dkb_action_hints(self, task: str) -> str:
        if self.dkb is None:
            return "(none)"
        try:
            hints = self.dkb.get_action_hints(task)
            return json.dumps(hints, indent=2) if hints else "(none)"
        except Exception:
            return "(none)"
