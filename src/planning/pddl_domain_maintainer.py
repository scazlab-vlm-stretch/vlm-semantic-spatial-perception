"""
PDDL representation builder with staged validation and repair.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import yaml
from PIL import Image

from ..utils.prompt_utils import render_prompt_template
from .llm_task_analyzer import LLMTaskAnalyzer
from .pddl_representation import PDDLRepresentation
from .utils.task_types import (
    AbstractGoal,
    ActionSchemaLibrary,
    GroundingSummary,
    LayeredDomainArtifact,
    PredicateInventory,
    TaskAnalysis,
)


LOGICAL_OPERATORS = {"and", "or", "not", "forall", "exists", "when", "imply"}
REPAIR_LAYER_ORDER = ["actions", "predicates", "goal"]


def sanitize_pddl_name(name: str, max_length: int = 50) -> str:
    """Sanitize a raw string for PDDL identifiers.

    Args:
        name: Raw name to sanitize.
        max_length: Maximum allowed identifier length.

    Returns:
        Sanitized PDDL identifier.

    Example:
        >>> sanitize_pddl_name("Blue plastic bottle")
        'blue_plastic_bottle'
    """

    cleaned = re.sub(r"\([^)]*\)", "", name.lower())
    cleaned = re.sub(r"['\"].*?['\"]", "", cleaned)
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if cleaned and not cleaned[0].isalpha():
        cleaned = f"obj_{cleaned}"
    if len(cleaned) > max_length:
        parts = cleaned[:max_length].split("_")
        cleaned = "_".join(parts[:-1]) if len(parts) > 1 else cleaned[:max_length]
    return cleaned or "object"


def sanitize_pddl_formula(formula: str) -> str:
    """Remove quoted strings and normalize whitespace in formulas."""

    if not formula:
        return formula
    formula = re.sub(r'"[^"]*"', "", formula)
    formula = re.sub(r"'[^']*'", "", formula)
    formula = re.sub(r"\s+", " ", formula)
    formula = re.sub(r"\(\s+", "(", formula)
    formula = re.sub(r"\s+\)", ")", formula)
    return formula.strip()


class PDDLDomainMaintainer:
    """Build and maintain a staged, validated PDDL representation.

    Args:
        pddl_representation: Mutable PDDL serializer/holder.
        api_key: Optional Gemini API key.
        model_name: Model name for staged analysis and repair.
        prompts_config_path: Optional override for repair prompts.
        task_analyzer_prompts_path: Optional override for analysis prompts.

    Example:
        >>> maintainer = PDDLDomainMaintainer(PDDLRepresentation(), api_key="test-key")
        >>> analysis = await maintainer.build_representation("put the mug on the coaster")
    """

    DEFAULT_PROMPTS_CONFIG = (
        Path(__file__).resolve().parents[2] / "config" / "pddl_domain_maintainer_prompts.yaml"
    )

    def __init__(
        self,
        pddl_representation: PDDLRepresentation,
        api_key: Optional[str] = None,
        model_name: str = "gemini-robotics-er-1.5-preview",
        prompts_config_path: Optional[Union[str, Path]] = None,
        task_analyzer_prompts_path: Optional[Union[str, Path]] = None,
        llm_client: Optional[Any] = None,  # src.llm_interface.LLMClient
    ):
        """
        Initialize domain maintainer.

        Args:
            pddl_representation: PDDL representation to maintain
            api_key: Gemini API key for LLM analysis
            model_name: LLM model to use
            prompts_config_path: Override path to prompts config YAML (defaults to config/pddl_domain_maintainer_prompts.yaml)
            task_analyzer_prompts_path: Override path to prompts config for LLMTaskAnalyzer (defaults to config/llm_task_analyzer_prompts.yaml)
            llm_client: Optional LLMClient instance; when provided, api_key/model_name are ignored
        """
        if prompts_config_path is None:
            prompts_config_path = self.DEFAULT_PROMPTS_CONFIG

        self.pddl = pddl_representation
        self.llm_analyzer = LLMTaskAnalyzer(
            api_key=api_key,
            model_name=model_name,
            prompts_config_path=task_analyzer_prompts_path,
            llm_client=llm_client,
        )
        self.robot_description = self.llm_analyzer.robot_description
        self.prompts_config_path = Path(prompts_config_path)

        with self.prompts_config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        self.prompt_templates = {
            key: value
            for key, value in config_data.items()
            if key.endswith("_prompt") and isinstance(value, str)
        }

        self.current_task: Optional[str] = None
        self.task_analysis: Optional[TaskAnalysis] = None

        self.observed_objects: List[Dict[str, Any]] = []
        self.observed_relationships: List[str] = []
        self.observed_predicate_strings: List[str] = []
        self.observed_object_types: Set[str] = set()
        self.observed_predicates: Set[str] = set()

        self.goal_object_types: Set[str] = set()
        self.global_predicates: Set[str] = set()
        self.domain_version = 0
        self.last_update_observations = 0

    async def build_representation(
        self,
        task: str,
        scene_context: Optional[Dict[str, Any]] = None,
        image: Optional[Union[np.ndarray, Image.Image, str, Path]] = None,
    ) -> TaskAnalysis:
        """Build the goal, predicate, and action layers.

        Args:
            task: Natural-language task.
            scene_context: Optional observed context with keys `objects`,
                `relationships`, and `predicates`.
            image: Optional scene image.

        Returns:
            Staged task analysis.
        """

        scene_context = scene_context or {}
        self.current_task = task
        self.observed_objects = list(scene_context.get("objects") or [])
        self.observed_relationships = list(scene_context.get("relationships") or [])
        self.observed_predicate_strings = list(scene_context.get("predicates") or [])
        self._refresh_observation_indices()

        abstract_goal = self.llm_analyzer.analyze_goal(
            task_description=task,
            observed_objects=self.observed_objects,
            observed_relationships=self.observed_relationships,
            environment_image=image,
        )
        predicate_inventory = self.llm_analyzer.analyze_predicates(
            task_description=task,
            abstract_goal=abstract_goal,
            observed_objects=self.observed_objects,
            observed_relationships=self.observed_relationships,
            environment_image=image,
        )
        action_schemas = self.llm_analyzer.analyze_actions(
            task_description=task,
            abstract_goal=abstract_goal,
            predicate_inventory=predicate_inventory,
            observed_objects=self.observed_objects,
            observed_relationships=self.observed_relationships,
            environment_image=image,
        )
        action_schemas = self._normalize_action_schema_library(action_schemas)

        self.task_analysis = TaskAnalysis(
            abstract_goal=abstract_goal,
            predicate_inventory=predicate_inventory,
            action_schemas=action_schemas,
            grounding_summary=GroundingSummary(),
            diagnostics=self._new_diagnostics(),
        )
        self.goal_object_types = set(self.task_analysis.goal_object_references())
        self.global_predicates = self._extract_global_predicates(self.task_analysis.predicate_signatures())

        await self._rebuild_pddl_from_analysis(include_grounding=False)
        validation = await self._validate_representation()
        self._record_validation(validation)
        self.domain_version += 1
        return self.task_analysis

    async def ground_representation(
        self,
        detected_objects: List[Dict[str, Any]],
        predicates: Optional[List[str]] = None,
        detected_relationships: Optional[List[str]] = None,
        image: Optional[Union[np.ndarray, Image.Image, str, Path]] = None,
    ) -> Dict[str, Any]:
        """Ground the current representation against the observed world.

        Args:
            detected_objects: Observed object summaries.
            predicates: Observed predicate strings.
            detected_relationships: Optional observed relationships.
            image: Optional scene image.

        Returns:
            Grounding/update statistics.
        """

        if self.task_analysis is None or self.current_task is None:
            return {"skipped": True, "reason": "no task analysis yet"}

        previous_object_ids = set(self.task_analysis.grounding_summary.observed_object_ids)
        previous_predicates = set(self.task_analysis.grounding_summary.grounded_predicates)

        self.observed_objects = list(detected_objects or [])
        self.observed_predicate_strings = list(predicates or [])
        if detected_relationships is not None:
            self.observed_relationships = list(detected_relationships)
        self._refresh_observation_indices()

        grounding_summary = self.llm_analyzer.analyze_grounding(
            task_description=self.current_task,
            abstract_goal=self.task_analysis.abstract_goal,
            predicate_inventory=self.task_analysis.predicate_inventory,
            action_schemas=self.task_analysis.action_schemas,
            observed_objects=self.observed_objects,
            observed_relationships=self.observed_relationships,
            predicates=self.observed_predicate_strings,
            environment_image=image,
        )
        if not grounding_summary.grounded_predicates:
            grounding_summary.grounded_predicates = list(self.observed_predicate_strings)

        self.task_analysis.grounding_summary = grounding_summary
        await self._rebuild_pddl_from_analysis(include_grounding=True)

        validation = await self._validate_representation()
        self._record_validation(validation)

        goal_found = sorted(self.goal_object_types - set(grounding_summary.missing_references))
        goal_missing = sorted(set(grounding_summary.missing_references))

        self.last_update_observations += len(self.observed_objects)
        self.domain_version += 1

        return {
            "objects_added": len(set(grounding_summary.observed_object_ids) - previous_object_ids),
            "new_predicates": sorted(set(grounding_summary.grounded_predicates) - previous_predicates),
            "total_observations": self.last_update_observations,
            "goal_objects_found": goal_found,
            "goal_objects_missing": goal_missing,
            "grounding_complete": not goal_missing,
            "validation_valid": validation["valid"],
        }

    async def repair_representation(
        self,
        failure_context: Dict[str, Any],
        layer: str,
    ) -> Dict[str, Any]:
        """Repair a representation layer and rebuild dependent layers.

        Args:
            failure_context: Structured failure context from validation or planning.
            layer: One of `actions`, `predicates`, or `goal`.

        Returns:
            Repair record with validation results.
        """

        if self.task_analysis is None or self.current_task is None:
            raise RuntimeError("No task analysis is available to repair.")
        if layer not in REPAIR_LAYER_ORDER:
            raise ValueError(f"Unsupported repair layer: {layer}")

        prompt_key = {
            "actions": "repair_actions_prompt",
            "predicates": "repair_predicates_prompt",
            "goal": "repair_goal_prompt",
        }[layer]

        payload = self._request_repair_json(
            template_key=prompt_key,
            failure_context=failure_context,
        )

        if layer == "actions":
            self.task_analysis.action_schemas = self._normalize_action_schema_library(ActionSchemaLibrary(
                actions=self._normalize_actions(payload.get("actions")),
                planning_notes=self._as_string_list(payload.get("repair_notes")),
            ))
        elif layer == "predicates":
            self.task_analysis.predicate_inventory = PredicateInventory(
                predicates=self._as_string_list(payload.get("predicates")),
                selection_rationale=self._as_string_list(payload.get("selection_rationale")),
                omitted_predicates=self._as_string_list(payload.get("omitted_predicates")),
            )
            self.global_predicates = self._extract_global_predicates(
                self.task_analysis.predicate_inventory.predicates
            )
            self.task_analysis.action_schemas = self.llm_analyzer.analyze_actions(
                task_description=self.current_task,
                abstract_goal=self.task_analysis.abstract_goal,
                predicate_inventory=self.task_analysis.predicate_inventory,
                observed_objects=self.observed_objects,
                observed_relationships=self.observed_relationships,
            )
            self.task_analysis.action_schemas = self._normalize_action_schema_library(
                self.task_analysis.action_schemas
            )
        else:
            self.task_analysis.abstract_goal = AbstractGoal(
                summary=str(payload.get("summary", "")).strip(),
                goal_literals=self._as_string_list(payload.get("goal_literals")),
                goal_objects=self._as_string_list(payload.get("goal_objects")),
                success_checks=self._as_string_list(payload.get("success_checks")),
            )
            self.goal_object_types = set(self.task_analysis.goal_object_references())
            self.task_analysis.predicate_inventory = self.llm_analyzer.analyze_predicates(
                task_description=self.current_task,
                abstract_goal=self.task_analysis.abstract_goal,
                observed_objects=self.observed_objects,
                observed_relationships=self.observed_relationships,
            )
            self.global_predicates = self._extract_global_predicates(
                self.task_analysis.predicate_inventory.predicates
            )
            self.task_analysis.action_schemas = self.llm_analyzer.analyze_actions(
                task_description=self.current_task,
                abstract_goal=self.task_analysis.abstract_goal,
                predicate_inventory=self.task_analysis.predicate_inventory,
                observed_objects=self.observed_objects,
                observed_relationships=self.observed_relationships,
            )
            self.task_analysis.action_schemas = self._normalize_action_schema_library(
                self.task_analysis.action_schemas
            )

        if self.observed_objects:
            self.task_analysis.grounding_summary = self.llm_analyzer.analyze_grounding(
                task_description=self.current_task,
                abstract_goal=self.task_analysis.abstract_goal,
                predicate_inventory=self.task_analysis.predicate_inventory,
                action_schemas=self.task_analysis.action_schemas,
                observed_objects=self.observed_objects,
                observed_relationships=self.observed_relationships,
                predicates=self.observed_predicate_strings,
            )
            if not self.task_analysis.grounding_summary.grounded_predicates:
                self.task_analysis.grounding_summary.grounded_predicates = list(
                    self.observed_predicate_strings
                )

        await self._rebuild_pddl_from_analysis(include_grounding=bool(self.observed_objects))
        validation = await self._validate_representation()
        repair_record = {
            "layer": layer,
            "failure_context": failure_context,
            "validation": validation,
        }
        self._record_validation(validation)
        self.task_analysis.diagnostics.setdefault("repair_history", []).append(repair_record)
        self.domain_version += 1
        return repair_record

    async def initialize_from_task(
        self,
        task_description: str,
        environment_image: Optional[Union[np.ndarray, Image.Image, str, Path]] = None,
    ) -> TaskAnalysis:
        """
            Compatibility wrapper for staged representation building.

            Performs LLM-based analysis to predict required predicates, actions,
            and object types, then initializes the domain accordingly.

            Args:
                task_description: Natural language task
                environment_image: Optional environment image for context

            Returns:
                TaskAnalysis with predicted requirements
        """
        self.current_task = task_description

        # Analyze task with LLM (no observations yet)
        self.task_analysis = self.llm_analyzer.analyze_task(
            task_description=task_description,
            observed_objects=None,
            observed_relationships=None,
            environment_image=environment_image,
            timeout=15.0
        )

        # Check if analysis succeeded
        if self.task_analysis is None:
            raise RuntimeError(f"LLM task analysis failed for task: {task_description}")

        # Extract goal object types from task
        self.goal_object_types = set(self.task_analysis.goal_objects)

        # Initialize domain with predicted components
        await self._initialize_domain_from_analysis()

        self.domain_version += 1

        return self.task_analysis

    async def initialize_from_layered_artifact(self, artifact: LayeredDomainArtifact) -> TaskAnalysis:
        """
        Initialize PDDL domain from a LayeredDomainArtifact produced by LayeredDomainGenerator.

        Converts the artifact to a TaskAnalysis via the bridge method and then runs the
        existing _initialize_domain_from_analysis() write path unchanged.

        Args:
            artifact: Output of LayeredDomainGenerator.generate_domain()

        Returns:
            TaskAnalysis (backward-compatible with callers expecting this type)
        """
        self.current_task = artifact.task_description
        self.task_analysis = artifact.to_task_analysis()
        self.goal_object_types = set(self.task_analysis.goal_objects)
        self._l5_artifact = artifact.l5  # stash for use in generate_pddl_files
        await self._initialize_domain_from_analysis()
        self.domain_version += 1
        return self.task_analysis

    async def _initialize_domain_from_analysis(self) -> None:
        """Initialize PDDL domain based on task analysis."""
        if not self.task_analysis:
            return

        # Store global predicates from task analysis
        if self.task_analysis.global_predicates:
            self.set_global_predicates(self.task_analysis.global_predicates)
            print(f"  • Identified {len(self.task_analysis.global_predicates)} global predicates: {', '.join(self.task_analysis.global_predicates)}")

            # Add global predicates to domain (they typically have no parameters)
            for pred_name in self.task_analysis.global_predicates:
                if pred_name not in self.pddl.predicates:
                    await self.pddl.add_predicate_async(pred_name, [])

        # Add predicted predicates
        # First, infer parameter counts from action definitions
        predicate_param_counts = self._infer_predicate_arities_from_actions(
            self.task_analysis.required_actions
        )

        normalized_predicates: List[str] = []
        seen_normalized: Set[str] = set()

        for predicate_signature in self.task_analysis.relevant_predicates:
            (
                predicate_name,
                parameter_defs,
                normalized_display,
            ) = self._normalize_predicate_signature(predicate_signature)

            if not predicate_name:
                continue

            # If predicate has no parameters but is used with parameters in actions,
            # infer the parameter count from action usage
            if not parameter_defs and predicate_name in predicate_param_counts:
                param_count = predicate_param_counts[predicate_name]
                parameter_defs = [(f"obj{i+1}", "object") for i in range(param_count)]
                # Update normalized display
                param_vars = ' '.join([f"?{name}" for name, _ in parameter_defs])
                normalized_display = f"({predicate_name} {param_vars})" if param_vars else predicate_name

            if predicate_name not in self.pddl.predicates:
                await self.pddl.add_predicate_async(
                    predicate_name,
                    parameter_defs,
                )

            if normalized_display and normalized_display not in seen_normalized:
                normalized_predicates.append(normalized_display)
                seen_normalized.add(normalized_display)

        # Replace the task analysis list with normalized, untyped predicate strings
        if normalized_predicates:
            self.task_analysis.relevant_predicates = normalized_predicates

        # Add LLM-generated actions
        for idx, action_def in enumerate(self.task_analysis.required_actions):
            if not isinstance(action_def, dict):
                raise ValueError(
                    f"Invalid required_actions[{idx}] type: expected dict, got {type(action_def).__name__}"
                )
            try:
                # Parse parameters
                params = []
                if "parameters" in action_def:
                    param_list = action_def["parameters"]
                    if isinstance(param_list, list):
                        for p in param_list:
                            if not isinstance(p, str):
                                continue
                            if " - " in p:
                                name, type_ = p.split(" - ")
                                params.append((name.strip("?"), type_.strip()))
                            else:
                                param_name = p.strip().lstrip("?")
                                if param_name:
                                    # Default to untyped STRIPS parameter
                                    params.append((param_name, "object"))

                # Sanitize preconditions and effects to remove quoted strings
                # LLMs sometimes add invalid quoted strings like: (is-empty ?obj "reservoir")
                precondition = sanitize_pddl_formula(action_def.get("precondition", ""))
                effect = sanitize_pddl_formula(action_def.get("effect", ""))

                # Add action to domain
                # Note: Using sync method here as PDDLRepresentation doesn't have async action methods yet
                self.pddl.add_llm_generated_action(
                    name=action_def.get("name", "unknown"),
                    parameters=params,
                    precondition=precondition,
                    effect=effect,
                    description=action_def.get("description", "")
                )
            except Exception as e:
                print(f"⚠ Failed to add action {action_def.get('name')}: {e}")

        # Validate that all predicates used in actions are defined
        # Auto-add any missing predicates to ensure domain consistency
        validation_result = await self.validate_and_fix_action_predicates()
        if validation_result["missing_predicates"]:
            print(f"✓ Auto-added {len(validation_result['missing_predicates'])} missing predicates")
        if validation_result["invalid_actions"]:
            print(f"⚠ Found {len(validation_result['invalid_actions'])} actions with parsing issues")

    async def update_from_observations(
        self,
        detected_objects: List[Dict[str, Any]],
        detected_relationships: Optional[List[str]] = None,
        predicates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compatibility wrapper for grounding refresh."""

        return await self.ground_representation(
            detected_objects=detected_objects,
            predicates=predicates,
            detected_relationships=detected_relationships,
        )

    async def is_domain_complete(self) -> bool:
        """Return whether the goal, predicate, and action layers validate."""

        if self.task_analysis is None:
            return False
        validation = await self._validate_representation()
        return validation["layer_validity"]["goal"] and validation["layer_validity"]["predicates"] and validation[
            "layer_validity"
        ]["actions"]

    async def validate_and_fix_action_predicates(self) -> Dict[str, List[str]]:
        """Ensure action predicates are defined in the current inventory."""

        if self.task_analysis is None:
            return {"missing_predicates": [], "invalid_actions": []}

        missing_predicates: List[str] = []
        invalid_actions: List[str] = []
        defined = {
            self._normalize_predicate_signature(sig)[0]
            for sig in self.task_analysis.predicate_inventory.predicates
            if self._normalize_predicate_signature(sig)[0]
        }
        inferred_arities = self._infer_action_predicate_arities(self.task_analysis.action_schemas.actions)

        for action in self.task_analysis.action_schemas.actions:
            try:
                used = self._extract_predicate_usages(action.get("precondition", "")) | self._extract_predicate_usages(
                    action.get("effect", "")
                )
                for pred_name, arity in used:
                    if pred_name in defined:
                        continue
                    params = " ".join(f"?obj{i+1}" for i in range(arity))
                    signature = f"({pred_name} {params})".strip()
                    signature = signature.replace(" )", ")")
                    self.task_analysis.predicate_inventory.predicates.append(signature)
                    defined.add(pred_name)
                    missing_predicates.append(signature)
            except Exception:
                invalid_actions.append(str(action.get("name", "unknown")))

        if missing_predicates:
            self.global_predicates = self._extract_global_predicates(
                self.task_analysis.predicate_inventory.predicates
            )
            await self._rebuild_pddl_from_analysis(include_grounding=bool(self.observed_objects))

        _ = inferred_arities
        return {"missing_predicates": missing_predicates, "invalid_actions": invalid_actions}

    async def set_goal_from_task_analysis(self) -> None:
        """Populate the PDDL goal from the staged analysis."""

        if self.task_analysis is None:
            return

        goal_formulas = self._resolved_goal_formulas()
        await self.pddl.clear_goal_state_async()

        for formula in goal_formulas:
            parsed = self._parse_literal_formula(formula)
            if parsed is None:
                await self.pddl.add_goal_formula_async(formula)
                continue
            predicate, arguments, negated = parsed
            if predicate not in self.pddl.predicates:
                await self.pddl.add_goal_formula_async(formula)
                continue
            await self.pddl.add_goal_literal_async(predicate, arguments, negated=negated)

    async def refine_domain_from_error(
        self,
        error_message: str,
        current_domain_pddl: Optional[str] = None,
        current_problem_pddl: Optional[str] = None,
    ) -> None:
        """Compatibility wrapper for targeted repair from error text."""

        validation = await self._validate_representation()
        layer = self.classify_failure_layer(error_message=error_message, validation=validation)
        await self.repair_representation(
            failure_context={
                "error_message": error_message,
                "current_domain_pddl": current_domain_pddl,
                "current_problem_pddl": current_problem_pddl,
                "validation": validation,
            },
            layer=layer,
        )

    async def update_object_tracker_from_domain(self, object_tracker: Any) -> None:
        """Update tracker predicates and actions from the staged representation."""

        if self.task_analysis is None:
            return
        object_tracker.set_pddl_predicates(self.task_analysis.predicate_signatures())
        object_tracker.set_pddl_actions(
            [action.get("name", "unknown") for action in self.task_analysis.action_context()]
        )

    async def are_goal_objects_observed(self) -> bool:
        """Return whether all symbolic goal references are grounded or matched."""

        return len(await self.get_missing_goal_objects()) == 0

    async def get_missing_goal_objects(self) -> List[str]:
        """Return goal references that are not grounded yet."""

        if self.task_analysis is None:
            return []
        missing = set(self.task_analysis.grounding_summary.missing_references)
        if missing:
            return sorted(missing)

        observed_labels = {
            sanitize_pddl_name(obj.get("object_id", ""))
            for obj in self.observed_objects
            if obj.get("object_id")
        } | {
            sanitize_pddl_name(obj.get("object_type", ""))
            for obj in self.observed_objects
            if obj.get("object_type")
        }

        unresolved = []
        for goal_object in self.task_analysis.goal_object_references():
            goal_key = sanitize_pddl_name(goal_object)
            bindings = self.task_analysis.grounding_summary.object_bindings.get(goal_object, [])
            if bindings:
                continue
            if any(goal_key in label or label in goal_key for label in observed_labels if label):
                continue
            unresolved.append(goal_object)
        return sorted(set(unresolved))

    async def get_domain_statistics(self) -> Dict[str, Any]:
        """Return planning representation statistics."""

        domain_snapshot = await self.pddl.get_domain_snapshot()
        problem_snapshot = await self.pddl.get_problem_snapshot()
        validation = await self._validate_representation()
        goal_total = len(self.goal_object_types)
        goal_missing = await self.get_missing_goal_objects()
        goal_observed = max(goal_total - len(goal_missing), 0)
        return {
            "domain_version": self.domain_version,
            "task": self.current_task,
            "predicates_defined": len(domain_snapshot["predicates"]),
            "actions_defined": len(domain_snapshot["predefined_actions"]) + len(domain_snapshot["llm_generated_actions"]),
            "object_types_observed": len(self.observed_object_types),
            "object_instances": len(problem_snapshot["object_instances"]),
            "initial_literals": len(problem_snapshot["initial_literals"]),
            "goal_literals": len(problem_snapshot["goal_literals"]) + len(self.pddl.goal_formulas),
            "goal_objects_total": goal_total,
            "goal_objects_observed": goal_observed,
            "domain_complete": validation["layer_validity"]["goal"]
            and validation["layer_validity"]["predicates"]
            and validation["layer_validity"]["actions"],
            "goals_observable": len(goal_missing) == 0,
            "validation": validation,
        }

    def get_global_predicates(self) -> List[str]:
        """Return zero-arity predicates treated as global state."""

        return sorted(self.global_predicates)

    def set_global_predicates(self, predicates: List[str]) -> None:
        """Override the tracked global predicates."""

        self.global_predicates = set(predicates)

    def clear_global_predicates(self) -> None:
        """Clear tracked global predicates."""

        self.global_predicates.clear()

    def classify_failure_layer(
        self,
        error_message: Optional[str],
        validation: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Classify the lowest likely failing abstraction layer.

        Args:
            error_message: Planner error, if any.
            validation: Optional precomputed validation result.

        Returns:
            Repair layer name.
        """

        validation = validation or {}
        layer_validity = validation.get("layer_validity") or {}
        if layer_validity and not layer_validity.get("actions", True):
            return "actions"
        if layer_validity and not layer_validity.get("predicates", True):
            return "predicates"
        if layer_validity and not layer_validity.get("goal", True):
            return "goal"

        error = (error_message or "").lower()
        if any(token in error for token in ["precondition", "effect", "action", "unsolvable", "no plan", "empty plan"]):
            return "actions"
        if any(token in error for token in ["predicate", "arity", "undefined predicate", "symbol"]):
            return "predicates"
        return "goal"

    async def _rebuild_pddl_from_analysis(self, include_grounding: bool) -> None:
        if self.task_analysis is None:
            return

        self._reset_pddl()

        for signature in self.task_analysis.predicate_inventory.predicates:
            pred_name, params, _ = self._normalize_predicate_signature(signature)
            if pred_name is None:
                continue
            await self.pddl.add_predicate_async(pred_name, params)

        for pred_name in self.global_predicates:
            if pred_name not in self.pddl.predicates:
                await self.pddl.add_predicate_async(pred_name, [])

        for action in self.task_analysis.action_schemas.actions:
            name = sanitize_pddl_name(str(action.get("name", "")))
            if not name:
                continue
            parameters = self._normalize_action_parameters(action.get("parameters") or [])
            precondition = sanitize_pddl_formula(str(action.get("precondition", "(and)")) or "(and)")
            precondition = self._strip_negative_preconditions(precondition)
            self.pddl.add_llm_generated_action(
                name=name,
                parameters=parameters,
                precondition=precondition,
                effect=sanitize_pddl_formula(str(action.get("effect", "(and)")) or "(and)"),
                description=str(action.get("description", "")).strip() or None,
            )

        if include_grounding:
            for obj in self.observed_objects:
                obj_id = obj.get("object_id")
                if obj_id and obj_id not in self.pddl.object_instances:
                    await self.pddl.add_object_instance_async(str(obj_id), "object")

            for pred_name in self.global_predicates:
                if pred_name in self.pddl.predicates:
                    await self.pddl.add_initial_literal_async(pred_name, [])

            grounded_predicates = (
                self.task_analysis.grounding_summary.grounded_predicates or self.observed_predicate_strings
            )
            for predicate_str in grounded_predicates:
                parsed = self._parse_predicate_instance(predicate_str)
                if parsed is None:
                    continue
                predicate, arguments, negated = parsed
                if predicate not in self.pddl.predicates:
                    continue
                if any(arg not in self.pddl.object_instances for arg in arguments):
                    continue
                await self.pddl.add_initial_literal_async(predicate, arguments, negated=negated)

        await self.set_goal_from_task_analysis()

    async def _validate_representation(self) -> Dict[str, Any]:
        if self.task_analysis is None:
            return {
                "valid": False,
                "layer_validity": {"goal": False, "predicates": False, "actions": False, "grounding": False},
                "issues": [{"layer": "goal", "message": "No task analysis has been built."}],
                "warnings": [],
                "suggested_repair_layer": "goal",
            }

        inventory_names = {
            name
            for name, _, _ in (
                self._normalize_predicate_signature(signature)
                for signature in self.task_analysis.predicate_inventory.predicates
            )
            if name
        }
        action_effects = set()
        action_usages = set()
        issues: List[Dict[str, str]] = []
        warnings: List[str] = []

        goal_formulas = self._resolved_goal_formulas()
        goal_predicate_names = set()
        goal_object_names = set()
        for formula in goal_formulas:
            if any(token in formula for token in ["forall", "exists", "implies", "="]) or "?" in formula:
                issues.append(
                    {
                        "layer": "goal",
                        "message": f"Goal formula is not a grounded STRIPS literal: `{formula}`.",
                    }
                )
                continue
            parsed = self._parse_literal_formula(formula)
            if parsed is not None:
                predicate, arguments, _ = parsed
                goal_predicate_names.add(predicate)
                goal_object_names.update(arguments)
            else:
                goal_predicate_names |= {
                    pred_name for pred_name, _ in self._extract_predicate_usages(formula)
                }

        for goal_predicate in goal_predicate_names:
            if goal_predicate not in inventory_names:
                issues.append(
                    {
                        "layer": "predicates",
                        "message": f"Goal predicate `{goal_predicate}` is not expressible by the predicate inventory.",
                    }
                )

        grounded_predicates = self.task_analysis.grounding_summary.grounded_predicates or self.observed_predicate_strings
        grounded_names = {
            parsed[0]
            for parsed in (self._parse_predicate_instance(value) for value in grounded_predicates)
            if parsed is not None
        }

        for action in self.task_analysis.action_schemas.actions:
            action_name = str(action.get("name", "unknown"))
            if "(not " in str(action.get("precondition", "")):
                issues.append(
                    {
                        "layer": "actions",
                        "message": f"Action `{action_name}` uses negative preconditions unsupported by the STRIPS solver backend.",
                    }
                )
            try:
                precondition_usages = self._extract_predicate_usages(action.get("precondition", ""))
                effect_usages = self._extract_predicate_usages(action.get("effect", ""))
            except Exception:
                issues.append({"layer": "actions", "message": f"Action `{action_name}` could not be parsed."})
                continue

            for pred_name, _ in precondition_usages | effect_usages:
                action_usages.add(pred_name)
                if pred_name not in inventory_names:
                    issues.append(
                        {
                            "layer": "predicates",
                            "message": f"Action `{action_name}` uses undefined predicate `{pred_name}`.",
                        }
                    )
            for pred_name, _ in effect_usages:
                action_effects.add(pred_name)

        for goal_predicate in goal_predicate_names:
            if goal_predicate not in action_effects and goal_predicate not in grounded_names:
                issues.append(
                    {
                        "layer": "actions",
                        "message": f"No grounded state or action effect can satisfy goal predicate `{goal_predicate}`.",
                    }
                )

        unused_predicates = sorted(inventory_names - goal_predicate_names - action_usages - grounded_names - self.global_predicates)
        if unused_predicates:
            warnings.append(f"Unused predicates: {', '.join(unused_predicates)}")

        grounding_complete = True
        missing_goal_objects = await self.get_missing_goal_objects()
        if self.observed_objects and missing_goal_objects:
            grounding_complete = False
            issues.append(
                {
                    "layer": "grounding",
                    "message": f"Missing grounded object references: {', '.join(missing_goal_objects)}.",
                }
            )

        observed_object_ids = {str(obj.get("object_id")) for obj in self.observed_objects if obj.get("object_id")}
        if include_grounding_objects := bool(observed_object_ids):
            for obj_name in goal_object_names:
                if obj_name not in observed_object_ids and self.observed_objects:
                    warnings.append(f"Goal references `{obj_name}` that is not a detected object ID.")
        _ = include_grounding_objects

        layer_validity = {
            "goal": not any(issue["layer"] == "goal" for issue in issues),
            "predicates": not any(issue["layer"] == "predicates" for issue in issues),
            "actions": not any(issue["layer"] == "actions" for issue in issues),
            "grounding": grounding_complete,
        }
        suggested_repair_layer = "goal"
        for layer in REPAIR_LAYER_ORDER:
            if not layer_validity[layer]:
                suggested_repair_layer = layer
                break

        return {
            "valid": layer_validity["goal"] and layer_validity["predicates"] and layer_validity["actions"] and grounding_complete,
            "layer_validity": layer_validity,
            "issues": issues,
            "warnings": warnings,
            "suggested_repair_layer": suggested_repair_layer,
        }

    def _request_repair_json(self, template_key: str, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        if self.task_analysis is None or self.current_task is None:
            raise RuntimeError("No task analysis is available to repair.")

        prompt = render_prompt_template(
            self._get_prompt_template(template_key),
            {
                "TASK": self.current_task,
                "FAILURE_CONTEXT_JSON": json.dumps(failure_context, indent=2, default=str),
                "ABSTRACT_GOAL_JSON": json.dumps(self._goal_dict(), indent=2),
                "PREDICATE_INVENTORY_JSON": json.dumps(self._predicate_inventory_dict(), indent=2),
                "ACTION_SCHEMAS_JSON": json.dumps(self.task_analysis.action_schemas.actions, indent=2),
                "GROUNDING_SUMMARY_JSON": json.dumps(self._grounding_summary_dict(), indent=2),
            },
        )
        text = self.llm_analyzer._generate_content(content_parts=[prompt], timeout=30.0)
        if not text:
            raise RuntimeError(f"Repair prompt `{template_key}` returned an empty response.")
        from .layered_domain_generator import _extract_json
        payload = _extract_json(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object from {template_key}")
        return payload

    def _goal_dict(self) -> Dict[str, Any]:
        assert self.task_analysis is not None
        return {
            "summary": self.task_analysis.abstract_goal.summary,
            "goal_literals": self.task_analysis.abstract_goal.goal_literals,
            "goal_objects": self.task_analysis.abstract_goal.goal_objects,
            "success_checks": self.task_analysis.abstract_goal.success_checks,
        }

    def _predicate_inventory_dict(self) -> Dict[str, Any]:
        assert self.task_analysis is not None
        return {
            "predicates": self.task_analysis.predicate_inventory.predicates,
            "selection_rationale": self.task_analysis.predicate_inventory.selection_rationale,
            "omitted_predicates": self.task_analysis.predicate_inventory.omitted_predicates,
        }

    def _grounding_summary_dict(self) -> Dict[str, Any]:
        assert self.task_analysis is not None
        grounding = self.task_analysis.grounding_summary
        return {
            "object_bindings": grounding.object_bindings,
            "grounded_goal_literals": grounding.grounded_goal_literals,
            "grounded_predicates": grounding.grounded_predicates,
            "available_skills": grounding.available_skills,
            "missing_references": grounding.missing_references,
            "observed_object_ids": grounding.observed_object_ids,
        }

    def _reset_pddl(self) -> None:
        self.pddl.object_types.clear()
        self.pddl.add_object_type("object")
        self.pddl.predicates.clear()
        self.pddl.predefined_actions.clear()
        self.pddl.llm_generated_actions.clear()
        self.pddl.object_instances.clear()
        self.pddl.initial_literals.clear()
        self.pddl.goal_literals.clear()
        self.pddl.goal_formulas.clear()
        self.pddl._domain_text_cache = None
        self.pddl._problem_text_cache = None
        self.pddl._cache_dirty = True

    def _new_diagnostics(self) -> Dict[str, Any]:
        return {
            "validation_history": [],
            "repair_history": [],
            "llm_call_count": self.llm_analyzer.call_count,
            "llm_elapsed_seconds": self.llm_analyzer.total_elapsed_seconds,
        }

    def _record_validation(self, validation: Dict[str, Any]) -> None:
        if self.task_analysis is None:
            return
        self.task_analysis.diagnostics.setdefault("validation_history", []).append(validation)
        self.task_analysis.diagnostics["last_validation"] = validation
        self.task_analysis.diagnostics["llm_call_count"] = self.llm_analyzer.call_count
        self.task_analysis.diagnostics["llm_elapsed_seconds"] = round(
            self.llm_analyzer.total_elapsed_seconds, 3
        )

    def _refresh_observation_indices(self) -> None:
        self.observed_object_types = {
            sanitize_pddl_name(str(obj.get("object_type", "")))
            for obj in self.observed_objects
            if obj.get("object_type")
        }
        self.observed_predicates = {
            parsed[0]
            for parsed in (
                self._parse_predicate_instance(predicate_str) for predicate_str in self.observed_predicate_strings
            )
            if parsed is not None
        }

    def _extract_global_predicates(self, predicate_signatures: Sequence[str]) -> Set[str]:
        globals_: Set[str] = set()
        for signature in predicate_signatures:
            pred_name, params, _ = self._normalize_predicate_signature(signature)
            if pred_name and not params:
                globals_.add(pred_name)
        return globals_

    @staticmethod
    def _infer_predicate_arities_from_actions(actions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Scan action preconditions/effects and return {predicate_name: arg_count}."""
        counts: Dict[str, int] = {}
        _logical = {"and", "or", "not", "forall", "exists", "when", "imply"}
        for action in actions:
            for formula in (action.get("precondition", ""), action.get("effect", "")):
                for m in re.finditer(r"\(([^()]+)\)", formula or ""):
                    tokens = m.group(1).strip().split()
                    if not tokens or tokens[0].lower() in _logical:
                        continue
                    name = tokens[0].lower()
                    arg_count = len([t for t in tokens[1:] if not t.startswith("(")])
                    if name not in counts or counts[name] < arg_count:
                        counts[name] = arg_count
        return counts

    @staticmethod
    def _normalize_predicate_signature(
        predicate_str: str,
    ) -> Tuple[Optional[str], List[Tuple[str, str]], str]:
        if not predicate_str:
            return None, [], ""
        tokens = re.findall(r"[^\s()]+", predicate_str)
        if not tokens:
            return None, [], ""
        predicate_name = sanitize_pddl_name(tokens[0])
        variables = [token for token in tokens[1:] if token.startswith("?")]
        params: List[Tuple[str, str]] = []
        for idx, variable in enumerate(variables):
            clean_name = sanitize_pddl_name(variable.lstrip("?") or f"arg{idx+1}")
            params.append((clean_name, "object"))
        normalized = f"({predicate_name}{(' ' + ' '.join(variables)) if variables else ''})"
        return predicate_name, params, normalized

    @staticmethod
    def _normalize_action_parameters(parameters: Sequence[str]) -> List[Tuple[str, str]]:
        normalized: List[Tuple[str, str]] = []
        for idx, parameter in enumerate(parameters):
            if not isinstance(parameter, str):
                continue
            raw_name = parameter.split("-")[0].strip()
            clean_name = sanitize_pddl_name(raw_name.lstrip("?") or f"arg{idx+1}")
            normalized.append((clean_name, "object"))
        return normalized

    @staticmethod
    def _extract_predicate_usages(formula: str) -> Set[Tuple[str, int]]:
        usages: Set[Tuple[str, int]] = set()
        for match in re.finditer(r"\(([^()]+)\)", formula or ""):
            tokens = match.group(1).strip().split()
            if not tokens:
                continue
            pred_name = sanitize_pddl_name(tokens[0])
            if pred_name in LOGICAL_OPERATORS:
                continue
            usages.add((pred_name, max(len(tokens) - 1, 0)))
        return usages

    @staticmethod
    def _infer_action_predicate_arities(actions: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        inferred: Dict[str, int] = {}
        for action in actions:
            for formula in (action.get("precondition", ""), action.get("effect", "")):
                for pred_name, arity in PDDLDomainMaintainer._extract_predicate_usages(formula):
                    inferred[pred_name] = max(inferred.get(pred_name, 0), arity)
        return inferred

    @staticmethod
    def _parse_predicate_instance(predicate_str: str) -> Optional[Tuple[str, List[str], bool]]:
        predicate_str = sanitize_pddl_formula(predicate_str).strip()
        if not predicate_str:
            return None

        negated = False
        text = predicate_str
        if text.startswith("(not "):
            negated = True
            text = text[5:-1].strip()

        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()

        tokens = text.split()
        if not tokens:
            return None
        pred_name = sanitize_pddl_name(tokens[0])
        if pred_name in LOGICAL_OPERATORS:
            return None
        return pred_name, [str(token) for token in tokens[1:]], negated

    @staticmethod
    def _parse_literal_formula(formula: str) -> Optional[Tuple[str, List[str], bool]]:
        parsed = PDDLDomainMaintainer._parse_predicate_instance(formula)
        if parsed is None:
            return None
        predicate, arguments, negated = parsed
        if any(arg.startswith("?") for arg in arguments):
            return None
        return predicate, arguments, negated

    def _resolved_goal_formulas(self) -> List[str]:
        assert self.task_analysis is not None
        grounded = self.task_analysis.grounding_summary.grounded_goal_literals
        if grounded:
            formulas = list(grounded)
        else:
            formulas = [
                self._apply_bindings_to_formula(formula, self.task_analysis.grounding_summary.object_bindings)
                for formula in self.task_analysis.abstract_goal.goal_literals
            ]
        return [sanitize_pddl_formula(formula) for formula in formulas if formula]

    @staticmethod
    def _apply_bindings_to_formula(formula: str, bindings: Dict[str, List[str]]) -> str:
        updated = formula
        for symbolic_name, object_ids in bindings.items():
            if not object_ids:
                continue
            updated = re.sub(rf"\b{re.escape(symbolic_name)}\b", object_ids[0], updated)
        return updated

    @staticmethod
    def _strip_negative_preconditions(formula: str) -> str:
        """Remove negative preconditions for STRIPS-only solvers."""

        if "(not " not in formula:
            return formula

        text = formula.strip()
        if text.startswith("(and") and text.endswith(")"):
            inner = text[4:-1].strip()
            clauses: List[str] = []
            depth = 0
            start = None
            for idx, char in enumerate(inner):
                if char == "(":
                    if depth == 0:
                        start = idx
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and start is not None:
                        clauses.append(inner[start : idx + 1].strip())
                        start = None
            filtered = [clause for clause in clauses if not clause.startswith("(not ")]
            if not filtered:
                return "(and)"
            if len(filtered) == 1:
                return filtered[0]
            return f"(and {' '.join(filtered)})"

        if text.startswith("(not "):
            return "(and)"
        return text

    def _normalize_action_schema_library(
        self,
        library: ActionSchemaLibrary,
    ) -> ActionSchemaLibrary:
        """Normalize actions into a solver-safe STRIPS subset."""

        return ActionSchemaLibrary(
            actions=[
                {
                    "name": action.get("name", ""),
                    "parameters": self._as_string_list(action.get("parameters")),
                    "precondition": self._strip_negative_preconditions(
                        sanitize_pddl_formula(str(action.get("precondition", "(and)")) or "(and)")
                    ),
                    "effect": sanitize_pddl_formula(str(action.get("effect", "(and)")) or "(and)"),
                    "description": str(action.get("description", "")).strip(),
                }
                for action in library.actions
                if isinstance(action, dict)
            ],
            planning_notes=list(library.planning_notes),
        )

    def _get_prompt_template(self, template_key: str) -> str:
        template = self.prompt_templates.get(template_key)
        if template is None:
            raise KeyError(f"Missing prompt template: {template_key}")
        return template

    @staticmethod
    def _normalize_actions(actions: Any) -> List[Dict[str, Any]]:
        if not isinstance(actions, list):
            return []
        normalized = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            normalized.append(
                {
                    "name": str(action.get("name", "")).strip(),
                    "parameters": PDDLDomainMaintainer._as_string_list(action.get("parameters")),
                    "precondition": sanitize_pddl_formula(str(action.get("precondition", ""))),
                    "effect": sanitize_pddl_formula(str(action.get("effect", ""))),
                    "description": str(action.get("description", "")).strip(),
                }
            )
        return normalized

    @staticmethod
    def _as_string_list(value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]
