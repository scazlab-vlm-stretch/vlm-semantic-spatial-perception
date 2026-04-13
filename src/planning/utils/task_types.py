"""
Task analysis and monitoring types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AbstractGoal:
    """Abstract task goal before grounding.

    Args:
        summary: Short natural-language summary of the intended outcome.
        goal_literals: Symbolic goal literals, typically using task-level object references.
        goal_objects: Object references mentioned in the task.
        success_checks: Optional human-readable checks for debugging.

    Example:
        >>> goal = AbstractGoal(
        ...     summary="Retrieve the jar while keeping fragile items intact.",
        ...     goal_literals=["(holding jar)", "(cabinet-open cabinet)"],
        ...     goal_objects=["jar", "cabinet"],
        ... )
    """

    summary: str
    goal_literals: List[str] = field(default_factory=list)
    goal_objects: List[str] = field(default_factory=list)
    success_checks: List[str] = field(default_factory=list)


@dataclass
class PredicateInventory:
    """Minimal predicate set required to express and solve a task.

    Args:
        predicates: Predicate signatures in PDDL-style notation.
        selection_rationale: Optional notes about why predicates were retained.
        omitted_predicates: Optional notes about intentionally omitted predicates.

    Example:
        >>> inventory = PredicateInventory(
        ...     predicates=["(holding ?obj)", "(in ?obj ?container)", "(hand-empty)"]
        ... )
    """

    predicates: List[str] = field(default_factory=list)
    selection_rationale: List[str] = field(default_factory=list)
    omitted_predicates: List[str] = field(default_factory=list)


@dataclass
class ActionSchemaLibrary:
    """Action schemas derived from the abstract goal and predicate inventory.

    Args:
        actions: PDDL action schema dictionaries.
        planning_notes: Optional notes about action coverage or assumptions.

    Example:
        >>> actions = ActionSchemaLibrary(
        ...     actions=[
        ...         {
        ...             "name": "pick",
        ...             "parameters": ["?obj"],
        ...             "precondition": "(and (hand-empty) (graspable ?obj))",
        ...             "effect": "(and (holding ?obj) (not (hand-empty)))",
        ...         }
        ...     ]
        ... )
    """

    actions: List[Dict[str, Any]] = field(default_factory=list)
    planning_notes: List[str] = field(default_factory=list)


@dataclass
class GroundingSummary:
    """Grounding information that connects symbolic references to the observed world.

    Args:
        object_bindings: Mapping from symbolic object references to detected object IDs.
        grounded_goal_literals: Goal literals rewritten with observed object IDs when possible.
        grounded_predicates: Initial-state predicates that were grounded from perception.
        available_skills: Executable skill names associated with grounded actions.
        missing_references: Symbolic references that could not be grounded yet.
        observed_object_ids: Observed object IDs considered during grounding.

    Example:
        >>> grounding = GroundingSummary(
        ...     object_bindings={"jar": ["jar_1"]},
        ...     grounded_goal_literals=["(holding jar_1)"],
        ... )
    """

    object_bindings: Dict[str, List[str]] = field(default_factory=dict)
    grounded_goal_literals: List[str] = field(default_factory=list)
    grounded_predicates: List[str] = field(default_factory=list)
    available_skills: List[str] = field(default_factory=list)
    missing_references: List[str] = field(default_factory=list)
    observed_object_ids: List[str] = field(default_factory=list)


@dataclass
class TaskAnalysis:
    """Staged task-analysis result used as the planning source of truth.

    Args:
        abstract_goal: Goal layer output.
        predicate_inventory: Predicate layer output.
        action_schemas: Action layer output.
        grounding_summary: Grounding layer output.
        diagnostics: Validation and repair metadata accumulated during the run.

    Example:
        >>> analysis = TaskAnalysis(
        ...     abstract_goal=AbstractGoal(summary="Place the mug on the coaster."),
        ...     predicate_inventory=PredicateInventory(predicates=["(on ?obj ?surface)"]),
        ...     action_schemas=ActionSchemaLibrary(),
        ...     grounding_summary=GroundingSummary(),
        ... )
    """

    abstract_goal: AbstractGoal
    predicate_inventory: PredicateInventory
    action_schemas: ActionSchemaLibrary
    grounding_summary: GroundingSummary
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def goal_object_references(self) -> List[str]:
        """Return symbolic goal-object references."""
        return list(self.abstract_goal.goal_objects)

    def predicate_signatures(self) -> List[str]:
        """Return predicate signatures used by perception and PDDL generation."""
        return list(self.predicate_inventory.predicates)

    def action_context(self) -> List[Dict[str, Any]]:
        """Return action schemas formatted for downstream components."""
        return list(self.action_schemas.actions)

    # ------------------------------------------------------------------
    # Backward-compat properties (map old flat names → new nested types)
    # ------------------------------------------------------------------

    @property
    def goal_predicates(self) -> List[str]:
        return list(self.abstract_goal.goal_literals)

    @property
    def goal_objects(self) -> List[str]:
        return list(self.abstract_goal.goal_objects)

    @property
    def relevant_predicates(self) -> List[str]:
        return list(self.predicate_inventory.predicates)

    @relevant_predicates.setter
    def relevant_predicates(self, value: List[str]) -> None:
        self.predicate_inventory.predicates = list(value)

    @property
    def required_actions(self) -> List[Dict[str, Any]]:
        return list(self.action_schemas.actions)

    @property
    def global_predicates(self) -> List[str]:
        """Zero-arity predicates derived from the predicate inventory."""
        return [
            sig.strip("()").split()[0]
            for sig in self.predicate_inventory.predicates
            if sig.count("?") == 0
        ]


# ---------------------------------------------------------------------------
# Layered Domain Generation Artifacts (L1–L5)
# ---------------------------------------------------------------------------

@dataclass
class L1GoalArtifact:
    """Output of L1: Goal Specification layer."""
    goal_predicates: List[str] = field(default_factory=list)
    goal_objects: List[str] = field(default_factory=list)
    global_predicates: List[str] = field(default_factory=list)
    generation_attempts: int = 1
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class L2PredicateArtifact:
    """Output of L2: Predicate Vocabulary layer."""
    predicate_signatures: List[str] = field(default_factory=list)
    sensed_predicates: List[str] = field(default_factory=list)
    checked_variants: List[str] = field(default_factory=list)
    # Maps predicate name → type classification string (robot_state|object_state|sensed|external|checked)
    type_classifications: Dict[str, str] = field(default_factory=dict)
    generation_attempts: int = 1
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class L3ActionArtifact:
    """Output of L3: Action Schema layer."""
    actions: List[Dict] = field(default_factory=list)
    sensing_actions: List[Dict] = field(default_factory=list)
    generation_attempts: int = 1
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class L4GroundingArtifact:
    """Output of L4: Grounding pre-check (algorithmic)."""
    object_bindings: Dict[str, str]  # action-param-type -> observed object_id example
    warnings: List[str]              # non-blocking feasibility warnings


@dataclass
class L5InitialStateArtifact:
    """Output of L5: Initial State Construction (algorithmic)."""
    true_literals: List[Tuple[str, List[str]]]   # [(pred_name, [arg1, arg2])]
    false_literals: List[Tuple[str, List[str]]]  # always-false (checked-X predicates)


@dataclass
class LayeredDomainArtifact:
    """Complete output of the layered domain generation pipeline."""
    l1: L1GoalArtifact
    l2: L2PredicateArtifact
    l3: L3ActionArtifact
    l4: Optional[L4GroundingArtifact]
    l5: Optional[L5InitialStateArtifact]
    task_description: str
    scene_objects: List[Dict]

    def to_task_analysis(self) -> "TaskAnalysis":
        """Bridge method: convert layered artifacts to the new TaskAnalysis."""
        all_actions = self.l3.actions + self.l3.sensing_actions

        grounded_predicates: List[str] = []
        if self.l5:
            for pred_name, args in self.l5.true_literals:
                if args:
                    grounded_predicates.append(f"({pred_name} {' '.join(args)})")
                else:
                    grounded_predicates.append(f"({pred_name})")

        return TaskAnalysis(
            abstract_goal=AbstractGoal(
                summary=self.task_description,
                goal_literals=self.l1.goal_predicates,
                goal_objects=self.l1.goal_objects,
            ),
            predicate_inventory=PredicateInventory(
                predicates=self.l2.predicate_signatures,
            ),
            action_schemas=ActionSchemaLibrary(
                actions=all_actions,
            ),
            grounding_summary=GroundingSummary(
                grounded_predicates=grounded_predicates,
                observed_object_ids=[o.get("object_id", "") for o in self.scene_objects],
            ),
        )


class TaskState(Enum):
    """Possible states for task execution."""

    EXPLORE = "explore"
    PLAN_AND_EXECUTE = "plan_and_execute"
    REFINE_DOMAIN = "refine_domain"
    GOAL_UNREACHABLE = "goal_unreachable"
    COMPLETE = "complete"


@dataclass
class TaskStateDecision:
    """Decision about the current task state."""

    state: TaskState
    confidence: float
    reasoning: str
    blockers: List[str]
    recommendations: List[str]
    metrics: Dict[str, Any]
