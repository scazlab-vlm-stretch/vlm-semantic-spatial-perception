from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.planning.pddl_domain_maintainer import PDDLDomainMaintainer
from src.planning.pddl_representation import PDDLRepresentation
from src.planning.utils.task_types import (
    AbstractGoal,
    ActionSchemaLibrary,
    GroundingSummary,
    PredicateInventory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHED_WORLDS = REPO_ROOT / "outputs" / "captured_worlds"


class FakeStagedAnalyzer:
    """Deterministic staged analyzer used for migration tests."""

    def __init__(self) -> None:
        self.robot_description = "test robot"
        self.call_count = 0
        self.total_elapsed_seconds = 0.0

    def analyze_goal(self, task_description: str, **_: Any) -> AbstractGoal:
        self.call_count += 1
        if "vegetables" in task_description:
            return AbstractGoal(
                summary="Place the vegetables into the box.",
                goal_literals=["(in red_pepper box)", "(in carrot box)"],
                goal_objects=["red_pepper", "carrot", "box"],
                success_checks=["Both vegetables are inside the box."],
            )
        return AbstractGoal(
            summary="Put the orange marker in the cup.",
            goal_literals=["(in orange_marker cup)"],
            goal_objects=["orange_marker", "cup"],
            success_checks=["The orange marker is inside the cup."],
        )

    def analyze_predicates(
        self,
        task_description: str,
        abstract_goal: AbstractGoal,
        **_: Any,
    ) -> PredicateInventory:
        self.call_count += 1
        return PredicateInventory(
            predicates=[
                "(in ?obj ?container)",
                "(holding ?obj)",
                "(hand-empty)",
                "(graspable ?obj)",
                "(containable ?obj)",
            ],
            selection_rationale=[f"Enough predicates to express `{abstract_goal.summary}`."],
            omitted_predicates=["No extra scene predicates."],
        )

    def analyze_actions(
        self,
        task_description: str,
        **_: Any,
    ) -> ActionSchemaLibrary:
        self.call_count += 1
        if "broken" in task_description:
            place_effect = "(and (hand-empty) (not (holding ?obj)))"
        else:
            place_effect = "(and (in ?obj ?container) (hand-empty) (not (holding ?obj)))"
        return ActionSchemaLibrary(
            actions=[
                {
                    "name": "pick",
                    "parameters": ["?obj"],
                    "precondition": "(and (hand-empty) (graspable ?obj))",
                    "effect": "(and (holding ?obj) (not (hand-empty)))",
                    "description": "Pick up an object.",
                },
                {
                    "name": "place_in",
                    "parameters": ["?obj", "?container"],
                    "precondition": "(and (holding ?obj) (containable ?container))",
                    "effect": place_effect,
                    "description": "Place an object into a container.",
                },
            ],
            planning_notes=["Minimal action set for pick-and-place."],
        )

    def analyze_grounding(
        self,
        task_description: str,
        observed_objects: List[Dict[str, Any]] | None = None,
        observed_predicates: List[str] | None = None,
        predicates: List[str] | None = None,
        **_: Any,
    ) -> GroundingSummary:
        self.call_count += 1
        observed_objects = observed_objects or []
        observed_predicates = observed_predicates or predicates or []
        ids = {obj["object_id"] for obj in observed_objects}
        if "vegetables" in task_description:
            missing = [name for name in ("red_pepper", "carrot", "box") if name not in ids]
            return GroundingSummary(
                object_bindings={
                    "red_pepper": ["red_pepper"] if "red_pepper" in ids else [],
                    "carrot": ["carrot"] if "carrot" in ids else [],
                    "box": ["box"] if "box" in ids else [],
                },
                grounded_goal_literals=["(in red_pepper box)", "(in carrot box)"] if not missing else [],
                grounded_predicates=list(observed_predicates),
                available_skills=["pick", "place_in"],
                missing_references=missing,
                observed_object_ids=sorted(ids),
            )
        missing = [name for name in ("orange_marker", "cup") if name not in ids]
        return GroundingSummary(
            object_bindings={},
            grounded_goal_literals=[],
            grounded_predicates=list(observed_predicates),
            available_skills=["pick", "place_in"],
            missing_references=missing,
            observed_object_ids=sorted(ids),
        )


def _load_world(world_name: str) -> tuple[str, List[Dict[str, Any]]]:
    world_dir = CACHED_WORLDS / world_name
    task = (world_dir / "TASK.md").read_text(encoding="utf-8").strip()
    registry = json.loads((world_dir / "registry.json").read_text(encoding="utf-8"))
    return task, registry["objects"]


@pytest.mark.asyncio
async def test_ground_representation_with_cached_vegetables_world() -> None:
    task, objects = _load_world("vegetables")
    maintainer = PDDLDomainMaintainer(PDDLRepresentation())
    maintainer.llm_analyzer = FakeStagedAnalyzer()

    analysis = await maintainer.build_representation(task)
    stats = await maintainer.ground_representation(
        detected_objects=objects,
        predicates=["graspable red_pepper", "graspable carrot", "containable box"],
    )

    assert analysis.abstract_goal.summary == "Place the vegetables into the box."
    assert stats["grounding_complete"] is True
    assert stats["goal_objects_missing"] == []
    assert set(maintainer.pddl.object_instances) == {"box", "red_pepper", "carrot"}
    assert maintainer.pddl.goal_formulas == []
    assert {lit.predicate for lit in maintainer.pddl.goal_literals} == {"in"}
    assert maintainer.task_analysis.diagnostics["last_validation"]["valid"] is True


@pytest.mark.asyncio
async def test_ground_representation_reports_missing_refs_for_empty_cached_world() -> None:
    task, objects = _load_world("markers")
    maintainer = PDDLDomainMaintainer(PDDLRepresentation())
    maintainer.llm_analyzer = FakeStagedAnalyzer()

    await maintainer.build_representation(task)
    stats = await maintainer.ground_representation(detected_objects=objects, predicates=[])

    assert objects == []
    assert stats["grounding_complete"] is False
    assert set(stats["goal_objects_missing"]) == {"orange_marker", "cup"}
    assert maintainer.task_analysis.grounding_summary.missing_references == ["orange_marker", "cup"]


@pytest.mark.asyncio
async def test_action_layer_repair_updates_only_actions() -> None:
    task, objects = _load_world("vegetables")
    task = f"{task} with broken actions"
    maintainer = PDDLDomainMaintainer(PDDLRepresentation())
    maintainer.llm_analyzer = FakeStagedAnalyzer()

    await maintainer.build_representation(task)
    await maintainer.ground_representation(
        detected_objects=objects,
        predicates=["graspable red_pepper", "graspable carrot", "containable box"],
    )
    validation = maintainer.task_analysis.diagnostics["last_validation"]
    assert maintainer.classify_failure_layer("no plan found", validation) == "actions"
    assert validation["layer_validity"]["actions"] is False

    original_goal = maintainer.task_analysis.abstract_goal.goal_literals.copy()
    original_predicates = maintainer.task_analysis.predicate_inventory.predicates.copy()

    maintainer._request_repair_json = lambda template_key, failure_context: {
        "actions": [
            {
                "name": "pick",
                "parameters": ["?obj"],
                "precondition": "(and (hand-empty) (graspable ?obj))",
                "effect": "(and (holding ?obj) (not (hand-empty)))",
                "description": "Pick up an object.",
            },
            {
                "name": "place_in",
                "parameters": ["?obj", "?container"],
                "precondition": "(and (holding ?obj) (containable ?container))",
                "effect": "(and (in ?obj ?container) (hand-empty) (not (holding ?obj)))",
                "description": "Place an object into a container.",
            },
        ],
        "repair_notes": [template_key, failure_context["error_message"]],
    }

    repair = await maintainer.repair_representation(
        failure_context={"error_message": "no plan found"},
        layer="actions",
    )

    assert maintainer.task_analysis.abstract_goal.goal_literals == original_goal
    assert maintainer.task_analysis.predicate_inventory.predicates == original_predicates
    assert repair["validation"]["valid"] is True
    assert repair["layer"] == "actions"
