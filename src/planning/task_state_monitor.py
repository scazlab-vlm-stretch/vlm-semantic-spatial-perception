"""
Task State Monitor

Monitors PDDL domain and problem state to determine what action the system should take.
Makes intelligent decisions about exploration, planning, or domain refinement.
"""

import asyncio
from typing import Dict, List, Optional, Tuple

from .pddl_representation import PDDLRepresentation
from .pddl_domain_maintainer import PDDLDomainMaintainer
from .utils.task_types import TaskState, TaskStateDecision


class TaskStateMonitor:
    """
    Monitors task execution state and determines appropriate actions.

    Makes decisions between:
    - EXPLORE: Gather more observations (missing goal objects, insufficient state)
    - PLAN_AND_EXECUTE: Generate plan with PDDL planner and execute
    - REFINE_DOMAIN: Domain incomplete (missing predicates/actions)
    - GOAL_UNREACHABLE: Cannot achieve goal with current knowledge
    - COMPLETE: Task finished

    Example:
        >>> monitor = TaskStateMonitor(domain_maintainer, pddl_repr)
        >>>
        >>> # Check current state
        >>> decision = await monitor.determine_state()
        >>>
        >>> if decision.state == TaskState.EXPLORE:
        >>>     # Continue observation
        >>>     print(f"Need to find: {decision.blockers}")
        >>> elif decision.state == TaskState.PLAN_AND_EXECUTE:
        >>>     # Ready for planning
        >>>     plan = call_pddl_planner(pddl_repr)
        >>> elif decision.state == TaskState.REFINE_DOMAIN:
        >>>     # Need domain updates
        >>>     await domain_maintainer.refine_domain_from_observations(...)
    """

    def __init__(
        self,
        domain_maintainer: PDDLDomainMaintainer,
        pddl_representation: PDDLRepresentation,
        min_observations_before_planning: int = 3,
        exploration_timeout_seconds: float = 30.0
    ):
        """
        Initialize task state monitor.

        Args:
            domain_maintainer: Domain maintainer to query
            pddl_representation: PDDL representation to monitor
            min_observations_before_planning: Minimum objects before attempting plan
            exploration_timeout_seconds: Max time to spend exploring before forcing decision
        """
        self.domain_maintainer = domain_maintainer
        self.pddl = pddl_representation
        self.min_observations = min_observations_before_planning
        self.exploration_timeout = exploration_timeout_seconds

        # Tracking
        self.last_decision: Optional[TaskStateDecision] = None
        self.decision_history: List[TaskStateDecision] = []
        self.exploration_start_time: Optional[float] = None

    async def determine_state(self) -> TaskStateDecision:
        """
        Determine current task state and recommended action.

        Analyzes domain completeness, observation coverage, and goal feasibility
        to make intelligent decision about next steps.

        Returns:
            TaskStateDecision with state and reasoning
        """
        # Gather current state information
        domain_stats = await self.domain_maintainer.get_domain_statistics()
        domain_snapshot = await self.pddl.get_domain_snapshot()
        problem_snapshot = await self.pddl.get_problem_snapshot()

        blockers = []
        recommendations = []
        confidence = 0.0

        # Check 1: Do we have goal objects in the problem?
        missing_goal_objects = await self.domain_maintainer.get_missing_goal_objects()

        if missing_goal_objects:
            # EXPLORE state - missing goal-relevant objects
            blockers.append(f"Missing goal objects: {', '.join(missing_goal_objects)}")
            recommendations.append("Continue environmental observation")
            recommendations.append(f"Look for: {', '.join(missing_goal_objects)}")

            decision = TaskStateDecision(
                state=TaskState.EXPLORE,
                confidence=0.9,
                reasoning=f"Task references {len(missing_goal_objects)} object type(s) not yet observed",
                blockers=blockers,
                recommendations=recommendations,
                metrics={
                    "missing_objects": missing_goal_objects,
                    "observed_objects": domain_stats["object_instances"],
                    "goal_coverage": domain_stats["goal_objects_observed"] / max(domain_stats["goal_objects_total"], 1)
                }
            )

            self._record_decision(decision)
            return decision

        # Check 2: Is domain complete?
        domain_complete = await self.domain_maintainer.is_domain_complete()

        if not domain_complete:
            # REFINE_DOMAIN state - domain missing components
            if not domain_snapshot["predicates"]:
                blockers.append("No predicates defined in domain")
            if not (domain_snapshot["predefined_actions"] or domain_snapshot["llm_generated_actions"]):
                blockers.append("No actions defined in domain")

            recommendations.append("Refine domain with current observations")
            recommendations.append("Re-analyze task with observed environment")

            decision = TaskStateDecision(
                state=TaskState.REFINE_DOMAIN,
                confidence=0.95,
                reasoning="Domain incomplete - missing required predicates or actions",
                blockers=blockers,
                recommendations=recommendations,
                metrics={
                    "predicates_defined": len(domain_snapshot["predicates"]),
                    "actions_defined": len(domain_snapshot["predefined_actions"]) + len(domain_snapshot["llm_generated_actions"]),
                    "domain_complete": domain_complete
                }
            )

            self._record_decision(decision)
            return decision

        # Check 3: Do we have sufficient observations?
        num_objects = len(problem_snapshot["object_instances"])

        if num_objects < self.min_observations:
            # EXPLORE state - insufficient observations
            blockers.append(f"Only {num_objects} object(s) observed (minimum: {self.min_observations})")
            recommendations.append("Continue observation to build world model")
            recommendations.append(f"Need at least {self.min_observations - num_objects} more object(s)")

            decision = TaskStateDecision(
                state=TaskState.EXPLORE,
                confidence=0.8,
                reasoning=f"Insufficient observations - only {num_objects} object(s) detected",
                blockers=blockers,
                recommendations=recommendations,
                metrics={
                    "objects_observed": num_objects,
                    "minimum_required": self.min_observations,
                    "progress": num_objects / self.min_observations
                }
            )

            self._record_decision(decision)
            return decision

        # Check 4: Do we have initial state?
        num_initial_literals = len(problem_snapshot["initial_literals"])

        if num_initial_literals == 0:
            # EXPLORE state - no state information
            blockers.append("No initial state literals (empty world state)")
            recommendations.append("Wait for predicate detection")
            recommendations.append("Ensure perception system is tracking PDDL predicates")

            decision = TaskStateDecision(
                state=TaskState.EXPLORE,
                confidence=0.85,
                reasoning="No initial state information available",
                blockers=blockers,
                recommendations=recommendations,
                metrics={
                    "initial_literals": num_initial_literals,
                    "objects": num_objects
                }
            )

            self._record_decision(decision)
            return decision

        # Check 5: Do we have goals?
        num_goal_literals = len(problem_snapshot["goal_literals"]) + len(self.pddl.goal_formulas)

        if num_goal_literals == 0:
            # REFINE_DOMAIN state - no goals set
            blockers.append("No goal state defined")
            recommendations.append("Set goal from task analysis")
            recommendations.append("Ensure task analysis extracted goal predicates")

            decision = TaskStateDecision(
                state=TaskState.REFINE_DOMAIN,
                confidence=0.9,
                reasoning="Goal state not defined",
                blockers=blockers,
                recommendations=recommendations,
                metrics={
                    "goal_literals": num_goal_literals
                }
            )

            self._record_decision(decision)
            return decision

        # Check 6: Validate goal completeness
        goal_valid, goal_issues = self.pddl.validate_goal_completeness()

        if not goal_valid:
            # REFINE_DOMAIN or GOAL_UNREACHABLE
            blockers.extend(goal_issues)

            # Determine if it's refinable or unreachable
            has_undefined_predicates = any("not defined" in issue for issue in goal_issues)
            has_undefined_objects = any("undefined" in issue.lower() for issue in goal_issues)

            if has_undefined_predicates:
                state = TaskState.REFINE_DOMAIN
                reasoning = "Goal references undefined predicates"
                recommendations.append("Add missing predicates to domain")
            elif has_undefined_objects:
                state = TaskState.EXPLORE
                reasoning = "Goal references objects not yet observed"
                recommendations.append("Continue observation to find goal objects")
            else:
                state = TaskState.GOAL_UNREACHABLE
                reasoning = "Goal validation failed - may be unreachable"
                recommendations.append("Review goal definition")
                recommendations.append("Check if goal is achievable with available actions")

            decision = TaskStateDecision(
                state=state,
                confidence=0.85,
                reasoning=reasoning,
                blockers=blockers,
                recommendations=recommendations,
                metrics={
                    "goal_valid": goal_valid,
                    "issues": goal_issues
                }
            )

            self._record_decision(decision)
            return decision

        # Check 7: Validate action completeness
        action_valid, action_issues = self.pddl.validate_action_completeness()

        if not action_valid:
            # REFINE_DOMAIN state
            blockers.extend(action_issues)
            recommendations.append("Add missing action definitions")
            recommendations.append("Ensure LLM-generated actions are complete")

            decision = TaskStateDecision(
                state=TaskState.REFINE_DOMAIN,
                confidence=0.8,
                reasoning="Action definitions incomplete or invalid",
                blockers=blockers,
                recommendations=recommendations,
                metrics={
                    "action_valid": action_valid,
                    "issues": action_issues
                }
            )

            self._record_decision(decision)
            return decision

        # All checks passed - ready for planning!
        recommendations.append("Generate PDDL plan using domain and problem")
        recommendations.append("Execute plan with monitoring")
        recommendations.append("Update world state as execution progresses")

        decision = TaskStateDecision(
            state=TaskState.PLAN_AND_EXECUTE,
            confidence=1.0,
            reasoning="Domain complete, goals defined, sufficient observations - ready for planning",
            blockers=[],
            recommendations=recommendations,
            metrics={
                "domain_complete": True,
                "goal_valid": True,
                "action_valid": True,
                "objects": num_objects,
                "initial_literals": num_initial_literals,
                "goal_literals": num_goal_literals,
                "predicates": len(domain_snapshot["predicates"]),
                "actions": len(domain_snapshot["predefined_actions"]) + len(domain_snapshot["llm_generated_actions"])
            }
        )

        self._record_decision(decision)
        return decision

    async def should_continue_exploration(self) -> Tuple[bool, str]:
        """
        Determine if exploration should continue.

        Returns:
            (should_continue, reason)
        """
        decision = await self.determine_state()

        if decision.state == TaskState.EXPLORE:
            return True, decision.reasoning
        elif decision.state == TaskState.PLAN_AND_EXECUTE:
            return False, "Sufficient information gathered - ready for planning"
        elif decision.state == TaskState.REFINE_DOMAIN:
            return False, "Domain needs refinement before continuing"
        else:
            return False, decision.reasoning

    async def get_exploration_targets(self) -> List[str]:
        """
        Get list of what to look for during exploration.

        Returns:
            List of object types or features to search for
        """
        missing_objects = await self.domain_maintainer.get_missing_goal_objects()
        return missing_objects

    def _record_decision(self, decision: TaskStateDecision) -> None:
        """Record decision in history."""
        self.last_decision = decision
        self.decision_history.append(decision)

        # Keep history bounded
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]

    async def get_decision_summary(self) -> Dict:
        """
        Get summary of recent decisions.

        Returns:
            Dict with decision statistics and trends
        """
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "current_state": None,
                "state_distribution": {},
                "recent_states": []
            }

        # Count state distribution
        state_counts = {}
        for decision in self.decision_history:
            state_name = decision.state.value
            state_counts[state_name] = state_counts.get(state_name, 0) + 1

        # Recent states (last 10)
        recent = [d.state.value for d in self.decision_history[-10:]]

        return {
            "total_decisions": len(self.decision_history),
            "current_state": self.last_decision.state.value if self.last_decision else None,
            "current_confidence": self.last_decision.confidence if self.last_decision else 0.0,
            "state_distribution": state_counts,
            "recent_states": recent,
            "current_blockers": self.last_decision.blockers if self.last_decision else [],
            "current_recommendations": self.last_decision.recommendations if self.last_decision else []
        }
