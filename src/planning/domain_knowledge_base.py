"""
Domain Knowledge Base (DKB)

Lightweight JSON-based persistence for predicates, action schemas, and execution
history that accumulates across tasks. Implements the DKB architecture described
in gtamp-update-plan.md §V.

Files written:
    <dkb_dir>/predicate_library.json     — known predicate signatures by name
    <dkb_dir>/action_schema_library.json — known action schemas by name
    <dkb_dir>/execution_history.jsonl    — append-only log of all pipeline runs

Usage:
    dkb = DomainKnowledgeBase(Path("outputs/dkb"))
    dkb.load()

    hints = dkb.get_predicate_hints("pick up the red block")
    dkb.record_execution("pick up the red block", artifact, solver_result)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class DomainKnowledgeBase:
    """
    Persistent store for domain knowledge accumulated across tasks.

    Predicate and action libraries grow monotonically — new entries are
    appended, existing entries are updated. Execution history is append-only.
    """

    def __init__(self, dkb_dir: Optional[Path] = None) -> None:
        self.dkb_dir = Path(dkb_dir) if dkb_dir else Path("outputs/dkb")
        self._predicates: Dict[str, Dict] = {}   # name → {signature, usage_count, ...}
        self._actions: Dict[str, Dict] = {}      # name → {parameters, precondition, effect, ...}

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load existing library files from disk. Creates directory if absent."""
        self.dkb_dir.mkdir(parents=True, exist_ok=True)

        pred_path = self.dkb_dir / "predicate_library.json"
        if pred_path.exists():
            try:
                self._predicates = json.loads(pred_path.read_text(encoding="utf-8"))
            except Exception:
                self._predicates = {}

        action_path = self.dkb_dir / "action_schema_library.json"
        if action_path.exists():
            try:
                self._actions = json.loads(action_path.read_text(encoding="utf-8"))
            except Exception:
                self._actions = {}

    def save(self) -> None:
        """Persist predicate and action libraries to disk."""
        self.dkb_dir.mkdir(parents=True, exist_ok=True)
        (self.dkb_dir / "predicate_library.json").write_text(
            json.dumps(self._predicates, indent=2), encoding="utf-8"
        )
        (self.dkb_dir / "action_schema_library.json").write_text(
            json.dumps(self._actions, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_predicate_hints(self, task: str) -> List[str]:
        """
        Return predicate signatures from the library that may be relevant to `task`.

        Simple heuristic: return signatures whose name appears as a word in the task,
        plus any with high usage_count. Returns at most 20 entries.
        """
        task_lower = task.lower()
        scored: List[tuple] = []
        for name, entry in self._predicates.items():
            score = entry.get("usage_count", 0)
            if any(word in task_lower for word in name.replace("-", " ").split()):
                score += 10
            scored.append((score, entry.get("signature", f"({name})")))

        scored.sort(reverse=True)
        return [sig for _, sig in scored[:20]]

    def get_action_hints(self, task: str) -> List[Dict]:
        """
        Return action schemas from the library relevant to `task`.
        Returns at most 10 entries.
        """
        task_lower = task.lower()
        scored: List[tuple] = []
        for name, entry in self._actions.items():
            score = entry.get("usage_count", 0)
            if any(word in task_lower for word in name.replace("-", " ").split()):
                score += 10
            scored.append((score, {k: v for k, v in entry.items() if k != "usage_count"}))

        scored.sort(reverse=True)
        return [schema for _, schema in scored[:10]]

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_execution(
        self,
        task: str,
        artifact: Any,
        solver_result: Optional[Any] = None,
    ) -> None:
        """
        Record a pipeline execution to the execution history and update libraries.

        Args:
            task: Natural language task description.
            artifact: LayeredDomainArtifact from the generator.
            solver_result: Optional SolverResult (may be None if solver not yet run).
        """
        self.dkb_dir.mkdir(parents=True, exist_ok=True)

        # Update predicate library
        try:
            for sig in getattr(artifact.l2, "predicate_signatures", []):
                name = _name_from_sig(sig)
                if name:
                    entry = self._predicates.get(name, {"signature": sig, "usage_count": 0})
                    entry["usage_count"] = entry.get("usage_count", 0) + 1
                    entry["last_task"] = task
                    self._predicates[name] = entry
        except Exception:
            pass

        # Update action schema library
        try:
            for action in getattr(artifact.l3, "actions", []) + getattr(artifact.l3, "sensing_actions", []):
                name = action.get("name", "")
                if name:
                    entry = self._actions.get(name, {})
                    entry.update({
                        "name": name,
                        "parameters": action.get("parameters", []),
                        "precondition": action.get("precondition", ""),
                        "effect": action.get("effect", ""),
                        "usage_count": entry.get("usage_count", 0) + 1,
                        "last_task": task,
                    })
                    self._actions[name] = entry
        except Exception:
            pass

        self.save()

        # Append to execution history (JSONL)
        history_path = self.dkb_dir / "execution_history.jsonl"
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": task,
        }
        try:
            entry["l1_goal_count"] = len(getattr(artifact.l1, "goal_predicates", []))
            entry["l2_predicate_count"] = len(getattr(artifact.l2, "predicate_signatures", []))
            entry["l3_action_count"] = len(getattr(artifact.l3, "actions", []))
            entry["l1_validation_errors"] = getattr(artifact.l1, "validation_errors", [])
            entry["l2_validation_errors"] = getattr(artifact.l2, "validation_errors", [])
            entry["l3_validation_errors"] = getattr(artifact.l3, "validation_errors", [])
        except Exception:
            pass
        if solver_result is not None:
            try:
                entry["solver_success"] = bool(getattr(solver_result, "success", False))
                entry["plan_length"] = getattr(solver_result, "plan_length", None)
                entry["solver_error"] = getattr(solver_result, "error_message", None)
            except Exception:
                pass

        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _name_from_sig(sig: str) -> Optional[str]:
    """Extract predicate name from a signature like '(on ?obj ?surface)'."""
    import re
    m = re.match(r"^\(?\s*([a-zA-Z][a-zA-Z0-9_-]*)", sig.strip())
    return m.group(1).lower() if m else None
