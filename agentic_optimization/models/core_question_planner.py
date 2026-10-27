"""Plant die nächste fachliche Rückfrage zur Grundmodellauswahl."""

from collections.abc import Collection, Mapping
from typing import Literal, TypedDict

from agentic_optimization.models.core_applicability import (
    ApplicabilityPlan,
    CORE_CONDITIONS,
)
from agentic_optimization.models.core_candidate_selection import (
    choose_candidate_condition,
)
from agentic_optimization.models.core_selector import CoreSelectionDecision


class CoreQuestionPlan(TypedDict):
    question_id: str
    kind: Literal[
        "candidate_condition", "applicability_condition", "core_choice"
    ]
    core_id: str | None
    condition_id: str | None
    reason: Literal["missing", "conflict"] | None


def plan_next_core_question(
    core_selection: CoreSelectionDecision,
    *,
    applicability_plan: ApplicabilityPlan | None,
    applicability_core_id: str | None,
    condition_answers: Mapping[str, str],
    asked_question_ids: Collection[str],
) -> CoreQuestionPlan | None:
    """Entscheidet über die Frage, ohne Text zu formulieren oder zu raten."""

    if applicability_plan is not None:
        condition_id = applicability_plan["question_condition_id"]
        if (
            applicability_plan["status"] == "needs_clarification"
            and applicability_core_id is not None
            and condition_id is not None
        ):
            return {
                "question_id": f"core_condition:{condition_id}",
                "kind": "applicability_condition",
                "core_id": applicability_core_id,
                "condition_id": condition_id,
                "reason": applicability_plan["question_reason"],
            }
        return None

    candidates = core_selection["candidate_core_ids"]
    if (
        core_selection["status"] != "needs_clarification"
        or len(candidates) < 2
    ):
        return None

    target = choose_candidate_condition(
        candidates,
        answered_condition_ids=set(condition_answers),
        asked_question_ids=set(asked_question_ids),
    )
    if target is not None:
        core_id, condition_id = target
        return {
            "question_id": f"core_condition:{condition_id}",
            "kind": "candidate_condition",
            "core_id": core_id,
            "condition_id": condition_id,
            "reason": None,
        }

    unknown_conditions = {
        condition_id for condition_id, answer in condition_answers.items()
        if answer == "unknown"
    }
    if any(
        unknown_conditions.intersection(CORE_CONDITIONS[core_id])
        for core_id in candidates
    ):
        return None
    return {
        "question_id": "core_choice",
        "kind": "core_choice",
        "core_id": None,
        "condition_id": None,
        "reason": None,
    }
