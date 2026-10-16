"""Kontrollierte Auswahl aus belegten LLM-Modellkandidaten."""

from typing import TypedDict
from agentic_optimization.models.core_applicability import (
    CORE_CONDITIONS,
    IMMEDIATE_CLARIFICATION_CONDITIONS,
    profile_condition_answers,
)
from agentic_optimization.models.core_registry import CORE_REGISTRY
from agentic_optimization.models.core_selection_catalog import CORE_SELECTION_RULES
from agentic_optimization.models.core_selector import CoreSelectionDecision
from agentic_optimization.models.problem_profile import BusinessProblemProfile


class CoreCandidate(TypedDict):
    core_id: str
    source_id: str
    excerpt: str


class CoreCandidateProposal(TypedDict):
    candidates: list[CoreCandidate]
    reason: str


def _matches_resolved_profile(
    core_id: str,
    profile: BusinessProblemProfile,
    condition_answers: dict[str, str],
    *,
    ignore_inferred: bool = False,
) -> bool:
    """Verwirft belegte Widersprüche, ohne aus Antworten ein Modell zu wählen."""

    rule = next(
        (item for item in CORE_SELECTION_RULES if item["core_id"] == core_id),
        None,
    )
    if rule is not None:
        for field_id, compatible_values in rule[
            "compatible_profile_values"
        ].items():
            # The registered decision quantities below are authoritative for
            # recommendations; legacy routing rules can be narrower.
            if field_id == "recommendation":
                continue
            # Answers about the same profile field apply across model
            # candidates. A denial only removes its value; it does not
            # confirm the alternative candidate's assumption.
            direct_answers = profile_condition_answers(
                field_id, condition_answers
            )
            if direct_answers:
                supported = {
                    expected for expected, answer in direct_answers
                    if answer == "supported"
                }
                contradicted = {
                    expected for expected, answer in direct_answers
                    if answer == "contradicted"
                }
                if (
                    len(supported) > 1
                    or supported.intersection(contradicted)
                    or not supported.issubset(compatible_values)
                    or compatible_values.issubset(contradicted)
                ):
                    return False
                continue
            field = profile[field_id]
            if ignore_inferred and field["status"] == "inferred":
                continue
            if (
                field["status"] in {"inferred", "confirmed"}
                and field["value"] not in compatible_values
            ):
                return False

    definition = CORE_REGISTRY[core_id]["definition"]
    recommendation = profile["recommendation"]
    if recommendation["status"] in {"inferred", "confirmed"} and not (
        ignore_inferred and recommendation["status"] == "inferred"
    ):
        requested = recommendation["value"]
        needed = (
            {"order_quantity", "reorder_point"}
            if requested == "order_quantity_and_reorder_point"
            else {requested}
        )
        decision_ids = {
            quantity_id
            for quantity_id, quantity in definition[
                "provided_quantities"
            ].items()
            if quantity["role"] == "decision"
        }
        if not needed.issubset(decision_ids):
            return False

    primary_goal = profile["primary_goal"]
    if primary_goal["status"] in {"inferred", "confirmed"} and not (
        ignore_inferred and primary_goal["status"] == "inferred"
    ):
        fixed_sense = definition.get("required_objective_sense")
        goal = primary_goal["value"]
        goals_with_required_components = definition.get(
            "required_objective_components_by_goal", {}
        )
        if (
            goals_with_required_components
            and goal not in goals_with_required_components
        ):
            return False
        if fixed_sense == "minimize" and goal != "minimize_cost":
            return False
        if fixed_sense == "maximize" and goal != "maximize_profit":
            return False
    return True


def compatible_registered_core_ids(
    profile: BusinessProblemProfile,
    condition_answers: dict[str, str],
    *,
    ignore_inferred: bool = False,
) -> list[str]:
    """Findet nur mögliche Kerne für Rückfragen, ohne einen auszuwählen."""

    rejected_conditions = {
        condition_id
        for condition_id, answer in condition_answers.items()
        if answer == "contradicted"
    }
    return [
        core_id
        for core_id in CORE_REGISTRY
        if (
            _matches_resolved_profile(
                core_id, profile, condition_answers,
                ignore_inferred=ignore_inferred,
            )
            and not rejected_conditions.intersection(CORE_CONDITIONS[core_id])
        )
    ]


def select_from_core_candidates(
    proposal: CoreCandidateProposal,
    profile: BusinessProblemProfile,
    *,
    condition_answers: dict[str, str] | None = None,
    chosen_core_id: str | None = None,
) -> CoreSelectionDecision:
    """Wählt nur registrierte, nicht ausdrücklich widerlegte Kandidaten."""

    rejected_conditions = {
        condition_id
        for condition_id, answer in (condition_answers or {}).items()
        if answer == "contradicted"
    }
    candidates = [
        candidate["core_id"]
        for candidate in proposal["candidates"]
        if (
            candidate["core_id"] in CORE_REGISTRY
            and _matches_resolved_profile(
                candidate["core_id"], profile, condition_answers or {}
            )
            and not rejected_conditions.intersection(
                CORE_CONDITIONS[candidate["core_id"]]
            )
        )
    ]
    if not candidates:
        return {
            "status": "unsupported",
            "selected_core_id": None,
            "candidate_core_ids": [],
            "reason": (
                "Kein vorgeschlagenes Grundmodell ist mit den "
                "bestätigten Angaben vereinbar. " + proposal["reason"]
            ),
        }
    if chosen_core_id in candidates:
        candidates = [chosen_core_id]
    if len(candidates) > 1:
        return {
            "status": "needs_clarification",
            "selected_core_id": None,
            "candidate_core_ids": candidates,
            "reason": (
                "Mehrere Grundmodelle sind plausibel. Bitte klären "
                "Sie, welche Entscheidung Sie benötigen."
            ),
        }
    return {
        "status": "selected",
        "selected_core_id": candidates[0],
        "candidate_core_ids": candidates,
        "reason": "Vorläufiger, belegter Modellkandidat; Annahmen folgen.",
    }


def choose_candidate_condition(
    candidate_core_ids: list[str],
    *,
    answered_condition_ids: set[str],
    asked_question_ids: set[str],
) -> tuple[str, str] | None:
    """Wählt höchstens eine strukturelle Unterscheidungsfrage.

    Die Frage ist ein zusätzliches Eignungssignal, keine automatische
    Bestätigung des anderen Kandidaten.
    """

    for core_id in candidate_core_ids:
        for condition_id in CORE_CONDITIONS[core_id]:
            if (
                condition_id in IMMEDIATE_CLARIFICATION_CONDITIONS
                and sum(
                    condition_id in CORE_CONDITIONS[other_core_id]
                    for other_core_id in candidate_core_ids
                ) == 1
                and condition_id not in answered_condition_ids
                and f"core_condition:{condition_id}" not in asked_question_ids
            ):
                return core_id, condition_id
    return None
