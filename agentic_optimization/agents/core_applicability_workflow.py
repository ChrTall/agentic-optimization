"""Verbindet Profil, Freitext und explizite Antworten zur Kerneignung."""

from collections.abc import Mapping

from agentic_optimization.models.core_applicability import (
    CORE_CONDITIONS,
    IMMEDIATE_CLARIFICATION_CONDITIONS,
    ApplicabilityPlan,
    Finding,
    UserConditionAnswer,
    plan_core_applicability,
)
from agentic_optimization.models.problem_profile import BusinessProblemProfile


PROFILE_CONDITIONS = {
    "single_period_newsvendor": (
        ("procurement_pattern", "one_time", "single_decision_period"),
        (
            "replenishment_during_period",
            "not_possible",
            "no_replenishment_within_period",
        ),
    ),
    "continuous_review_q_r": (
        ("inventory_review", "continuous", "continuous_inventory_review"),
    ),
}


def plan_candidate_applicability(
    core_id: str,
    profile: BusinessProblemProfile,
    text_findings: Mapping[str, Finding],
    user_answers: Mapping[str, UserConditionAnswer] | None = None,
) -> ApplicabilityPlan:
    """Explizite Antworten überstimmen ältere, ggf. widersprüchliche Texte."""

    if core_id not in CORE_CONDITIONS:
        raise ValueError(f"Unbekanntes Grundmodell: {core_id}")

    # Ein LLM-Befund ist keine ausdrückliche Nutzerbestätigung.
    # Insbesondere ein vermeintlicher Widerspruch darf das Modell
    # nicht ohne Rückfrage endgültig verwerfen.
    findings: dict[str, Finding] = {
        condition_id: (
            "conflict" if finding == "contradicted" else finding
        )
        for condition_id, finding in text_findings.items()
    }
    confirmed_conditions = {
        condition_id
        for condition_id, answer in (user_answers or {}).items()
        if answer == "supported"
    }
    for field_id, expected, condition_id in PROFILE_CONDITIONS.get(
        core_id, ()
    ):
        field = profile[field_id]
        if field["status"] not in {"inferred", "confirmed"}:
            continue
        profile_finding: Finding = (
            "supported" if field["value"] == expected else "contradicted"
        )
        if (
            profile_finding == "supported"
            and field["status"] == "confirmed"
            and any(
                item["source_type"] == "user_selection"
                for item in field["evidence"]
            )
        ):
            confirmed_conditions.add(condition_id)
        current = findings.get(condition_id)
        findings[condition_id] = (
            "conflict"
            if current is not None and current != profile_finding
            else profile_finding
        )

    for condition_id, answer in (user_answers or {}).items():
        if condition_id not in CORE_CONDITIONS[core_id] or answer not in {
            "supported", "contradicted", "unknown"
        }:
            raise ValueError("Ungültige Antwort zur Modellvoraussetzung.")
        if answer == "unknown":
            findings.pop(condition_id, None)
            confirmed_conditions.discard(condition_id)
        else:
            findings[condition_id] = answer

    # Ein zutreffend erscheinender Freitextbeleg ist noch keine ausdrückliche
    # Nutzerantwort. Strukturentscheidende Voraussetzungen werden deshalb
    # erst nach einer Auswahl im Formular oder einer Rückfrage erfüllt.
    for condition_id in IMMEDIATE_CLARIFICATION_CONDITIONS:
        if (
            findings.get(condition_id) == "supported"
            and condition_id not in confirmed_conditions
        ):
            findings.pop(condition_id)

    return plan_core_applicability(
        core_id,
        findings,
        confirmed_conditions=confirmed_conditions,
        deferred_conditions={
            condition_id
            for condition_id, answer in (user_answers or {}).items()
            if answer == "unknown"
        },
    )
