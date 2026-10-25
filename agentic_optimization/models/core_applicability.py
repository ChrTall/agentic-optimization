"""Prüft fachliche Voraussetzungen eines Modellkandidaten."""

from collections.abc import Collection, Mapping, Sequence
from typing import Literal, TypedDict
from agentic_optimization.i18n import t


Finding = Literal["supported", "contradicted", "conflict"]
UserConditionAnswer = Literal["supported", "contradicted", "unknown"]
ApplicabilityStatus = Literal[
    "eligible",
    "needs_clarification",
    "rejected",
]


class ApplicabilityAssessment(TypedDict):
    status: ApplicabilityStatus
    unresolved: list[str]
    contradicted: list[str]
    conflicted: list[str]


FollowUpStatus = Literal[
    "rejected",
    "needs_clarification",
    "provisional",
    "review_ready",
]


class ApplicabilityPlan(TypedDict):
    status: FollowUpStatus
    question_condition_id: str | None
    question_reason: Literal["missing", "conflict"] | None
    pending_confirmation: list[str]
    contradicted: list[str]


# Fehlender Eintrag bedeutet: noch unbekannt, nicht automatisch erfüllt.
CORE_CONDITIONS: dict[str, tuple[str, ...]] = {
    "eoq": (
        "recurring_orders",
        "steady_demand",
        "no_quantity_discounts",
        "no_shortages",
    ),
    "single_period_newsvendor": (
        "single_decision_period",
        "no_replenishment_within_period",
    ),
    "continuous_review_q_r": (
        "recurring_orders",
        "continuous_inventory_review",
        "constant_lead_time",
    ),
}


# Profile fields that describe structural requirements. Their condition
# answers may also correct an older inference when switching candidates.
PROFILE_CONDITIONS = {
    "eoq": (
        ("procurement_pattern", "recurring", "recurring_orders"),
    ),
    "single_period_newsvendor": (
        ("procurement_pattern", "one_time", "single_decision_period"),
        (
            "replenishment_during_period",
            "not_possible",
            "no_replenishment_within_period",
        ),
    ),
    "continuous_review_q_r": (
        ("procurement_pattern", "recurring", "recurring_orders"),
        ("inventory_review", "continuous", "continuous_inventory_review"),
    ),
}


def profile_condition_answers(
    field_id: str,
    condition_answers: Mapping[str, str],
) -> list[tuple[str, str]]:
    """Direkte Antworten zu einem Profilfeld, unabhängig vom Kandidaten."""

    related = {
        (expected, condition_id)
        for mappings in PROFILE_CONDITIONS.values()
        for mapped_field, expected, condition_id in mappings
        if mapped_field == field_id
    }
    return [
        (expected, condition_answers[condition_id])
        for expected, condition_id in sorted(related)
        if condition_id in condition_answers
    ]


# Diese Voraussetzungen bestimmen die Entscheidungsstruktur des
# Grundmodells. Sie müssen vor einem Modellentwurf geklärt werden.
# Andere unbekannte Annahmen können im Review ausdrücklich bestätigt
# werden; ohne Bestätigung darf nicht optimiert werden.
IMMEDIATE_CLARIFICATION_CONDITIONS = frozenset({
    "recurring_orders",
    "single_decision_period",
    "no_replenishment_within_period",
    "continuous_inventory_review",
})


CONDITION_LABELS: dict[str, str] = {
    "recurring_orders": (
        "Der Artikel wird dauerhaft in wiederkehrenden Bestellzyklen beschafft."
    ),
    "steady_demand": (
        "Der Bedarf ist über die Zeit ungefähr gleichmäßig und planbar."
    ),
    "no_quantity_discounts": (
        "Es gelten keine von der Bestellmenge abhängigen "
        "Einkaufspreise oder Mengenrabatte."
    ),
    "no_shortages": (
        "Geplante Fehlbestände oder Rückstände sind kein Teil "
        "der Bestellstrategie."
    ),
    "single_decision_period": (
        "Die Bestellmenge wird einmal vor dem betrachteten "
        "Zeitraum festgelegt."
    ),
    "no_replenishment_within_period": (
        "Während des betrachteten Zeitraums kann nicht "
        "nachbestellt werden."
    ),
    "continuous_inventory_review": (
        "Der Bestand wird laufend statt nur in festen "
        "Abständen überprüft."
    ),
    "constant_lead_time": (
        "Die Lieferzeit wird für dieses Modell als konstant angenommen."
    ),
}


# Hinweise zur Interpretation sind für die Analyse bestimmt, nicht für
# Checkboxen, mit denen Nutzer die Voraussetzung selbst bestätigen.
CONDITION_GUIDANCE: dict[str, str] = {
    "recurring_orders": CONDITION_LABELS["recurring_orders"],
    "steady_demand": (
        CONDITION_LABELS["steady_demand"]
        + " Ein Jahresbedarf allein belegt das nicht."
    ),
    "no_quantity_discounts": CONDITION_LABELS["no_quantity_discounts"],
    "no_shortages": CONDITION_LABELS["no_shortages"],
    "single_decision_period": (
        "Die Bestellmenge wird einmal vor dem betrachteten "
        "Zeitraum festgelegt. Mehrere Verkaufstage allein "
        "widersprechen dem nicht."
    ),
    "no_replenishment_within_period": CONDITION_LABELS[
        "no_replenishment_within_period"
    ],
    "continuous_inventory_review": CONDITION_LABELS[
        "continuous_inventory_review"
    ],
    "constant_lead_time": CONDITION_LABELS["constant_lead_time"],
}

CORE_ASSUMPTION_CONFIRMATION_PREFIX = "confirm_core_assumption_"


def ground_applicability_findings(
    core_id: str,
    proposed: Sequence[dict[str, object]],
    sources: Mapping[str, str],
) -> dict[str, Finding]:
    """Akzeptiert nur Befunde mit nachweisbarem Textbeleg."""

    if core_id not in CORE_CONDITIONS:
        raise ValueError(t("Unbekanntes Grundmodell: {core_id}", core_id=core_id))

    allowed_conditions = set(CORE_CONDITIONS[core_id])
    grounded: dict[str, Finding] = {}

    for item in proposed:
        condition_id = item.get("condition_id")
        status = item.get("status")
        source_id = item.get("source_id")
        excerpt = item.get("excerpt")

        if (
            not isinstance(condition_id, str)
            or condition_id not in allowed_conditions
        ):
            raise ValueError(t("Unbekannte Modellvoraussetzung."))

        if status not in {"supported", "contradicted"}:
            raise ValueError(t("Ungültiger Eignungsbefund."))

        if (
            not isinstance(source_id, str)
            or source_id not in sources
            or not isinstance(excerpt, str)
            or not excerpt.strip()
            or excerpt not in sources[source_id]
        ):
            raise ValueError(
                t("Nicht belegter Befund für {condition_id}.", condition_id=condition_id)
            )

        finding: Finding = (
            "supported" if status == "supported" else "contradicted"
        )
        previous = grounded.get(condition_id)
        grounded[condition_id] = (
            "conflict"
            if previous is not None and previous != finding
            else finding
        )

    return grounded


def assess_core_applicability(
    core_id: str,
    findings: Mapping[str, Finding],
) -> ApplicabilityAssessment:
    """Bewertet belegte, widersprochene und offene Voraussetzungen."""

    if core_id not in CORE_CONDITIONS:
        raise ValueError(t("Unbekanntes Grundmodell: {core_id}", core_id=core_id))

    conditions = CORE_CONDITIONS[core_id]
    unknown_ids = set(findings) - set(conditions)
    if unknown_ids:
        raise ValueError(
            t("Unbekannte Voraussetzungen: {value1}", value1=sorted(unknown_ids))
        )

    if any(
        value not in {"supported", "contradicted", "conflict"}
        for value in findings.values()
    ):
        raise ValueError(t("Ungültiger Befund zur Modelleignung."))

    contradicted = [
        condition
        for condition in conditions
        if findings.get(condition) == "contradicted"
    ]
    conflicted = [
        condition
        for condition in conditions
        if findings.get(condition) == "conflict"
    ]
    unresolved = [
        condition
        for condition in conditions
        if condition not in findings or condition in conflicted
    ]

    if contradicted:
        status: ApplicabilityStatus = "rejected"
    elif unresolved:
        status = "needs_clarification"
    else:
        status = "eligible"

    return {
        "status": status,
        "unresolved": unresolved,
        "contradicted": contradicted,
        "conflicted": conflicted,
    }


def plan_core_applicability(
    core_id: str,
    findings: Mapping[str, Finding],
    *,
    confirmed_conditions: Collection[str] = (),
    deferred_conditions: Collection[str] = (),
) -> ApplicabilityPlan:
    """Plant Klärung und Review, ohne einen Kern endgültig auszuwählen."""

    if core_id not in CORE_CONDITIONS:
        raise ValueError(t("Unbekanntes Grundmodell: {core_id}", core_id=core_id))
    deferred = set(deferred_conditions)
    if deferred - set(CORE_CONDITIONS[core_id]):
        raise ValueError(t("Unbekannte offen gelassene Modellvoraussetzung."))
    effective_findings = {
        condition_id: finding
        for condition_id, finding in findings.items()
        if condition_id not in deferred
    }
    assessment = assess_core_applicability(core_id, effective_findings)
    unknown_confirmations = set(confirmed_conditions) - set(
        CORE_CONDITIONS[core_id]
    )
    if unknown_confirmations:
        raise ValueError(
            t("Unbekannte bestätigte Modellvoraussetzungen: {value1}", value1=sorted(unknown_confirmations))
        )
    pending_confirmation = [
        condition
        for condition in CORE_CONDITIONS[core_id]
        if (
            condition not in confirmed_conditions
            and condition not in assessment["contradicted"]
            and condition not in assessment["conflicted"]
        )
    ]
    base = {
        "pending_confirmation": pending_confirmation,
        "contradicted": assessment["contradicted"],
    }

    if assessment["contradicted"]:
        return {
            **base,
            "status": "rejected",
            "question_condition_id": None,
            "question_reason": None,
        }

    if assessment["conflicted"]:
        return {
            **base,
            "status": "needs_clarification",
            "question_condition_id": assessment["conflicted"][0],
            "question_reason": "conflict",
        }

    immediate_missing = [
        condition
        for condition in assessment["unresolved"]
        if condition in IMMEDIATE_CLARIFICATION_CONDITIONS
    ]
    if immediate_missing:
        question_candidates = [
            condition for condition in immediate_missing
            if condition not in deferred
        ]
        if question_candidates:
            return {
                **base,
                "status": "needs_clarification",
                "question_condition_id": question_candidates[0],
                "question_reason": "missing",
            }
        return {
            **base,
            "status": "needs_clarification",
            "question_condition_id": None,
            "question_reason": None,
        }

    if deferred:
        return {
            **base,
            "status": "needs_clarification",
            "question_condition_id": None,
            "question_reason": None,
        }

    if pending_confirmation:
        return {
            **base,
            "status": "provisional",
            "question_condition_id": None,
            "question_reason": None,
        }

    return {
        **base,
        "status": "review_ready",
        "question_condition_id": None,
        "question_reason": None,
    }
