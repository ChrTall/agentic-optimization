"""Modellneutrales Geschäftsprofil für den Anwendungseinstieg."""

from copy import deepcopy
from typing import (
    Generic,
    Literal,
    NotRequired,
    TypeVar,
    TypedDict,
    cast,
)

from agentic_optimization.models.specification import (
    DemandForecast,
)


RecommendationType = Literal[
    "order_quantity",
    "reorder_point",
    "order_quantity_and_reorder_point",
]

ProcurementPattern = Literal[
    "one_time",
    "recurring",
]

ReplenishmentOption = Literal[
    "possible",
    "not_possible",
]

InventoryReviewMode = Literal[
    "continuous",
    "periodic",
]

BusinessProductScope = Literal[
    "single_product",
    "multiple_products",
]

PrimaryGoal = Literal[
    "minimize_cost",
    "maximize_profit",
    "avoid_stockouts",
    "reach_service_level",
]


BusinessProfileFieldId = Literal[
    "recommendation",
    "procurement_pattern",
    "replenishment_during_period",
    "inventory_review",
    "product_scope",
    "primary_goal",
]


PROFILE_FIELD_ALLOWED_VALUES: dict[
    BusinessProfileFieldId,
    frozenset[str],
] = {
    "recommendation": frozenset(
        {
            "order_quantity",
            "reorder_point",
            "order_quantity_and_reorder_point",
        }
    ),
    "procurement_pattern": frozenset(
        {
            "one_time",
            "recurring",
        }
    ),
    "replenishment_during_period": frozenset(
        {
            "possible",
            "not_possible",
        }
    ),
    "inventory_review": frozenset(
        {
            "continuous",
            "periodic",
        }
    ),
    "product_scope": frozenset(
        {
            "single_product",
            "multiple_products",
        }
    ),
    "primary_goal": frozenset(
        {
            "minimize_cost",
            "maximize_profit",
            "avoid_stockouts",
            "reach_service_level",
        }
    ),
}


KnowledgeStatus = Literal[
    "missing",
    "user_unknown",
    "inferred",
    "confirmed",
    "ambiguous",
    "conflict",
]

EvidenceSourceType = Literal[
    "user_selection",
    "free_text",
    "document",
    "q_foundry",
    "system",
]


class ProfileEvidence(TypedDict):
    """Beleg für einen Wert im Geschäftsprofil."""

    source_id: str
    source_type: EvidenceSourceType
    reference: NotRequired[str]
    excerpt: NotRequired[str]


ProfileValue = TypeVar("ProfileValue")


class ProfileField(TypedDict, Generic[ProfileValue]):
    """Ein fachlicher Wert mit Wissensstatus und Herkunft."""

    value: ProfileValue | None
    status: KnowledgeStatus
    evidence: list[ProfileEvidence]
    candidate_values: NotRequired[list[ProfileValue]]


class BusinessProblemProfile(TypedDict):
    """Modellneutrale Beschreibung der betrieblichen Situation."""

    recommendation: ProfileField[RecommendationType]
    procurement_pattern: ProfileField[ProcurementPattern]
    replenishment_during_period: ProfileField[
        ReplenishmentOption
    ]
    inventory_review: ProfileField[InventoryReviewMode]
    product_scope: ProfileField[BusinessProductScope]
    product_ids: ProfileField[list[str]]
    primary_goal: ProfileField[PrimaryGoal]
    forecasts: dict[str, ProfileField[DemandForecast]]


def create_profile_field(
    *,
    status: KnowledgeStatus,
    value: ProfileValue | None = None,
    evidence: list[ProfileEvidence] | None = None,
    candidate_values: list[ProfileValue] | None = None,
) -> ProfileField[ProfileValue]:
    """Erzeugt ein Profilfeld und erzwingt konsistente Zustände."""

    field_evidence = list(evidence or [])
    unresolved_statuses = {
        "missing",
        "user_unknown",
        "ambiguous",
        "conflict",
    }

    if status in unresolved_statuses and value is not None:
        raise ValueError(
            f"Ein Feld mit Status '{status}' darf keinen "
            "bestätigten Wert enthalten."
        )

    if status in {"inferred", "confirmed"} and value is None:
        raise ValueError(
            f"Ein Feld mit Status '{status}' benötigt einen Wert."
        )

    if status != "missing" and not field_evidence:
        raise ValueError(
            f"Ein Feld mit Status '{status}' benötigt mindestens "
            "einen Quellenbeleg."
        )

    if status in {"ambiguous", "conflict"}:
        if candidate_values is None or len(candidate_values) < 2:
            raise ValueError(
                f"Ein Feld mit Status '{status}' benötigt mindestens "
                "zwei mögliche Werte."
            )
    elif candidate_values:
        raise ValueError(
            "Mögliche Werte sind nur bei mehrdeutigen oder "
            "widersprüchlichen Feldern erlaubt."
        )

    field: ProfileField[ProfileValue] = {
        "value": value,
        "status": status,
        "evidence": field_evidence,
    }

    if candidate_values is not None:
        field["candidate_values"] = list(candidate_values)

    return field


def _merge_profile_evidence(
    first: list[ProfileEvidence],
    second: list[ProfileEvidence],
) -> list[ProfileEvidence]:
    """Führt Quellenbelege zusammen, ohne Duplikate zu erzeugen."""

    merged: list[ProfileEvidence] = []

    for evidence in [*first, *second]:
        if evidence not in merged:
            merged.append(deepcopy(evidence))

    return merged


def _collect_candidate_values(
    *fields: ProfileField[ProfileValue],
) -> list[ProfileValue]:
    """Sammelt unterschiedliche mögliche Werte aus Profilfeldern."""

    candidate_values: list[ProfileValue] = []

    for field in fields:
        values = field.get("candidate_values", [])

        if field["value"] is not None:
            values = [field["value"], *values]

        for value in values:
            if value not in candidate_values:
                candidate_values.append(deepcopy(value))

    return candidate_values


def merge_profile_fields(
    current: ProfileField[ProfileValue],
    incoming: ProfileField[ProfileValue],
) -> ProfileField[ProfileValue]:
    """Führt zwei Angaben zusammen, ohne Konflikte zu verstecken."""

    evidence = _merge_profile_evidence(
        current["evidence"],
        incoming["evidence"],
    )

    if current["status"] == "missing":
        merged = deepcopy(incoming)
        merged["evidence"] = evidence
        return merged

    if incoming["status"] == "missing":
        merged = deepcopy(current)
        merged["evidence"] = evidence
        return merged

    if (
        current["status"] == "user_unknown"
        and incoming["status"] == "user_unknown"
    ):
        return create_profile_field(
            status="user_unknown",
            evidence=evidence,
        )

    if (
        current["status"] == "user_unknown"
        and incoming["value"] is not None
    ):
        merged = deepcopy(incoming)
        merged["evidence"] = evidence
        return merged

    if (
        incoming["status"] == "user_unknown"
        and current["value"] is not None
    ):
        merged = deepcopy(current)
        merged["evidence"] = evidence
        return merged

    if (
        current["value"] is not None
        and incoming["value"] is not None
    ):
        if current["value"] == incoming["value"]:
            merged_status: KnowledgeStatus = (
                "confirmed"
                if "confirmed"
                in {
                    current["status"],
                    incoming["status"],
                }
                else "inferred"
            )

            return create_profile_field(
                status=merged_status,
                value=deepcopy(current["value"]),
                evidence=evidence,
            )

        return create_profile_field(
            status="conflict",
            evidence=evidence,
            candidate_values=[
                deepcopy(current["value"]),
                deepcopy(incoming["value"]),
            ],
        )

    candidate_values = _collect_candidate_values(
        current,
        incoming,
    )

    if len(candidate_values) < 2:
        raise ValueError(
            "Die offenen Profilangaben enthalten nicht "
            "genügend mögliche Werte."
        )

    merged_status: KnowledgeStatus = (
        "conflict"
        if (
            current["status"] == "conflict"
            or incoming["status"] == "conflict"
            or current["value"] is not None
            or incoming["value"] is not None
        )
        else "ambiguous"
    )

    return create_profile_field(
        status=merged_status,
        evidence=evidence,
        candidate_values=candidate_values,
    )


def create_empty_business_problem_profile(
) -> BusinessProblemProfile:
    """Erzeugt ein vollständig unbekanntes Geschäftsprofil."""

    return {
        "recommendation": cast(
            ProfileField[RecommendationType],
            create_profile_field(status="missing"),
        ),
        "procurement_pattern": cast(
            ProfileField[ProcurementPattern],
            create_profile_field(status="missing"),
        ),
        "replenishment_during_period": cast(
            ProfileField[ReplenishmentOption],
            create_profile_field(status="missing"),
        ),
        "inventory_review": cast(
            ProfileField[InventoryReviewMode],
            create_profile_field(status="missing"),
        ),
        "product_scope": cast(
            ProfileField[BusinessProductScope],
            create_profile_field(status="missing"),
        ),
        "product_ids": cast(
            ProfileField[list[str]],
            create_profile_field(status="missing"),
        ),
        "primary_goal": cast(
            ProfileField[PrimaryGoal],
            create_profile_field(status="missing"),
        ),
        "forecasts": {},
    }
