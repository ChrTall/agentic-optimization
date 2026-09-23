"""Modellneutraler Auswahlkatalog registrierter Grundmodelle."""

from copy import deepcopy
from typing import TypedDict

from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)
from agentic_optimization.models.problem_profile import (
    BusinessProblemProfile,
    BusinessProfileFieldId,
)


class CoreSelectionRule(TypedDict):
    """Strukturelle Voraussetzungen eines Grundmodells."""

    core_id: str
    description: str
    compatible_profile_values: dict[
        BusinessProfileFieldId,
        frozenset[str],
    ]


class CoreSelectionCandidate(TypedDict):
    """Für die agentische Auswahl freigegebener Grundmodellkern."""

    core_id: str
    name: str
    description: str
    unresolved_profile_fields: list[
        BusinessProfileFieldId
    ]
    compatible_profile_values: dict[
        BusinessProfileFieldId,
        list[str],
    ]


CORE_SELECTION_RULES: tuple[CoreSelectionRule, ...] = (
    {
        "core_id": "single_period_newsvendor",
        "description": (
            "Einmalige Bestellentscheidung vor einem "
            "begrenzten Verkaufs- oder Bedarfszeitraum, "
            "in dem keine Nachbestellung möglich ist."
        ),
        "compatible_profile_values": {
            "recommendation": frozenset(
                {"order_quantity"}
            ),
            "procurement_pattern": frozenset(
                {"one_time"}
            ),
            "replenishment_during_period": frozenset(
                {"not_possible"}
            ),
        },
    },
    {
        "core_id": "eoq",
        "description": (
            "Wiederkehrende gleichartige Bestellzyklen, "
            "bei denen eine wirtschaftliche Bestellmenge "
            "bestimmt werden soll."
        ),
        "compatible_profile_values": {
            "recommendation": frozenset(
                {"order_quantity"}
            ),
            "primary_goal": frozenset(
                {"minimize_cost"}
            ),
            "procurement_pattern": frozenset(
                {"recurring"}
            ),
        },
    },
    {
        "core_id": "continuous_review_q_r",
        "description": (
            "Wiederkehrende Bestellungen bei laufender "
            "Bestandsüberwachung, für die Bestellmenge "
            "und/oder Meldebestand bestimmt werden."
        ),
        "compatible_profile_values": {
            "recommendation": frozenset(
                {
                    "reorder_point",
                    "order_quantity_and_reorder_point",
                }
            ),
            "primary_goal": frozenset({"minimize_cost"}),
            "procurement_pattern": frozenset(
                {"recurring"}
            ),
            "inventory_review": frozenset(
                {"continuous"}
            ),
        },
    },
)


def _is_compatible_with_resolved_profile(
    profile: BusinessProblemProfile,
    rule: CoreSelectionRule,
) -> bool:
    """Verwirft ausschließlich eindeutig widersprechende Kerne."""

    for field_id, allowed_values in rule[
        "compatible_profile_values"
    ].items():
        profile_field = profile[field_id]

        if profile_field["status"] not in {
            "inferred",
            "confirmed",
        }:
            continue

        if profile_field["value"] not in allowed_values:
            return False

    return True


def _get_unresolved_profile_fields(
    profile: BusinessProblemProfile,
    rule: CoreSelectionRule,
) -> list[BusinessProfileFieldId]:
    """Ermittelt noch ungeklärte Voraussetzungen eines Kerns."""

    return [
        field_id
        for field_id in rule["compatible_profile_values"]
        if field_id != "primary_goal"
        and profile[field_id]["status"]
        not in {"inferred", "confirmed"}
    ]


def build_core_selection_catalog(
    profile: BusinessProblemProfile,
) -> list[CoreSelectionCandidate]:
    """Liefert die mit bekannten Profilwerten kompatiblen Kerne."""

    candidates: list[CoreSelectionCandidate] = []

    for rule in CORE_SELECTION_RULES:
        registered_core = CORE_REGISTRY.get(
            rule["core_id"]
        )

        if registered_core is None:
            raise ValueError(
                f"Das Grundmodell '{rule['core_id']}' "
                "ist nicht registriert."
            )

        if not _is_compatible_with_resolved_profile(
            profile,
            rule,
        ):
            continue

        definition = registered_core["definition"]

        candidates.append(
            {
                "core_id": rule["core_id"],
                "name": definition["name"],
                "description": rule["description"],
                "unresolved_profile_fields": (
                    _get_unresolved_profile_fields(
                        profile,
                        rule,
                    )
                ),
                "compatible_profile_values": {
                    field_id: sorted(values)
                    for field_id, values in rule[
                        "compatible_profile_values"
                    ].items()
                },
            }
        )

    return deepcopy(candidates)