"""Deterministische Auswahlentscheidung für Grundmodelle."""

from typing import Literal, TypedDict

from agentic_optimization.models.core_selection_catalog import (
    build_core_selection_catalog,
)
from agentic_optimization.models.problem_profile import (
    BusinessProblemProfile,
)


CoreSelectionStatus = Literal[
    "selected",
    "needs_clarification",
    "unsupported",
]


class CoreSelectionDecision(TypedDict):
    """Ergebnis der kontrollierten Grundmodell-Auswahl."""

    status: CoreSelectionStatus
    selected_core_id: str | None
    candidate_core_ids: list[str]
    reason: str


def select_model_core(
    profile: BusinessProblemProfile,
) -> CoreSelectionDecision:
    """Wählt einen vollständigen eindeutigen Kern aus."""

    candidates = build_core_selection_catalog(profile)

    candidate_core_ids = [
        candidate["core_id"]
        for candidate in candidates
    ]

    if not candidates:
        return {
            "status": "unsupported",
            "selected_core_id": None,
            "candidate_core_ids": [],
            "reason": (
                "Die bestätigten Geschäftsangaben passen "
                "zu keinem derzeit unterstützten "
                "Grundmodell."
            ),
        }

    if len(candidates) > 1:
        return {
            "status": "needs_clarification",
            "selected_core_id": None,
            "candidate_core_ids": candidate_core_ids,
            "reason": (
                "Die vorhandenen Geschäftsangaben reichen "
                "noch nicht für eine eindeutige Auswahl "
                "des Grundmodells aus."
            ),
        }

    candidate = candidates[0]

    if candidate["unresolved_profile_fields"]:
        return {
            "status": "needs_clarification",
            "selected_core_id": None,
            "candidate_core_ids": candidate_core_ids,
            "reason": (
                "Für das verbleibende Grundmodell sind "
                "noch Geschäftsangaben zu klären."
            ),
        }

    return {
        "status": "selected",
        "selected_core_id": candidate["core_id"],
        "candidate_core_ids": candidate_core_ids,
        "reason": (
            "Die bestätigten Geschäftsangaben passen "
            "eindeutig zu diesem Grundmodell."
        ),
    }