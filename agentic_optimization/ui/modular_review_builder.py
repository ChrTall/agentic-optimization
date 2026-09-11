"""
Bereitet modulare Modellspezifikationen für die Prüfung auf.
"""

from copy import deepcopy
from typing import Any, TypedDict

from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
    OBJECTIVE_COMPONENTS,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)
from agentic_optimization.models.proposal_mapper import (
    validate_component_selection_proposal,
)
from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ModelSpecification,
)
from agentic_optimization.models.validator import (
    MissingInput,
    validate_model_specification,
)


class ReviewObjectiveComponent(TypedDict):
    """Ein ausgewählter Zielfunktionsbaustein."""

    component_id: str
    name: str
    effect: str
    reason: str
    sources: list[str]


class ReviewConstraint(TypedDict):
    """Eine ausgewählte Nebenbedingung."""

    component_id: str
    name: str
    scope: str
    product_id: str | None
    parameters: dict[str, Any]
    reason: str
    sources: list[str]


class ModularReviewData(TypedDict):
    """Prüfbare Darstellung einer modularen Spezifikation."""

    core_id: str
    core_name: str
    objective_sense: str
    objective_components: list[ReviewObjectiveComponent]
    constraints: list[ReviewConstraint]
    missing_inputs: list[MissingInput]
    validation_errors: list[str]
    ready_for_confirmation: bool


def _validate_proposal_specification_alignment(
    proposal: ComponentSelectionProposal,
    specification: ModelSpecification,
) -> None:
    """Verhindert die Prüfung einer abweichenden Spezifikation."""

    errors: list[str] = []

    if proposal["core_id"] != specification["core_id"]:
        errors.append(
            "Grundmodell von Vorschlag und Spezifikation stimmt nicht überein."
        )

    if (
        proposal["objective_sense"]
        != specification["objective"]["sense"]
    ):
        errors.append(
            "Zielrichtung von Vorschlag und Spezifikation stimmt nicht überein."
        )

    proposed_objective_ids = [
        component["component_id"]
        for component in proposal["objective_components"]
    ]
    specification_objective_ids = specification[
        "objective"
    ]["component_ids"]

    if proposed_objective_ids != specification_objective_ids:
        errors.append(
            "Zielfunktionsbausteine von Vorschlag und "
            "Spezifikation stimmen nicht überein."
        )

    proposed_constraints = [
        (
            constraint["component_id"],
            constraint.get("product_id"),
        )
        for constraint in proposal["constraints"]
    ]
    specification_constraints = [
        (
            constraint["component_id"],
            constraint.get("product_id"),
        )
        for constraint in specification["constraints"]
    ]

    if proposed_constraints != specification_constraints:
        errors.append(
            "Nebenbedingungen von Vorschlag und "
            "Spezifikation stimmen nicht überein."
        )

    if errors:
        raise ValueError(
            "Inkonsistente Review-Daten: "
            + "; ".join(errors)
        )


def build_modular_review_data(
    proposal: ComponentSelectionProposal,
    specification: ModelSpecification,
) -> ModularReviewData:
    """Erzeugt die Review-Darstellung eines modularen Modells."""

    proposal_errors = (
        validate_component_selection_proposal(proposal)
    )

    if proposal_errors:
        raise ValueError(
            "Ungültiger Bausteinvorschlag: "
            + "; ".join(proposal_errors)
        )

    _validate_proposal_specification_alignment(
        proposal,
        specification,
    )

    registered_core = CORE_REGISTRY[
        specification["core_id"]
    ]
    validation_result = validate_model_specification(
        specification
    )

    objective_components: list[
        ReviewObjectiveComponent
    ] = []

    for proposed_component in proposal[
        "objective_components"
    ]:
        component_id = proposed_component["component_id"]
        definition = OBJECTIVE_COMPONENTS[component_id]

        objective_components.append(
            {
                "component_id": component_id,
                "name": definition["name"],
                "effect": definition["effect"],
                "reason": proposed_component["reason"],
                "sources": list(
                    proposed_component["sources"]
                ),
            }
        )

    constraints: list[ReviewConstraint] = []

    for index, proposed_constraint in enumerate(
        proposal["constraints"]
    ):
        specification_constraint = specification[
            "constraints"
        ][index]
        component_id = proposed_constraint["component_id"]
        definition = CONSTRAINT_COMPONENTS[component_id]

        constraints.append(
            {
                "component_id": component_id,
                "name": definition["name"],
                "scope": definition["scope"],
                "product_id": specification_constraint.get(
                    "product_id"
                ),
                "parameters": deepcopy(
                    specification_constraint["parameters"]
                ),
                "reason": proposed_constraint["reason"],
                "sources": list(
                    proposed_constraint["sources"]
                ),
            }
        )

    return {
        "core_id": specification["core_id"],
        "core_name": registered_core["definition"]["name"],
        "objective_sense": specification[
            "objective"
        ]["sense"],
        "objective_components": objective_components,
        "constraints": constraints,
        "missing_inputs": validation_result["missing_inputs"],
        "validation_errors": validation_result["errors"],
        "ready_for_confirmation": validation_result["is_valid"],
    }