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
    SelectionStatus,
    UnresolvedRequirement,
    ExcludedRequirement,
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


class ReviewProduct(TypedDict):
    """Produktdaten für die fachliche Überprüfung."""

    product_id: str
    forecast: dict[str, Any]
    facts: dict[str, Any]


class ModularSelectionIssueData(TypedDict):
    """Darstellung einer unvollständigen Bausteinauswahl."""

    core_id: str
    core_name: str
    selection_status: SelectionStatus
    unresolved_requirements: list[
        UnresolvedRequirement
    ]


class ModularReviewData(TypedDict):
    """Prüfbare Darstellung einer modularen Spezifikation."""

    core_id: str
    core_name: str
    core_description: str
    decision_description: str
    assumptions: list[str]
    excluded_requirements: list[ExcludedRequirement]
    objective_sense: str
    objective_summary: str
    objective_components: list[ReviewObjectiveComponent]
    constraints: list[ReviewConstraint]
    missing_inputs: list[MissingInput]
    validation_errors: list[str]
    ready_for_confirmation: bool
    products: list[ReviewProduct]
    global_facts: dict[str, Any]


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


def build_modular_selection_issue_data(
    proposal: ComponentSelectionProposal,
) -> ModularSelectionIssueData:
    """Bereitet ungelöste Anforderungen für die UI auf."""

    proposal_errors = (
        validate_component_selection_proposal(proposal)
    )

    if proposal_errors:
        raise ValueError(
            "Ungültiger Bausteinvorschlag: "
            + "; ".join(proposal_errors)
        )

    registered_core = CORE_REGISTRY[
        proposal["core_id"]
    ]

    return {
        "core_id": proposal["core_id"],
        "core_name": registered_core[
            "definition"
        ]["name"],
        "selection_status": proposal[
            "selection_status"
        ],
        "unresolved_requirements": deepcopy(
            proposal["unresolved_requirements"]
        ),
    }


def _build_objective_summary(
    objective_sense: str,
    components: list[ReviewObjectiveComponent],
) -> str:
    """Beschreibt die zusammengesetzte Zielfunktion verständlich."""

    if not components:
        return (
            "Es wurden keine Zielfunktionsbausteine "
            "ausgewählt."
        )

    positive_effect = (
        "benefit"
        if objective_sense == "maximize"
        else "cost"
    )
    expression_parts: list[str] = []

    for component in components:
        is_positive = (
            component["effect"] == positive_effect
        )
        component_name = component["name"]

        if not expression_parts:
            expression_parts.append(
                component_name
                if is_positive
                else f"− {component_name}"
            )
        elif is_positive:
            expression_parts.append(
                f" + {component_name}"
            )
        else:
            expression_parts.append(
                f" − {component_name}"
            )

    action = (
        "maximieren"
        if objective_sense == "maximize"
        else "minimieren"
    )

    return f"{''.join(expression_parts)} {action}"


def build_default_review_explanation(
    review: ModularReviewData,
) -> str:
    """
    Erzeugt eine verständliche Erklärung ohne Sprachmodell.
    """

    product_count = len(review["products"])

    if product_count == 1:
        product_text = "Für ein Produkt"
    else:
        product_text = f"Für {product_count} Produkte"

    explanation = (
        f"{product_text} wird das Modell "
        f"„{review['core_name']}“ verwendet. "
        f"{review['decision_description']} "
        f"Das Ziel lautet: "
        f"{review['objective_summary']}."
    )

    constraint_names = list(
        dict.fromkeys(
            constraint["name"]
            for constraint in review["constraints"]
        )
    )

    if constraint_names:
        explanation += (
            " Zusätzlich berücksichtigt das Modell "
            "folgende betriebliche Vorgaben: "
            + ", ".join(constraint_names)
            + "."
        )
    else:
        explanation += (
            " Weitere betriebliche Vorgaben werden "
            "nicht berücksichtigt."
        )

    if review["excluded_requirements"]:
        explanation += (
            " Mindestens eine weitere Anforderung "
            "wird bei der Berechnung ausdrücklich "
            "nicht berücksichtigt."
        )

    return explanation


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

    objective_summary = _build_objective_summary(
        specification["objective"]["sense"],
        objective_components,
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
        "core_description": registered_core[
            "definition"
        ]["description"],
        "decision_description": registered_core[
            "definition"
        ]["decision_description"],
        "products": [
            {
                "product_id": product["product_id"],
                "forecast": deepcopy(
                    product.get("forecast", {})
                ),
                "facts": deepcopy(
                    product.get("facts", {})
                ),
            }
            for product in specification["products"]
        ],
        "assumptions": list(
            registered_core["definition"]["assumptions"]
        ),
        "excluded_requirements": deepcopy(
            specification.get(
                "excluded_requirements",
                [],
            )
        ),
        "global_facts": deepcopy(
            specification["global_facts"]
        ),
        "objective_sense": specification[
            "objective"
        ]["sense"],
        "objective_summary": objective_summary,
        "objective_components": objective_components,
        "constraints": constraints,
        "missing_inputs": validation_result["missing_inputs"],
        "validation_errors": validation_result["errors"],
        "ready_for_confirmation": validation_result["is_valid"],
    }