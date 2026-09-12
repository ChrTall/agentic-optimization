"""
Überführt Agentenvorschläge in technische Modellspezifikationen.
"""

from copy import deepcopy
from typing import Any

from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ConstraintSpecification,
    ModelSpecification,
    ProductSpecification,
)
from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
    OBJECTIVE_COMPONENTS,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_valid_sources(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_has_text(source) for source in value)
    )


def validate_component_selection_proposal(
    proposal: object,
) -> list[str]:
    """Prüft die Struktur eines untrusted Agentenvorschlags."""

    if not isinstance(proposal, dict):
        return [
            "Der Bausteinvorschlag muss als Objekt vorliegen."
        ]

    errors: list[str] = []

    core_id = proposal.get("core_id")

    if not _has_text(core_id):
        errors.append(
            "Der Bausteinvorschlag enthält kein gültiges Grundmodell."
        )
    elif core_id not in CORE_REGISTRY:
        errors.append(
            f"Das Grundmodell '{core_id}' ist nicht registriert."
        )

    if proposal.get("objective_sense") not in {
        "maximize",
        "minimize",
    }:
        errors.append(
            "Die Zielrichtung muss 'maximize' oder 'minimize' sein."
        )

    objective_components = proposal.get(
        "objective_components"
    )

    if not isinstance(objective_components, list):
        errors.append(
            "Die Zielfunktionsbausteine müssen als Liste vorliegen."
        )
    elif not objective_components:
        errors.append(
            "Es wurde kein Zielfunktionsbaustein vorgeschlagen."
        )
    else:
        for index, component in enumerate(
            objective_components
        ):
            if not isinstance(component, dict):
                errors.append(
                    f"Der Zielfunktionsbaustein an Position "
                    f"{index} muss als Objekt vorliegen."
                )
                continue

            component_id = component.get("component_id")

            if not _has_text(component_id):
                errors.append(
                    f"Der Zielfunktionsbaustein an Position "
                    f"{index} enthält keine gültige ID."
                )
            elif component_id not in OBJECTIVE_COMPONENTS:
                errors.append(
                    f"Der Zielfunktionsbaustein "
                    f"'{component_id}' ist nicht registriert."
                )

            elif (
                _has_text(core_id)
                and core_id in CORE_REGISTRY
                and core_id
                not in OBJECTIVE_COMPONENTS[
                    component_id
                ]["compatible_core_ids"]
            ):
                errors.append(
                    f"Der Zielfunktionsbaustein "
                    f"'{component_id}' ist nicht mit "
                    f"'{core_id}' kompatibel."
                )

            if not _has_text(component.get("reason")):
                errors.append(
                    f"Für den Zielfunktionsbaustein an "
                    f"Position {index} fehlt eine Begründung."
                )

            if not _has_valid_sources(
                component.get("sources")
            ):
                errors.append(
                    f"Für den Zielfunktionsbaustein an "
                    f"Position {index} fehlt eine gültige Quelle."
                )
    if (
        _has_text(core_id)
        and core_id in CORE_REGISTRY
        and isinstance(objective_components, list)
    ):
        selected_component_ids = {
            component.get("component_id")
            for component in objective_components
            if isinstance(component, dict)
        }

        required_component_ids = CORE_REGISTRY[
            core_id
        ]["definition"][
            "required_objective_component_ids"
        ]

        for required_component_id in (
            required_component_ids
        ):
            if (
                required_component_id
                not in selected_component_ids
            ):
                errors.append(
                    f"Das Grundmodell '{core_id}' benötigt "
                    "den Zielfunktionsbaustein "
                    f"'{required_component_id}'."
                )


    constraints = proposal.get("constraints")

    if not isinstance(constraints, list):
        errors.append(
            "Die Nebenbedingungen müssen als Liste vorliegen."
        )
    else:
        for index, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                errors.append(
                    f"Die Nebenbedingung an Position "
                    f"{index} muss als Objekt vorliegen."
                )
                continue

            component_id = constraint.get("component_id")

            if not _has_text(component_id):
                errors.append(
                    f"Die Nebenbedingung an Position "
                    f"{index} enthält keine gültige ID."
                )
            elif component_id not in CONSTRAINT_COMPONENTS:
                errors.append(
                    f"Die Nebenbedingung "
                    f"'{component_id}' ist nicht registriert."
                )

            elif (
                _has_text(core_id)
                and core_id in CORE_REGISTRY
                and core_id
                not in CONSTRAINT_COMPONENTS[
                    component_id
                ]["compatible_core_ids"]
            ):
                errors.append(
                    f"Die Nebenbedingung "
                    f"'{component_id}' ist nicht mit "
                    f"'{core_id}' kompatibel."
                )

            if not isinstance(
                constraint.get("parameters"),
                dict,
            ):
                errors.append(
                    f"Die Parameter der Nebenbedingung an "
                    f"Position {index} müssen als Objekt vorliegen."
                )

            product_id = constraint.get("product_id")

            if (
                product_id is not None
                and not _has_text(product_id)
            ):
                errors.append(
                    f"Die Produkt-ID der Nebenbedingung an "
                    f"Position {index} ist ungültig."
                )

            if not _has_text(constraint.get("reason")):
                errors.append(
                    f"Für die Nebenbedingung an Position "
                    f"{index} fehlt eine Begründung."
                )

            if not _has_valid_sources(
                constraint.get("sources")
            ):
                errors.append(
                    f"Für die Nebenbedingung an Position "
                    f"{index} fehlt eine gültige Quelle."
                )

    return list(dict.fromkeys(errors))


def build_specification_from_proposal(
    proposal: ComponentSelectionProposal,
    products: list[ProductSpecification],
    global_facts: dict[str, Any],
) -> ModelSpecification:
    """Erzeugt deterministisch eine Modellspezifikation."""

    proposal_errors = (
        validate_component_selection_proposal(proposal)
    )

    if proposal_errors:
        raise ValueError(
            "Ungültiger Bausteinvorschlag: "
            + "; ".join(proposal_errors)
        )

    constraints: list[ConstraintSpecification] = []

    for proposed_constraint in proposal["constraints"]:
        constraint: ConstraintSpecification = {
            "component_id": proposed_constraint["component_id"],
            "parameters": deepcopy(
                proposed_constraint["parameters"]
            ),
        }

        product_id = proposed_constraint.get("product_id")

        if product_id is not None:
            constraint["product_id"] = product_id

        constraints.append(constraint)

    return {
        "core_id": proposal["core_id"],
        "products": deepcopy(products),
        "global_facts": deepcopy(global_facts),
        "objective": {
            "sense": proposal["objective_sense"],
            "component_ids": [
                component["component_id"]
                for component in proposal[
                    "objective_components"
                ]
            ],
        },
        "constraints": constraints,
    }