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
    ExcludedRequirement,
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

    selection_status = proposal.get(
        "selection_status"
    )

    if selection_status not in {
        "complete",
        "needs_clarification",
        "unsupported",
    }:
        errors.append(
            "Der Auswahlstatus muss 'complete', "
            "'needs_clarification' oder "
            "'unsupported' sein."
        )

    unresolved_requirements = proposal.get(
        "unresolved_requirements"
    )

    if not isinstance(unresolved_requirements, list):
        errors.append(
            "Die ungelösten Anforderungen müssen als "
            "Liste vorliegen."
        )
    else:
        resolutions: set[str] = set()

        for index, requirement in enumerate(
            unresolved_requirements
        ):
            if not isinstance(requirement, dict):
                errors.append(
                    f"Die ungelöste Anforderung an "
                    f"Position {index} muss als Objekt "
                    "vorliegen."
                )
                continue

            if not _has_text(
                requirement.get("description")
            ):
                errors.append(
                    f"Für die ungelöste Anforderung an "
                    f"Position {index} fehlt eine "
                    "Beschreibung."
                )

            resolution = requirement.get("resolution")

            if resolution not in {
                "clarification",
                "unsupported",
            }:
                errors.append(
                    f"Die Auflösungsart der ungelösten "
                    f"Anforderung an Position {index} "
                    "muss 'clarification' oder "
                    "'unsupported' sein."
                )
            else:
                resolutions.add(resolution)

            candidate_component_ids = requirement.get(
                "candidate_component_ids"
            )

            if (
                not isinstance(
                    candidate_component_ids,
                    list,
                )
                or not all(
                    _has_text(component_id)
                    for component_id
                    in candidate_component_ids
                )
            ):
                errors.append(
                    f"Die ungelöste Anforderung an "
                    f"Position {index} benötigt eine Liste "
                    "gültiger Kandidaten-Bausteine."
                )
            elif (
                resolution == "clarification"
                and not candidate_component_ids
            ):
                errors.append(
                    f"Für die fachliche Rückfrage an "
                    f"Position {index} fehlt mindestens "
                    "ein möglicher Baustein."
                )
            elif (
                resolution == "unsupported"
                and candidate_component_ids
            ):
                errors.append(
                    f"Die nicht unterstützte Anforderung "
                    f"an Position {index} darf keine "
                    "Kandidaten-Bausteine enthalten."
                )

            if not _has_text(requirement.get("reason")):
                errors.append(
                    f"Für die ungelöste Anforderung an "
                    f"Position {index} fehlt eine "
                    "Begründung."
                )

            if not _has_valid_sources(
                requirement.get("sources")
            ):
                errors.append(
                    f"Für die ungelöste Anforderung an "
                    f"Position {index} fehlt eine "
                    "gültige Quelle."
                )

        if "unsupported" in resolutions:
            expected_status = "unsupported"
        elif "clarification" in resolutions:
            expected_status = "needs_clarification"
        else:
            expected_status = "complete"

        if (
            selection_status
            in {
                "complete",
                "needs_clarification",
                "unsupported",
            }
            and selection_status != expected_status
        ):
            errors.append(
                f"Der Auswahlstatus "
                f"'{selection_status}' passt nicht zu "
                "den ungelösten Anforderungen; erwartet "
                f"wird '{expected_status}'."
            )

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

    selected_component_ids: set[str] = set()

    if isinstance(objective_components, list):
        selected_component_ids.update(
            component.get("component_id")
            for component in objective_components
            if (
                isinstance(component, dict)
                and _has_text(
                    component.get("component_id")
                )
            )
        )

    if isinstance(constraints, list):
        selected_component_ids.update(
            constraint.get("component_id")
            for constraint in constraints
            if (
                isinstance(constraint, dict)
                and _has_text(
                    constraint.get("component_id")
                )
            )
        )

    if isinstance(unresolved_requirements, list):
        for index, requirement in enumerate(
            unresolved_requirements
        ):
            if (
                not isinstance(requirement, dict)
                or requirement.get("resolution")
                != "clarification"
            ):
                continue

            candidate_component_ids = requirement.get(
                "candidate_component_ids"
            )

            if not isinstance(
                candidate_component_ids,
                list,
            ):
                continue

            for candidate_component_id in (
                candidate_component_ids
            ):
                if not _has_text(candidate_component_id):
                    continue

                component_definition = (
                    OBJECTIVE_COMPONENTS.get(
                        candidate_component_id
                    )
                    or CONSTRAINT_COMPONENTS.get(
                        candidate_component_id
                    )
                )

                if component_definition is None:
                    errors.append(
                        f"Der Kandidaten-Baustein "
                        f"'{candidate_component_id}' der "
                        "fachlichen Rückfrage an Position "
                        f"{index} ist nicht registriert."
                    )
                    continue

                if (
                    _has_text(core_id)
                    and core_id in CORE_REGISTRY
                    and core_id
                    not in component_definition[
                        "compatible_core_ids"
                    ]
                ):
                    errors.append(
                        f"Der Kandidaten-Baustein "
                        f"'{candidate_component_id}' der "
                        "fachlichen Rückfrage an Position "
                        f"{index} ist nicht mit "
                        f"'{core_id}' kompatibel."
                    )

                if (
                    candidate_component_id
                    in selected_component_ids
                ):
                    errors.append(
                        f"Der Kandidaten-Baustein "
                        f"'{candidate_component_id}' der "
                        "fachlichen Rückfrage an Position "
                        f"{index} wurde bereits ausgewählt."
                    )

    return list(dict.fromkeys(errors))


def exclude_unsupported_requirements(
    proposal: ComponentSelectionProposal,
    *,
    confirmation_source: str,
) -> tuple[
    ComponentSelectionProposal,
    list[ExcludedRequirement],
]:
    """
    Entfernt ausschließlich ausdrücklich bestätigte,
    nicht unterstützte Anforderungen aus einem Vorschlag.
    """

    if not _has_text(confirmation_source):
        raise ValueError(
            "Für den Ausschluss wird eine gültige "
            "Bestätigungsquelle benötigt."
        )

    proposal_errors = (
        validate_component_selection_proposal(proposal)
    )

    if proposal_errors:
        raise ValueError(
            "Ungültiger Bausteinvorschlag: "
            + "; ".join(proposal_errors)
        )

    if proposal["selection_status"] != "unsupported":
        raise ValueError(
            "Nur ein nicht unterstützter "
            "Bausteinvorschlag kann auf diese Weise "
            "fortgesetzt werden."
        )

    unsupported_requirements = [
        requirement
        for requirement in proposal[
            "unresolved_requirements"
        ]
        if requirement["resolution"] == "unsupported"
    ]

    if not unsupported_requirements:
        raise ValueError(
            "Der Vorschlag enthält keine nicht "
            "unterstützten Anforderungen."
        )

    remaining_requirements = [
        deepcopy(requirement)
        for requirement in proposal[
            "unresolved_requirements"
        ]
        if requirement["resolution"] != "unsupported"
    ]

    excluded_requirements: list[
        ExcludedRequirement
    ] = [
        {
            "description": requirement["description"],
            "reason": requirement["reason"],
            "sources": list(requirement["sources"]),
            "confirmation_source": confirmation_source,
        }
        for requirement in unsupported_requirements
    ]

    updated_proposal = deepcopy(proposal)
    updated_proposal["unresolved_requirements"] = (
        remaining_requirements
    )
    updated_proposal["selection_status"] = (
        "needs_clarification"
        if remaining_requirements
        else "complete"
    )

    updated_errors = (
        validate_component_selection_proposal(
            updated_proposal
        )
    )

    if updated_errors:
        raise ValueError(
            "Der bereinigte Bausteinvorschlag ist "
            "ungültig: "
            + "; ".join(updated_errors)
        )

    return updated_proposal, excluded_requirements


def build_specification_from_proposal(
    proposal: ComponentSelectionProposal,
    products: list[ProductSpecification],
    global_facts: dict[str, Any],
    *,
    excluded_requirements: (
        list[ExcludedRequirement] | None
    ) = None,
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

    if proposal["selection_status"] != "complete":
        unresolved_descriptions = ", ".join(
            requirement["description"]
            for requirement in proposal[
                "unresolved_requirements"
            ]
        )

        raise ValueError(
            "Aus einem nicht vollständigen "
            "Bausteinvorschlag darf keine "
            "Modellspezifikation erzeugt werden. "
            "Ungelöste Anforderungen: "
            f"{unresolved_descriptions}."
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
        "excluded_requirements": deepcopy(
            excluded_requirements or []
        ),
    }
