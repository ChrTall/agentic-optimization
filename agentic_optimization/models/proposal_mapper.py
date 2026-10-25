"""
Überführt Agentenvorschläge in technische Modellspezifikationen.
"""

from copy import deepcopy
from typing import Any, cast

from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ConstraintSpecification,
    ModelSpecification,
    ProductSpecification,
    ExcludedRequirement,
    ComponentSelectionDecision,
    SelectionStatus,
    UnresolvedRequirement,
)
from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
    OBJECTIVE_COMPONENTS,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)
from agentic_optimization.i18n import t


OBJECTIVE_SENSE_BY_GOAL = {
    "maximize_profit": "maximize",
    "minimize_cost": "minimize",
}


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_valid_sources(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_has_text(source) for source in value)
    )


def _derive_selection_status(
    unresolved_requirements: object,
) -> SelectionStatus:
    """Leitet den Status aus offenen Anforderungen ab."""

    if not isinstance(unresolved_requirements, list):
        return "complete"

    resolutions = {
        requirement.get("resolution")
        for requirement in unresolved_requirements
        if isinstance(requirement, dict)
    }

    if "unsupported" in resolutions:
        return "unsupported"

    if "clarification" in resolutions:
        return "needs_clarification"

    return "complete"


def build_component_selection_proposal(
    *,
    core_id: str,
    decision: ComponentSelectionDecision,
    objective_goal: str | None = None,
) -> ComponentSelectionProposal:
    """Ergänzt technische Felder einer fachlichen Auswahl."""

    unresolved_requirements = deepcopy(
        decision.get("unresolved_requirements")
    )

    objective_components = deepcopy(
        decision.get("objective_components")
    )

    registered_core = CORE_REGISTRY.get(core_id)
    core_definition = (
        registered_core["definition"]
        if registered_core is not None else {}
    )

    if (
        isinstance(objective_components, list)
        and registered_core is not None
    ):
        selected_component_ids = {
            component.get("component_id")
            for component in objective_components
            if isinstance(component, dict)
        }

        required_component_ids = [
            *core_definition["required_objective_component_ids"],
            *core_definition.get(
                "required_objective_components_by_goal", {}
            ).get(objective_goal or "", []),
        ]

        for component_id in required_component_ids:
            if component_id in selected_component_ids:
                continue

            objective_components.append(
                {
                    "component_id": component_id,
                    "reason": (
                        t("Vom ausgewählten Grundmodell verpflichtend benötigt.")
                    ),
                    "sources": ["core_definition"],
                }
            )

    objective_sense = core_definition.get("required_objective_sense")
    if objective_sense is None:
        goal_components = core_definition.get(
            "required_objective_components_by_goal", {}
        )
        objective_sense = (
            OBJECTIVE_SENSE_BY_GOAL.get(objective_goal)
            if objective_goal in goal_components
            else decision.get("objective_sense")
        )

    proposal = {
        "core_id": core_id,
        "selection_status": _derive_selection_status(
            unresolved_requirements
        ),
        "objective_sense": objective_sense,
        "objective_components": objective_components,
        "constraints": deepcopy(
            decision.get("constraints")
        ),
        "unresolved_requirements": (
            unresolved_requirements
        ),
    }

    return cast(
        ComponentSelectionProposal,
        proposal,
    )


def validate_component_selection_proposal(
    proposal: object,
    *,
    allow_selected_candidate_overlap: bool = False,
) -> list[str]:
    """Prüft die Struktur eines untrusted Agentenvorschlags."""

    if not isinstance(proposal, dict):
        return [
            t("Der Bausteinvorschlag muss als Objekt vorliegen.")
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
            t("Der Auswahlstatus muss 'complete', 'needs_clarification' oder 'unsupported' sein.")
        )

    unresolved_requirements = proposal.get(
        "unresolved_requirements"
    )

    if not isinstance(unresolved_requirements, list):
        errors.append(
            t("Die ungelösten Anforderungen müssen als Liste vorliegen.")
        )
    else:
        resolutions: set[str] = set()

        for index, requirement in enumerate(
            unresolved_requirements
        ):
            if not isinstance(requirement, dict):
                errors.append(
                    t("Die ungelöste Anforderung an Position {index} muss als Objekt vorliegen.", index=index)
                )
                continue

            if not _has_text(
                requirement.get("description")
            ):
                errors.append(
                    t("Für die ungelöste Anforderung an Position {index} fehlt eine Beschreibung.", index=index)
                )

            resolution = requirement.get("resolution")

            if resolution not in {
                "clarification",
                "unsupported",
            }:
                errors.append(
                    t("Die Auflösungsart der ungelösten Anforderung an Position {index} muss 'clarification' oder 'unsupported' sein.", index=index)
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
                    t("Die ungelöste Anforderung an Position {index} benötigt eine Liste gültiger Kandidaten-Bausteine.", index=index)
                )
            elif (
                resolution == "clarification"
                and not candidate_component_ids
            ):
                errors.append(
                    t("Für die fachliche Rückfrage an Position {index} fehlt mindestens ein möglicher Baustein.", index=index)
                )
            elif (
                resolution == "unsupported"
                and candidate_component_ids
            ):
                errors.append(
                    t("Die nicht unterstützte Anforderung an Position {index} darf keine Kandidaten-Bausteine enthalten.", index=index)
                )

            if not _has_text(requirement.get("reason")):
                errors.append(
                    t("Für die ungelöste Anforderung an Position {index} fehlt eine Begründung.", index=index)
                )

            if not _has_valid_sources(
                requirement.get("sources")
            ):
                errors.append(
                    t("Für die ungelöste Anforderung an Position {index} fehlt eine gültige Quelle.", index=index)
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
                t("Der Auswahlstatus '{selection_status}' passt nicht zu den ungelösten Anforderungen; erwartet wird '{expected_status}'.", selection_status=selection_status, expected_status=expected_status)
            )

    core_id = proposal.get("core_id")

    if not _has_text(core_id):
        errors.append(
            t("Der Bausteinvorschlag enthält kein gültiges Grundmodell.")
        )
    elif core_id not in CORE_REGISTRY:
        errors.append(
            t("Das Grundmodell '{core_id}' ist nicht registriert.", core_id=core_id)
        )

    if proposal.get("objective_sense") not in {
        "maximize",
        "minimize",
    }:
        errors.append(
            t("Die Zielrichtung muss 'maximize' oder 'minimize' sein.")
        )
    elif (
        _has_text(core_id)
        and core_id in CORE_REGISTRY
        and (
            required_sense := CORE_REGISTRY[core_id][
                "definition"
            ].get("required_objective_sense")
        ) is not None
        and proposal["objective_sense"] != required_sense
    ):
        errors.append(
            t("Das Grundmodell '{core_id}' erfordert die Zielrichtung '{required_sense}'.", core_id=core_id, required_sense=required_sense)
        )

    objective_components = proposal.get(
        "objective_components"
    )

    if not isinstance(objective_components, list):
        errors.append(
            t("Die Zielfunktionsbausteine müssen als Liste vorliegen.")
        )
    elif not objective_components:
        errors.append(
            t("Es wurde kein Zielfunktionsbaustein vorgeschlagen.")
        )
    else:
        for index, component in enumerate(
            objective_components
        ):
            if not isinstance(component, dict):
                errors.append(
                    t("Der Zielfunktionsbaustein an Position {index} muss als Objekt vorliegen.", index=index)
                )
                continue

            component_id = component.get("component_id")

            if not _has_text(component_id):
                errors.append(
                    t("Der Zielfunktionsbaustein an Position {index} enthält keine gültige ID.", index=index)
                )
            elif component_id not in OBJECTIVE_COMPONENTS:
                errors.append(
                    t("Der Zielfunktionsbaustein '{component_id}' ist nicht registriert.", component_id=component_id)
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
                    t("Der Zielfunktionsbaustein '{component_id}' ist nicht mit '{core_id}' kompatibel.", component_id=component_id, core_id=core_id)
                )

            if not _has_text(component.get("reason")):
                errors.append(
                    t("Für den Zielfunktionsbaustein an Position {index} fehlt eine Begründung.", index=index)
                )

            if not _has_valid_sources(
                component.get("sources")
            ):
                errors.append(
                    t("Für den Zielfunktionsbaustein an Position {index} fehlt eine gültige Quelle.", index=index)
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

        core_definition = CORE_REGISTRY[core_id]["definition"]
        required_component_ids = list(
            core_definition["required_objective_component_ids"]
        )
        goal_components = core_definition.get(
            "required_objective_components_by_goal", {}
        )
        if goal_components:
            matching_goals = [
                goal for goal in goal_components
                if OBJECTIVE_SENSE_BY_GOAL.get(goal)
                == proposal.get("objective_sense")
            ]
            if not matching_goals:
                errors.append(
                    t("Das Grundmodell '{core_id}' unterstützt die angegebene Zielrichtung nicht.", core_id=core_id)
                )
            for goal in matching_goals:
                required_component_ids.extend(goal_components[goal])

        for required_component_id in (
            required_component_ids
        ):
            if (
                required_component_id
                not in selected_component_ids
            ):
                errors.append(
                    t("Das Grundmodell '{core_id}' benötigt den Zielfunktionsbaustein '{required_component_id}'.", core_id=core_id, required_component_id=required_component_id)
                )


    constraints = proposal.get("constraints")

    if not isinstance(constraints, list):
        errors.append(
            t("Die Nebenbedingungen müssen als Liste vorliegen.")
        )
    else:
        for index, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                errors.append(
                    t("Die Nebenbedingung an Position {index} muss als Objekt vorliegen.", index=index)
                )
                continue

            component_id = constraint.get("component_id")

            if not _has_text(component_id):
                errors.append(
                    t("Die Nebenbedingung an Position {index} enthält keine gültige ID.", index=index)
                )
            elif component_id not in CONSTRAINT_COMPONENTS:
                errors.append(
                    t("Die Nebenbedingung '{component_id}' ist nicht registriert.", component_id=component_id)
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
                    t("Die Nebenbedingung '{component_id}' ist nicht mit '{core_id}' kompatibel.", component_id=component_id, core_id=core_id)
                )

            if not isinstance(
                constraint.get("parameters"),
                dict,
            ):
                errors.append(
                    t("Die Parameter der Nebenbedingung an Position {index} müssen als Objekt vorliegen.", index=index)
                )

            product_id = constraint.get("product_id")

            if (
                product_id is not None
                and not _has_text(product_id)
            ):
                errors.append(
                    t("Die Produkt-ID der Nebenbedingung an Position {index} ist ungültig.", index=index)
                )

            if not _has_text(constraint.get("reason")):
                errors.append(
                    t("Für die Nebenbedingung an Position {index} fehlt eine Begründung.", index=index)
                )

            if not _has_valid_sources(
                constraint.get("sources")
            ):
                errors.append(
                    t("Für die Nebenbedingung an Position {index} fehlt eine gültige Quelle.", index=index)
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
                        t("Der Kandidaten-Baustein '{candidate_component_id}' der fachlichen Rückfrage an Position {index} ist nicht registriert.", candidate_component_id=candidate_component_id, index=index)
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
                        t("Der Kandidaten-Baustein '{candidate_component_id}' der fachlichen Rückfrage an Position {index} ist nicht mit '{core_id}' kompatibel.", candidate_component_id=candidate_component_id, index=index, core_id=core_id)
                    )

                if (
                    candidate_component_id
                    in selected_component_ids
                    and not allow_selected_candidate_overlap
                ):
                    errors.append(
                        t("Der Kandidaten-Baustein '{candidate_component_id}' der fachlichen Rückfrage an Position {index} wurde bereits ausgewählt.", candidate_component_id=candidate_component_id, index=index)
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
            t("Für den Ausschluss wird eine gültige Bestätigungsquelle benötigt.")
        )

    proposal_errors = (
        validate_component_selection_proposal(proposal)
    )

    if proposal_errors:
        raise ValueError(
            t("Ungültiger Bausteinvorschlag: ")
            + "; ".join(proposal_errors)
        )

    if proposal["selection_status"] != "unsupported":
        raise ValueError(
            t("Nur ein nicht unterstützter Bausteinvorschlag kann auf diese Weise fortgesetzt werden.")
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
            t("Der Vorschlag enthält keine nicht unterstützten Anforderungen.")
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
            t("Der bereinigte Bausteinvorschlag ist ungültig: ")
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
            t("Ungültiger Bausteinvorschlag: ")
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
            t("Aus einem nicht vollständigen Bausteinvorschlag darf keine Modellspezifikation erzeugt werden. Ungelöste Anforderungen: {unresolved_descriptions}.", unresolved_descriptions=unresolved_descriptions)
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
