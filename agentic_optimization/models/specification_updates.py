"""
Aktualisierung modularer Modellspezifikationen.
"""

from copy import deepcopy
from typing import Any

from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.validator import (
    MissingInput,
)
from agentic_optimization.i18n import t


def apply_modular_input_value(
    specification: ModelSpecification,
    missing_input: MissingInput,
    value: Any,
) -> ModelSpecification:
    """Übernimmt einen ergänzten Wert in eine Kopie der Spezifikation."""

    updated_specification = deepcopy(specification)

    source = missing_input["source"]
    field_name = missing_input["field"]["name"]

    if source in {"forecast", "product_fact"}:
        product_id = missing_input.get("product_id")

        if not product_id:
            raise ValueError(
                t("Für die Produkteingabe fehlt die Produkt-ID.")
            )

        product = next(
            (
                product
                for product
                in updated_specification["products"]
                if product["product_id"] == product_id
            ),
            None,
        )

        if product is None:
            raise ValueError(
                t("Das Produkt '{product_id}' ist nicht vorhanden.", product_id=product_id)
            )

        if source == "forecast":
            product.setdefault(
                "forecast",
                {},
            )[field_name] = value
        else:
            product.setdefault(
                "facts",
                {},
            )[field_name] = value

        return updated_specification

    if source == "global_fact":
        updated_specification["global_facts"][
            field_name
        ] = value

        return updated_specification

    if source == "constraint_parameter":
        constraint_index = missing_input.get(
            "constraint_index"
        )

        if constraint_index is None:
            raise ValueError(
                t("Für den Constraint-Parameter fehlt der Index.")
            )

        try:
            constraint = updated_specification[
                "constraints"
            ][constraint_index]
        except IndexError as error:
            raise ValueError(
                t("Der zugehörige Constraint ist nicht vorhanden.")
            ) from error

        constraint["parameters"][field_name] = value

        return updated_specification

    raise ValueError(
        t("Die Eingabequelle '{source}' wird nicht unterstützt.", source=source)
    )

# ---------------------------------------------------------------------------
# Bausteine und Produktnamen in Formular 3 bearbeiten
# ---------------------------------------------------------------------------

USER_FORM_SOURCE = "modular_input_form"
"""Quellenkennung für Änderungen, die der Nutzer in Formular 3 vornimmt."""

MAX_PRODUCT_NAME_CHARS = 40


def _goal_for_sense(sense: str) -> str:
    return "maximize_profit" if sense == "maximize" else "minimize_cost"


def _locked_objective_ids(specification: ModelSpecification) -> set[str]:
    """Zielfunktionsbausteine, die das Grundmodell zwingend braucht."""

    from agentic_optimization.models.core_registry import CORE_REGISTRY

    registered = CORE_REGISTRY.get(specification["core_id"])

    if registered is None:
        return set()

    definition = registered["definition"]
    goal = _goal_for_sense(specification["objective"]["sense"])

    return {
        *definition.get("required_objective_component_ids", []),
        *definition.get("required_objective_components_by_goal", {}).get(
            goal, []
        ),
    }


def list_optional_components(
    specification: ModelSpecification,
) -> list[dict[str, Any]]:
    """Alle mit dem Grundmodell kompatiblen Bausteine für Formular 3.

    ``locked`` kennzeichnet Bausteine, die das Modell zwingend braucht und
    die deshalb nicht abgewählt werden können.
    """

    from agentic_optimization.models.component_registry import (
        CONSTRAINT_COMPONENTS,
        OBJECTIVE_COMPONENTS,
    )

    core_id = specification["core_id"]
    active_objective = set(specification["objective"]["component_ids"])
    active_constraints = {
        constraint["component_id"]
        for constraint in specification["constraints"]
    }
    locked = _locked_objective_ids(specification)
    components: list[dict[str, Any]] = []

    for component_id, component in OBJECTIVE_COMPONENTS.items():
        if core_id not in component["compatible_core_ids"]:
            continue

        components.append({
            "component_id": component_id,
            "kind": "objective",
            "name": t(component["name"]),
            "active": component_id in active_objective,
            "locked": component_id in locked,
        })

    for component_id, component in CONSTRAINT_COMPONENTS.items():
        if core_id not in component["compatible_core_ids"]:
            continue

        components.append({
            "component_id": component_id,
            "kind": "constraint",
            "name": t(component["name"]),
            "active": component_id in active_constraints,
            "locked": False,
        })

    return components


def update_specification_components(
    specification: ModelSpecification,
    proposal: dict[str, Any],
    *,
    active_component_ids: set[str],
) -> tuple[ModelSpecification, dict[str, Any]]:
    """Wählt optionale Bausteine an oder ab.

    Neue Nebenbedingungen erhalten leere Parameter; die fehlenden Werte
    erscheinen anschließend als offene Eingaben in Formular 3.
    """

    from agentic_optimization.models.component_registry import (
        CONSTRAINT_COMPONENTS,
        OBJECTIVE_COMPONENTS,
    )

    updated_specification = deepcopy(specification)
    updated_proposal = deepcopy(proposal)
    core_id = specification["core_id"]
    active = set(active_component_ids) | _locked_objective_ids(specification)

    # Zielfunktion
    current_objective = updated_specification["objective"]["component_ids"]
    objective_ids = [cid for cid in current_objective if cid in active]
    objective_ids += [
        cid
        for cid, component in OBJECTIVE_COMPONENTS.items()
        if cid in active
        and cid not in objective_ids
        and core_id in component["compatible_core_ids"]
    ]
    updated_specification["objective"]["component_ids"] = objective_ids

    proposed_objective = [
        component
        for component in updated_proposal.get("objective_components", [])
        if component["component_id"] in objective_ids
    ]
    proposed_ids = {c["component_id"] for c in proposed_objective}
    proposed_objective += [
        {
            "component_id": cid,
            "reason": t("Im Formular ergänzt."),
            "sources": [USER_FORM_SOURCE],
        }
        for cid in objective_ids
        if cid not in proposed_ids
    ]
    updated_proposal["objective_components"] = proposed_objective

    # Nebenbedingungen
    kept_constraints = [
        constraint
        for constraint in updated_specification["constraints"]
        if constraint["component_id"] in active
    ]
    present = {constraint["component_id"] for constraint in kept_constraints}
    product_ids = [p["product_id"] for p in updated_specification["products"]]
    new_constraints: list[dict[str, Any]] = []

    for cid, component in CONSTRAINT_COMPONENTS.items():
        if (
            cid not in active
            or cid in present
            or core_id not in component["compatible_core_ids"]
        ):
            continue

        if component["scope"] == "per_product":
            new_constraints += [
                {"component_id": cid, "product_id": product_id, "parameters": {}}
                for product_id in product_ids
            ]
        else:
            new_constraints.append({"component_id": cid, "parameters": {}})

    updated_specification["constraints"] = kept_constraints + new_constraints
    updated_proposal["constraints"] = [
        constraint
        for constraint in updated_proposal.get("constraints", [])
        if constraint["component_id"] in active
    ] + [
        {
            **deepcopy(constraint),
            "reason": t("Im Formular ergänzt."),
            "sources": [USER_FORM_SOURCE],
        }
        for constraint in new_constraints
    ]

    return updated_specification, updated_proposal


def validate_product_name(
    new_name: str,
    *,
    old_name: str,
    existing_names: list[str],
) -> str:
    """Prüft und bereinigt einen neuen Produktnamen."""

    cleaned = " ".join(str(new_name).split())

    if not cleaned:
        raise ValueError(t("Der Produktname darf nicht leer sein."))

    if len(cleaned) > MAX_PRODUCT_NAME_CHARS:
        raise ValueError(
            t("Der Produktname darf höchstens {MAX_PRODUCT_NAME_CHARS} Zeichen lang sein.", MAX_PRODUCT_NAME_CHARS=MAX_PRODUCT_NAME_CHARS)
        )

    if any(
        name.casefold() == cleaned.casefold() and name != old_name
        for name in existing_names
    ):
        raise ValueError(t("Den Produktnamen „{cleaned}“ gibt es bereits.", cleaned=cleaned))

    return cleaned


def _rename_in_items(items: list[dict[str, Any]], old: str, new: str) -> None:
    for item in items:
        if item.get("product_id") == old:
            item["product_id"] = new


def rename_product(
    modular_state: dict[str, Any],
    old_name: str,
    new_name: str,
) -> dict[str, Any]:
    """Benennt ein Produkt überall im modularen Zustand um."""

    specification = modular_state.get("specification") or {}
    existing = [p["product_id"] for p in specification.get("products", [])]

    if old_name not in existing:
        raise ValueError(t("Das Produkt „{old_name}“ ist nicht vorhanden.", old_name=old_name))

    cleaned = validate_product_name(
        new_name, old_name=old_name, existing_names=existing
    )

    if cleaned == old_name:
        return modular_state

    updated = deepcopy(modular_state)

    for key in ("products",):
        _rename_in_items(updated.get(key) or [], old_name, cleaned)

    if updated.get("specification"):
        _rename_in_items(updated["specification"]["products"], old_name, cleaned)
        _rename_in_items(updated["specification"]["constraints"], old_name, cleaned)

    for key in ("proposal", "preliminary_proposal"):
        if updated.get(key):
            _rename_in_items(updated[key].get("constraints", []), old_name, cleaned)

    return updated
