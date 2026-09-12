"""
Erzeugt einen serialisierbaren Bausteinkatalog für Agenten.
"""

from copy import deepcopy
from typing import TypedDict

from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
    OBJECTIVE_COMPONENTS,
    ComponentScope,
    ObjectiveEffect,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)
from agentic_optimization.models.types import (
    ConstraintParameterDefinition,
    ModelInputField,
)


class CoreCatalogEntry(TypedDict):
    """Für Agenten relevante Beschreibung eines Grundmodells."""

    core_id: str
    name: str
    minimum_products: int
    required_forecast_fields: list[ModelInputField]
    required_objective_component_ids: list[str]
    required_product_facts: list[ModelInputField]
    required_global_facts: list[ModelInputField]


class ObjectiveComponentCatalogEntry(TypedDict):
    """Serialisierbare Beschreibung eines Zielfunktionsbausteins."""

    component_id: str
    name: str
    effect: ObjectiveEffect
    required_product_facts: list[ModelInputField]
    required_global_facts: list[ModelInputField]


class ConstraintComponentCatalogEntry(TypedDict):
    """Serialisierbare Beschreibung eines Constraints."""

    component_id: str
    name: str
    scope: ComponentScope
    required_product_facts: list[ModelInputField]
    required_global_facts: list[ModelInputField]
    parameters: list[ConstraintParameterDefinition]


class ComponentCatalog(TypedDict):
    """Erlaubte Bausteine eines ausgewählten Grundmodells."""

    core: CoreCatalogEntry
    objective_components: list[ObjectiveComponentCatalogEntry]
    constraint_components: list[ConstraintComponentCatalogEntry]


def build_component_catalog(
    core_id: str,
) -> ComponentCatalog:
    """Liefert ausschließlich zum Grundmodell passende Bausteine."""

    registered_core = CORE_REGISTRY.get(core_id)

    if registered_core is None:
        raise ValueError(
            f"Das Grundmodell '{core_id}' ist nicht registriert."
        )

    core_definition = registered_core["definition"]

    objective_components: list[
        ObjectiveComponentCatalogEntry
    ] = []

    for definition in OBJECTIVE_COMPONENTS.values():
        if core_id not in definition["compatible_core_ids"]:
            continue

        objective_components.append(
            {
                "component_id": definition["component_id"],
                "name": definition["name"],
                "effect": definition["effect"],
                "required_product_facts": deepcopy(
                    definition["required_product_facts"]
                ),
                "required_global_facts": deepcopy(
                    definition["required_global_facts"]
                ),
            }
        )

    constraint_components: list[
        ConstraintComponentCatalogEntry
    ] = []

    for definition in CONSTRAINT_COMPONENTS.values():
        if core_id not in definition["compatible_core_ids"]:
            continue

        constraint_components.append(
            {
                "component_id": definition["component_id"],
                "name": definition["name"],
                "scope": definition["scope"],
                "required_product_facts": deepcopy(
                    definition["required_product_facts"]
                ),
                "required_global_facts": deepcopy(
                    definition["required_global_facts"]
                ),
                "parameters": deepcopy(
                    definition["parameters"]
                ),
            }
        )

    return {
        "core": {
            "core_id": core_definition["core_id"],
            "name": core_definition["name"],
            "minimum_products": core_definition[
                "minimum_products"
            ],
            "required_forecast_fields": deepcopy(
                core_definition["required_forecast_fields"]
            ),
            "required_objective_component_ids": list(
                core_definition[
                    "required_objective_component_ids"
                ]
            ),
            "required_product_facts": deepcopy(
                core_definition["required_product_facts"]
            ),
            "required_global_facts": deepcopy(
                core_definition["required_global_facts"]
            ),
        },
        "objective_components": objective_components,
        "constraint_components": constraint_components,
    }