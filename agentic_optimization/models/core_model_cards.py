"""Aktuelle Modellsteckbriefe aus den registrierten Definitionen."""

from typing import TypedDict

from agentic_optimization.models.component_catalog import (
    build_component_catalog,
)
from agentic_optimization.models.core_applicability import (
    CONDITION_LABELS,
    CORE_CONDITIONS,
)
from agentic_optimization.models.core_registry import CORE_REGISTRY


class ModelCardComponent(TypedDict):
    component_id: str
    name: str
    selection_guidance: str


class ModelCardCondition(TypedDict):
    condition_id: str
    meaning: str


class CoreModelCard(TypedDict):
    core_id: str
    name: str
    description: str
    decision_description: str
    assumptions: list[str]
    conditions: list[ModelCardCondition]
    minimum_products: int
    required_objective_sense: str | None
    required_forecast_fields: list[str]
    required_product_facts: list[str]
    required_global_facts: list[str]
    required_objective_components: list[ModelCardComponent]
    compatible_objective_components: list[ModelCardComponent]
    compatible_constraints: list[ModelCardComponent]


def _component_card(component: dict) -> ModelCardComponent:
    return {
        "component_id": component["component_id"],
        "name": component["name"],
        "selection_guidance": component["selection_guidance"],
    }


def build_core_model_cards() -> list[CoreModelCard]:
    """Erzeugt Steckbriefe ohne eine zweite, manuell gepflegte Modellliste."""

    cards: list[CoreModelCard] = []
    for core_id in CORE_REGISTRY:
        catalog = build_component_catalog(core_id)
        core = catalog["core"]
        objectives = [
            _component_card(component)
            for component in catalog["objective_components"]
        ]
        required_ids = set(core["required_objective_component_ids"])
        cards.append({
            "core_id": core_id,
            "name": core["name"],
            "description": core["description"],
            "decision_description": core["decision_description"],
            "assumptions": core["assumptions"],
            "conditions": [
                {
                    "condition_id": condition_id,
                    "meaning": CONDITION_LABELS[condition_id],
                }
                for condition_id in CORE_CONDITIONS[core_id]
            ],
            "minimum_products": core["minimum_products"],
            "required_objective_sense": core.get(
                "required_objective_sense"
            ),
            "required_forecast_fields": [
                field["name"]
                for field in core["required_forecast_fields"]
                if field["required"]
            ],
            "required_product_facts": [
                field["name"]
                for field in core["required_product_facts"]
                if field["required"]
            ],
            "required_global_facts": [
                field["name"]
                for field in core["required_global_facts"]
                if field["required"]
            ],
            "required_objective_components": [
                component for component in objectives
                if component["component_id"] in required_ids
            ],
            "compatible_objective_components": objectives,
            "compatible_constraints": [
                _component_card(component)
                for component in catalog["constraint_components"]
            ],
        })
    return cards
