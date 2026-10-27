import json

from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
)
from agentic_optimization.models.core_model_cards import (
    build_core_model_cards,
)
from agentic_optimization.models.core_applicability import CONDITION_LABELS
from agentic_optimization.models.core_registry import CORE_REGISTRY


def test_cards_are_derived_from_all_registered_cores() -> None:
    cards = build_core_model_cards()

    assert {card["core_id"] for card in cards} == set(CORE_REGISTRY)
    assert all(card["decision_description"] for card in cards)
    assert json.dumps(cards, ensure_ascii=False)


def test_eoq_card_exposes_conditions_and_required_objective() -> None:
    card = next(
        card for card in build_core_model_cards()
        if card["core_id"] == "eoq"
    )

    assert card["required_objective_sense"] == "minimize"
    assert {item["condition_id"] for item in card["conditions"]} == {
        "recurring_orders",
        "steady_demand", "no_quantity_discounts", "no_shortages"
    }
    steady_demand = next(
        item for item in card["conditions"]
        if item["condition_id"] == "steady_demand"
    )
    assert steady_demand["meaning"] == (
        CONDITION_LABELS["steady_demand"]
    )
    assert {item["component_id"] for item in card[
        "required_objective_components"
    ]} == {"annual_ordering_cost", "annual_holding_cost"}


def test_newsvendor_card_exposes_profit_goal_components() -> None:
    card = next(
        card for card in build_core_model_cards()
        if card["core_id"] == "single_period_newsvendor"
    )

    assert card["required_objective_components_by_goal"] == {
        "maximize_profit": ["sales_revenue", "purchase_cost"],
    }


def test_new_compatible_component_appears_without_editing_cards(
    monkeypatch,
) -> None:
    component = dict(CONSTRAINT_COMPONENTS["minimum_order_quantity"])
    component["component_id"] = "test_new_constraint"
    component["compatible_core_ids"] = frozenset({"eoq"})
    monkeypatch.setitem(
        CONSTRAINT_COMPONENTS, "test_new_constraint", component
    )

    card = next(
        card for card in build_core_model_cards()
        if card["core_id"] == "eoq"
    )

    assert "test_new_constraint" in {
        item["component_id"] for item in card["compatible_constraints"]
    }
