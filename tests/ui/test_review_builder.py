from agentic_optimization.models.newsvendor.single_product_single_period import (
    VARIANT_DEFINITION,
)
from agentic_optimization.ui.review_builder import (
    build_review_data,
)


def create_review_state() -> dict:
    return {
        "selected_model_family": "newsvendor",
        "selected_model_variant": (
            "single_product_single_period"
        ),
        "model_config": {
            "variant_name": (
                "Ein Produkt, eine Periode"
            ),
        },
        "requirements": {
            "decision": "order_quantity",
            "objective": "maximize_profit",
            "demand_type": "uncertain",
            "product_scope": "single_product",
            "planning_horizon": "single_period",
        },
        "model_specific_inputs": {
            "newsvendor": {
                "single_product_single_period": {
                    "selling_price": 20.0,
                    "purchase_cost": 12.0,
                    "salvage_value": 3.0,
                    "demand_scenarios": [
                        80.0,
                        100.0,
                        120.0,
                    ],
                    "scenario_probabilities": [
                        0.2,
                        0.5,
                        0.3,
                    ],
                },
            },
        },
        "structured_constraints": [
            {
                "type": "minimum_order_quantity",
                "parameters": {
                    "quantity": 30.0,
                },
                "source_text": (
                    "Mindestens 30 Stück bestellen."
                ),
            },
            {
                "type": "budget_limit",
                "parameters": {
                    "budget": 5000.0,
                },
                "source_text": (
                    "Maximales Budget von 5000 Euro."
                ),
            },
        ],
    }


def test_builds_problem_review_items() -> None:
    review_data = build_review_data(
        create_review_state(),
        VARIANT_DEFINITION,
    )

    assert review_data["problem"] == [
        {
            "label": "Entscheidung",
            "value": "Bestellmenge bestimmen",
        },
        {
            "label": "Ziel",
            "value": "Gewinn maximieren",
        },
        {
            "label": "Nachfrage",
            "value": "Unsicher",
        },
        {
            "label": "Produktumfang",
            "value": "Ein Produkt",
        },
        {
            "label": "Planungszeitraum",
            "value": "Einmalige Entscheidung",
        },
    ]


def test_builds_parameters_from_registry_metadata() -> None:
    review_data = build_review_data(
        create_review_state(),
        VARIANT_DEFINITION,
    )

    assert review_data["parameters"] == [
        {
            "label": "Verkaufspreis pro Einheit",
            "value": "20 € pro Einheit",
        },
        {
            "label": "Einkaufskosten pro Einheit",
            "value": "12 € pro Einheit",
        },
        {
            "label": (
                "Restwert pro unverkaufter Einheit"
            ),
            "value": "3 € pro Einheit",
        },
        {
            "label": "Mögliche Nachfragewerte",
            "value": "80, 100, 120 Einheiten",
        },
        {
            "label": (
                "Wahrscheinlichkeiten der Nachfragewerte"
            ),
            "value": "0.2, 0.5, 0.3",
        },
    ]


def test_builds_constraints_from_registry_metadata() -> None:
    review_data = build_review_data(
        create_review_state(),
        VARIANT_DEFINITION,
    )

    assert review_data["constraints"] == [
        {
            "label": "Mindestbestellmenge",
            "value": "30 Einheiten",
        },
        {
            "label": "Budgetgrenze",
            "value": "5000 €",
        },
    ]


def test_allows_review_without_constraints() -> None:
    state = create_review_state()
    state["structured_constraints"] = []

    review_data = build_review_data(
        state,
        VARIANT_DEFINITION,
    )

    assert review_data["constraints"] == []


def test_uses_registered_variant_name_as_fallback() -> None:
    state = create_review_state()
    state["model_config"] = {}

    review_data = build_review_data(
        state,
        VARIANT_DEFINITION,
    )

    assert (
        review_data["model_name"]
        == VARIANT_DEFINITION["variant_name"]
    )