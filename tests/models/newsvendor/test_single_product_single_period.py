import pytest

from agentic_optimization.models.newsvendor.single_product_single_period import (
    optimize_single_product_single_period,
)


@pytest.fixture
def valid_inputs() -> dict:
    return {
        "selling_price": 10.0,
        "purchase_cost": 5.0,
        "salvage_value": 2.0,
        "demand_scenarios": [
            80.0,
            100.0,
            120.0,
        ],
        "scenario_probabilities": [
            0.25,
            0.5,
            0.25,
        ],
    }


def test_optimizes_without_optional_constraints(
    valid_inputs: dict,
) -> None:
    result = optimize_single_product_single_period(
        inputs=valid_inputs,
        constraints=[],
    )

    assert result["solver_status"] == "Optimal"

    assert result["order_quantity"] == pytest.approx(
        100.0
    )

    assert result["expected_profit"] == pytest.approx(
        460.0
    )

    assert len(result["scenario_results"]) == 3


def test_applies_maximum_order_quantity(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "maximum_order_quantity",
            "parameters": {
                "quantity": 90.0,
            },
            "source_text": (
                "Es dürfen höchstens 90 Einheiten "
                "bestellt werden."
            ),
        }
    ]

    result = optimize_single_product_single_period(
        inputs=valid_inputs,
        constraints=constraints,
    )

    assert result["solver_status"] == "Optimal"

    assert result["order_quantity"] == pytest.approx(
        90.0
    )

    assert result["expected_profit"] == pytest.approx(
        430.0
    )


def test_applies_minimum_order_quantity(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {
                "quantity": 110.0,
            },
            "source_text": (
                "Es müssen mindestens 110 Einheiten "
                "bestellt werden."
            ),
        }
    ]

    result = optimize_single_product_single_period(
        inputs=valid_inputs,
        constraints=constraints,
    )

    assert result["solver_status"] == "Optimal"

    assert result["order_quantity"] == pytest.approx(
        110.0
    )


def test_applies_budget_limit(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "budget_limit",
            "parameters": {
                "budget": 450.0,
            },
            "source_text": (
                "Es stehen höchstens 450 Euro zur Verfügung."
            ),
        }
    ]

    result = optimize_single_product_single_period(
        inputs=valid_inputs,
        constraints=constraints,
    )

    assert result["solver_status"] == "Optimal"

    # Einkaufskosten 5 × Bestellmenge <= Budget 450
    assert result["order_quantity"] == pytest.approx(
        90.0
    )


def test_applies_storage_capacity(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "storage_capacity",
            "parameters": {
                "capacity": 85.0,
            },
            "source_text": (
                "Es können höchstens 85 Einheiten "
                "gelagert werden."
            ),
        }
    ]

    result = optimize_single_product_single_period(
        inputs=valid_inputs,
        constraints=constraints,
    )

    assert result["solver_status"] == "Optimal"

    assert result["order_quantity"] == pytest.approx(
        85.0
    )


def test_returns_infeasible_for_conflicting_constraints(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {
                "quantity": 100.0,
            },
            "source_text": (
                "Es müssen mindestens 100 Einheiten "
                "bestellt werden."
            ),
        },
        {
            "type": "budget_limit",
            "parameters": {
                "budget": 400.0,
            },
            "source_text": (
                "Es stehen höchstens 400 Euro zur Verfügung."
            ),
        },
    ]

    result = optimize_single_product_single_period(
        inputs=valid_inputs,
        constraints=constraints,
    )

    # Das Budget erlaubt bei Einkaufskosten von 5 nur
    # 80 Einheiten, die Mindestmenge verlangt aber 100.
    assert result["solver_status"] == "Infeasible"
    assert result["order_quantity"] is None
    assert result["expected_profit"] is None
    assert result["scenario_results"] == ()


def test_model_representation_contains_core_model(
    valid_inputs: dict,
) -> None:
    result = optimize_single_product_single_period(
        inputs=valid_inputs,
        constraints=[],
    )

    model_text = result["model_representation"]

    assert (
        "Newsvendor_Single_Product_Single_Period"
        in model_text
    )
    assert "MAXIMIZE" in model_text
    assert "SUBJECT TO" in model_text

    assert "order_quantity" in model_text
    assert "sold_0" in model_text
    assert "leftover_0" in model_text

    assert "demand_limit_0" in model_text
    assert "inventory_balance_0" in model_text

    assert "minimum_order_quantity" not in model_text
    assert "maximum_order_quantity" not in model_text
    assert "budget_limit" not in model_text
    assert "storage_capacity" not in model_text


def test_model_representation_contains_active_constraints(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {
                "quantity": 40.0,
            },
            "source_text": None,
        },
        {
            "type": "budget_limit",
            "parameters": {
                "budget": 600.0,
            },
            "source_text": None,
        },
    ]

    result = optimize_single_product_single_period(
        inputs=valid_inputs,
        constraints=constraints,
    )

    model_text = result["model_representation"]

    assert "minimum_order_quantity" in model_text
    assert "budget_limit" in model_text
    assert "maximum_order_quantity" not in model_text
    assert "storage_capacity" not in model_text