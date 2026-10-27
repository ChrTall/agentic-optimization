import pytest

from agentic_optimization.models.eoq.core import (
    CORE_DEFINITION,
    annualize_expected_demand,
    calculate_eoq_product,
    validate_eoq_forecast,
    validate_eoq_semantics,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)


def create_specification() -> ModelSpecification:
    return {
        "core_id": "eoq",
        "products": [
            {
                "product_id": "product_1",
                "forecast": {
                    "expected_value": 1000.0,
                    "horizon_days": 100,
                },
                "facts": {},
            }
        ],
        "global_facts": {},
        "objective": {
            "sense": "minimize",
            "component_ids": [
                "annual_ordering_cost",
                "annual_holding_cost",
            ],
        },
        "constraints": [],
    }


def test_defines_eoq_core_quantities() -> None:
    assert set(CORE_DEFINITION["provided_quantities"]) == {
        "order_quantity",
        "orders_per_year",
        "average_inventory",
        "cycle_length_days",
    }


def test_annualizes_expected_demand() -> None:
    annual_demand = annualize_expected_demand(
        {
            "expected_value": 1000.0,
            "horizon_days": 100,
        }
    )

    assert annual_demand == 3650.0


def test_accepts_forecast_without_quantiles() -> None:
    errors = validate_eoq_forecast(
        {
            "expected_value": 1000.0,
            "horizon_days": 100,
        }
    )

    assert errors == []


def test_requires_cost_minimization() -> None:
    specification = create_specification()
    specification["objective"]["sense"] = "maximize"

    errors = validate_eoq_semantics(specification)

    assert any(
        "Kosten minimieren" in error
        for error in errors
    )


def test_declares_classical_eoq_cost_components() -> None:
    assert CORE_DEFINITION[
        "required_objective_component_ids"
    ] == [
        "annual_ordering_cost",
        "annual_holding_cost",
    ]


def test_calculates_classical_eoq() -> None:
    result = calculate_eoq_product(
        annual_demand=3500.0,
        ordering_cost=120.0,
        holding_cost=12.0,
    )

    assert result.unconstrained_order_quantity == pytest.approx(
        264.5751311064591
    )
    # Ganzzahlige Empfehlung: 265 ist günstiger als 264.
    assert result.order_quantity == 265.0
    assert result.annual_ordering_cost == pytest.approx(
        result.annual_holding_cost, rel=1e-2
    )
    assert result.annual_relevant_cost == pytest.approx(
        3174.9056603773583
    )


def test_applies_order_quantity_bounds() -> None:
    result = calculate_eoq_product(
        annual_demand=3500.0,
        ordering_cost=120.0,
        holding_cost=12.0,
        minimum_order_quantity=300.0,
        maximum_order_quantity=500.0,
    )

    assert result.unconstrained_order_quantity < 300.0
    assert result.order_quantity == 300.0
    assert result.annual_relevant_cost == pytest.approx(
        3200.0
    )


def test_selects_cost_minimal_order_multiple() -> None:
    result = calculate_eoq_product(
        annual_demand=3500.0,
        ordering_cost=120.0,
        holding_cost=12.0,
        order_multiple=100,
    )

    assert result.unconstrained_order_quantity == (
        pytest.approx(264.5751311064591)
    )
    assert result.order_quantity == 300.0
    assert result.annual_relevant_cost == pytest.approx(
        3200.0
    )
