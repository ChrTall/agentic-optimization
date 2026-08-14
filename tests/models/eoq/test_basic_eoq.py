import math

import pytest

from agentic_optimization.models.eoq.basic_eoq import (
    INPUT_FIELDS,
    SUPPORTED_CONSTRAINTS,
    VARIANT_DEFINITION,
    BasicEoqSpecification,
    build_basic_eoq_result_presentation,
    create_basic_eoq_specification,
    optimize_basic_eoq,
    solve_basic_eoq,
)


@pytest.fixture
def valid_inputs() -> dict:
    return {
        "annual_demand": 3500.0,
        "ordering_cost": 120.0,
        "holding_cost": 12.0,
    }


def test_creates_normalized_specification(
    valid_inputs: dict,
) -> None:
    specification = create_basic_eoq_specification(
        valid_inputs
    )

    assert specification == BasicEoqSpecification(
        annual_demand=3500.0,
        ordering_cost=120.0,
        holding_cost=12.0,
    )


def test_calculates_optimal_order_quantity() -> None:
    specification = BasicEoqSpecification(
        annual_demand=3500.0,
        ordering_cost=120.0,
        holding_cost=12.0,
    )

    result = solve_basic_eoq(specification)

    expected_quantity = math.sqrt(
        2 * 3500 * 120 / 12
    )

    assert result.solver_status == "Optimal"

    assert result.order_quantity == pytest.approx(
        expected_quantity
    )


def test_calculates_derived_results() -> None:
    specification = BasicEoqSpecification(
        annual_demand=3500.0,
        ordering_cost=120.0,
        holding_cost=12.0,
    )

    result = solve_basic_eoq(specification)

    assert result.orders_per_year == pytest.approx(
        3500 / result.order_quantity
    )

    assert result.average_inventory == pytest.approx(
        result.order_quantity / 2
    )

    assert result.cycle_length_years == pytest.approx(
        result.order_quantity / 3500
    )

    assert result.cycle_length_days == pytest.approx(
        result.cycle_length_years * 365
    )


def test_ordering_and_holding_costs_are_equal_at_optimum() -> None:
    specification = BasicEoqSpecification(
        annual_demand=3500.0,
        ordering_cost=120.0,
        holding_cost=12.0,
    )

    result = solve_basic_eoq(specification)

    assert result.annual_ordering_cost == pytest.approx(
        result.annual_holding_cost
    )

    assert result.annual_relevant_cost == pytest.approx(
        result.annual_ordering_cost
        + result.annual_holding_cost
    )


def test_optimizer_returns_dictionary(
    valid_inputs: dict,
) -> None:
    result = optimize_basic_eoq(
        inputs=valid_inputs,
        constraints=[],
    )

    assert result["solver_status"] == "Optimal"

    assert result["order_quantity"] == pytest.approx(
        math.sqrt(70000)
    )

    assert "Q* = sqrt(2 * D * S / H)" in (
        result["model_representation"]
    )

    assert (
        "Mindestbestellmenge = nicht gesetzt"
        in result["model_representation"]
    )

    assert (
        "Höchstbestellmenge = nicht gesetzt"
        in result["model_representation"]
    )

    assert "None" not in result["model_representation"]


def test_minimum_order_quantity_limits_result(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {
                "quantity": 300.0,
            },
            "source_text": None,
        }
    ]

    result = optimize_basic_eoq(
        inputs=valid_inputs,
        constraints=constraints,
    )

    assert result[
        "unconstrained_order_quantity"
    ] == pytest.approx(
        math.sqrt(70000)
    )

    assert result["order_quantity"] == 300.0


def test_maximum_order_quantity_limits_result(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "maximum_order_quantity",
            "parameters": {
                "quantity": 200.0,
            },
            "source_text": None,
        }
    ]

    result = optimize_basic_eoq(
        inputs=valid_inputs,
        constraints=constraints,
    )

    assert result[
        "unconstrained_order_quantity"
    ] == pytest.approx(
        math.sqrt(70000)
    )

    assert result["order_quantity"] == 200.0


def test_non_binding_constraints_keep_eoq(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {
                "quantity": 100.0,
            },
            "source_text": None,
        },
        {
            "type": "maximum_order_quantity",
            "parameters": {
                "quantity": 400.0,
            },
            "source_text": None,
        },
    ]

    result = optimize_basic_eoq(
        inputs=valid_inputs,
        constraints=constraints,
    )

    assert result["order_quantity"] == pytest.approx(
        math.sqrt(70000)
    )


def test_optimizer_rejects_invalid_inputs(
    valid_inputs: dict,
) -> None:
    inputs = {
        **valid_inputs,
        "holding_cost": 0.0,
    }

    with pytest.raises(
        ValueError,
        match="Lagerhaltungskosten",
    ):
        optimize_basic_eoq(
            inputs=inputs,
            constraints=[],
        )


def test_builds_result_metrics(
    valid_inputs: dict,
) -> None:
    optimization_result = optimize_basic_eoq(
        inputs=valid_inputs,
        constraints=[],
    )

    presentation = (
        build_basic_eoq_result_presentation(
            optimization_result
        )
    )

    assert presentation["metrics"] == [
        {
            "label": "Empfohlene Bestellmenge",
            "value": "264,58 Einheiten",
        },
        {
            "label": (
                "Jährliche relevante Gesamtkosten"
            ),
            "value": "3.174,90 €",
        },
    ]


def test_builds_result_details(
    valid_inputs: dict,
) -> None:
    optimization_result = optimize_basic_eoq(
        inputs=valid_inputs,
        constraints=[],
    )

    presentation = (
        build_basic_eoq_result_presentation(
            optimization_result
        )
    )

    assert (
        presentation["detail_title"]
        == "EOQ-Berechnungsdetails"
    )

    assert presentation["detail_rows"] == [
        {
            "Kennzahl": "Bestellungen pro Jahr",
            "Wert": "13,23",
        },
        {
            "Kennzahl": (
                "Durchschnittlicher Lagerbestand"
            ),
            "Wert": "132,29 Einheiten",
        },
        {
            "Kennzahl": (
                "Durchschnittliche Zyklusdauer"
            ),
            "Wert": "27,59 Tage",
        },
        {
            "Kennzahl": "Jährliche Bestellkosten",
            "Wert": "1.587,45 €",
        },
        {
            "Kennzahl": (
                "Jährliche Lagerhaltungskosten"
            ),
            "Wert": "1.587,45 €",
        },
    ]


def test_variant_definition_uses_analytical_eoq_components() -> None:
    assert (
        VARIANT_DEFINITION["variant_id"]
        == "basic_eoq"
    )

    assert VARIANT_DEFINITION[
        "selection_criteria"
    ] == {
        "decision": "order_quantity",
        "objective": "minimize_cost",
        "demand_type": "known",
        "product_scope": "single_product",
        "planning_horizon": "repeated_cycles",
    }

    assert (
        VARIANT_DEFINITION["input_fields"]
        == INPUT_FIELDS
    )

    assert (
        VARIANT_DEFINITION[
            "supported_constraints"
        ]
        == SUPPORTED_CONSTRAINTS
    )

    assert (
        VARIANT_DEFINITION["optimizer"]
        is optimize_basic_eoq
    )

    assert (
        VARIANT_DEFINITION["result_builder"]
        is build_basic_eoq_result_presentation
    )


def test_optimizer_rejects_missing_constraint_quantity(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {},
            "source_text": None,
        }
    ]

    with pytest.raises(
        ValueError,
        match="fehlt der Parameter 'quantity'",
    ):
        optimize_basic_eoq(
            inputs=valid_inputs,
            constraints=constraints,
        )
