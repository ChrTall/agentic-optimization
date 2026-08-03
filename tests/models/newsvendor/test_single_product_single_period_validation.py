import math

import pytest

from agentic_optimization.models.newsvendor.single_product_single_period import (
    validate_single_product_single_period_constraints,
    validate_single_product_single_period_inputs,
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


def test_accepts_valid_inputs(
    valid_inputs: dict,
) -> None:
    errors = validate_single_product_single_period_inputs(
        valid_inputs
    )

    assert errors == []


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("selling_price", 0.0),
        ("selling_price", -1.0),
        ("purchase_cost", 0.0),
        ("purchase_cost", -1.0),
    ],
)
def test_rejects_non_positive_prices_and_costs(
    valid_inputs: dict,
    field_name: str,
    value: float,
) -> None:
    inputs = {
        **valid_inputs,
        field_name: value,
    }

    errors = validate_single_product_single_period_inputs(
        inputs
    )

    assert errors


def test_rejects_selling_price_not_above_purchase_cost(
    valid_inputs: dict,
) -> None:
    inputs = {
        **valid_inputs,
        "selling_price": 5.0,
        "purchase_cost": 5.0,
    }

    errors = validate_single_product_single_period_inputs(
        inputs
    )

    assert any(
        "größer als die Einkaufskosten" in error
        for error in errors
    )


def test_rejects_negative_salvage_value(
    valid_inputs: dict,
) -> None:
    inputs = {
        **valid_inputs,
        "salvage_value": -1.0,
    }

    errors = validate_single_product_single_period_inputs(
        inputs
    )

    assert any(
        "Restwert darf nicht negativ" in error
        for error in errors
    )


@pytest.mark.parametrize(
    "salvage_value",
    [
        5.0,
        6.0,
    ],
)
def test_rejects_salvage_value_not_below_purchase_cost(
    valid_inputs: dict,
    salvage_value: float,
) -> None:
    inputs = {
        **valid_inputs,
        "salvage_value": salvage_value,
    }

    errors = validate_single_product_single_period_inputs(
        inputs
    )

    assert any(
        (
            "Restwert muss kleiner als die "
            "Einkaufskosten sein."
        )
        in error
        for error in errors
    )


def test_rejects_empty_demand_scenarios(
    valid_inputs: dict,
) -> None:
    inputs = {
        **valid_inputs,
        "demand_scenarios": [],
    }

    errors = validate_single_product_single_period_inputs(
        inputs
    )

    assert any(
        "mindestens ein Nachfrageszenario" in error
        for error in errors
    )


def test_rejects_empty_probabilities(
    valid_inputs: dict,
) -> None:
    inputs = {
        **valid_inputs,
        "scenario_probabilities": [],
    }

    errors = validate_single_product_single_period_inputs(
        inputs
    )

    assert any(
        "mindestens eine Wahrscheinlichkeit" in error
        for error in errors
    )


def test_rejects_different_scenario_list_lengths(
    valid_inputs: dict,
) -> None:
    inputs = {
        **valid_inputs,
        "scenario_probabilities": [
            0.5,
            0.5,
        ],
    }

    errors = validate_single_product_single_period_inputs(
        inputs
    )

    assert any(
        "Anzahl der Nachfrageszenarien" in error
        for error in errors
    )


def test_rejects_negative_demand(
    valid_inputs: dict,
) -> None:
    inputs = {
        **valid_inputs,
        "demand_scenarios": [
            -10.0,
            100.0,
            120.0,
        ],
    }

    errors = validate_single_product_single_period_inputs(
        inputs
    )

    assert any(
        "nicht negativ" in error
        for error in errors
    )


def test_rejects_probability_outside_valid_range(
    valid_inputs: dict,
) -> None:
    inputs = {
        **valid_inputs,
        "scenario_probabilities": [
            -0.1,
            0.6,
            0.5,
        ],
    }

    errors = validate_single_product_single_period_inputs(
        inputs
    )

    assert any(
        "zwischen 0 und 1" in error
        for error in errors
    )


def test_rejects_probability_sum_not_equal_to_one(
    valid_inputs: dict,
) -> None:
    inputs = {
        **valid_inputs,
        "scenario_probabilities": [
            0.2,
            0.2,
            0.2,
        ],
    }

    errors = validate_single_product_single_period_inputs(
        inputs
    )

    assert any(
        "zusammen 1" in error
        for error in errors
    )


@pytest.mark.parametrize(
    "invalid_value",
    [
        math.nan,
        math.inf,
        -math.inf,
        True,
    ],
)
def test_rejects_invalid_numeric_values(
    valid_inputs: dict,
    invalid_value: float,
) -> None:
    inputs = {
        **valid_inputs,
        "selling_price": invalid_value,
    }

    errors = validate_single_product_single_period_inputs(
        inputs
    )

    assert errors


def test_accepts_valid_optional_constraints(
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
            "type": "maximum_order_quantity",
            "parameters": {
                "quantity": 120.0,
            },
            "source_text": None,
        },
        {
            "type": "budget_limit",
            "parameters": {
                "budget": 1000.0,
            },
            "source_text": None,
        },
        {
            "type": "storage_capacity",
            "parameters": {
                "capacity": 150.0,
            },
            "source_text": None,
        },
    ]

    errors = (
        validate_single_product_single_period_constraints(
            valid_inputs,
            constraints,
        )
    )

    assert errors == []


def test_rejects_minimum_above_maximum(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {
                "quantity": 120.0,
            },
            "source_text": None,
        },
        {
            "type": "maximum_order_quantity",
            "parameters": {
                "quantity": 100.0,
            },
            "source_text": None,
        },
    ]

    errors = (
        validate_single_product_single_period_constraints(
            valid_inputs,
            constraints,
        )
    )

    assert any(
        "Mindestbestellmenge darf die Höchstbestellmenge"
        in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("constraint_type", "parameter_name", "value"),
    [
        (
            "minimum_order_quantity",
            "quantity",
            0.0,
        ),
        (
            "maximum_order_quantity",
            "quantity",
            -1.0,
        ),
        (
            "budget_limit",
            "budget",
            -1.0,
        ),
        (
            "storage_capacity",
            "capacity",
            -1.0,
        ),
    ],
)
def test_rejects_invalid_constraint_values(
    valid_inputs: dict,
    constraint_type: str,
    parameter_name: str,
    value: float,
) -> None:
    constraints = [
        {
            "type": constraint_type,
            "parameters": {
                parameter_name: value,
            },
            "source_text": None,
        }
    ]

    errors = (
        validate_single_product_single_period_constraints(
            valid_inputs,
            constraints,
        )
    )

    assert errors