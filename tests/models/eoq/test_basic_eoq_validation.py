import math

import pytest

from agentic_optimization.models.eoq.basic_eoq import (
    validate_basic_eoq_constraints,
    validate_basic_eoq_inputs,
)


@pytest.fixture
def valid_inputs() -> dict:
    return {
        "annual_demand": 3500.0,
        "ordering_cost": 120.0,
        "holding_cost": 12.0,
    }


def test_accepts_valid_inputs(
    valid_inputs: dict,
) -> None:
    errors = validate_basic_eoq_inputs(
        valid_inputs
    )

    assert errors == []


@pytest.mark.parametrize(
    "field_name",
    [
        "annual_demand",
        "ordering_cost",
        "holding_cost",
    ],
)
def test_rejects_missing_required_input(
    valid_inputs: dict,
    field_name: str,
) -> None:
    inputs = {
        **valid_inputs,
        field_name: None,
    }

    errors = validate_basic_eoq_inputs(inputs)

    assert errors


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("annual_demand", 0.0),
        ("annual_demand", -1.0),
        ("ordering_cost", 0.0),
        ("ordering_cost", -1.0),
        ("holding_cost", 0.0),
        ("holding_cost", -1.0),
    ],
)
def test_rejects_non_positive_input(
    valid_inputs: dict,
    field_name: str,
    value: float,
) -> None:
    inputs = {
        **valid_inputs,
        field_name: value,
    }

    errors = validate_basic_eoq_inputs(inputs)

    assert errors


@pytest.mark.parametrize(
    "invalid_value",
    [
        "100",
        True,
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_rejects_non_numeric_or_non_finite_input(
    valid_inputs: dict,
    invalid_value: object,
) -> None:
    inputs = {
        **valid_inputs,
        "annual_demand": invalid_value,
    }

    errors = validate_basic_eoq_inputs(inputs)

    assert any(
        "endliche Zahl" in error
        for error in errors
    )


def test_reports_all_missing_inputs() -> None:
    errors = validate_basic_eoq_inputs({})

    assert len(errors) == 3


def test_accepts_empty_constraint_list(
    valid_inputs: dict,
) -> None:
    errors = validate_basic_eoq_constraints(
        valid_inputs,
        [],
    )

    assert errors == []


def test_accepts_minimum_order_quantity(
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

    errors = validate_basic_eoq_constraints(
        valid_inputs,
        constraints,
    )

    assert errors == []


def test_accepts_maximum_order_quantity(
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

    errors = validate_basic_eoq_constraints(
        valid_inputs,
        constraints,
    )

    assert errors == []


def test_rejects_minimum_above_maximum(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {
                "quantity": 300.0,
            },
            "source_text": None,
        },
        {
            "type": "maximum_order_quantity",
            "parameters": {
                "quantity": 200.0,
            },
            "source_text": None,
        },
    ]

    errors = validate_basic_eoq_constraints(
        valid_inputs,
        constraints,
    )

    assert errors == [
        (
            "Die Mindestbestellmenge darf die "
            "Höchstbestellmenge nicht überschreiten."
        )
    ]


def test_rejects_non_positive_constraint_quantity(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "maximum_order_quantity",
            "parameters": {
                "quantity": 0.0,
            },
            "source_text": None,
        }
    ]

    errors = validate_basic_eoq_constraints(
        valid_inputs,
        constraints,
    )

    assert errors == [
        (
            "Die Bestellmengenbegrenzung muss größer "
            "als 0 sein."
        )
    ]


def test_rejects_unsupported_constraint(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "budget_limit",
            "parameters": {
                "budget": 5000.0,
            },
            "source_text": None,
        }
    ]

    errors = validate_basic_eoq_constraints(
        valid_inputs,
        constraints,
    )

    assert errors == [
        (
            "Der Constraint-Typ 'budget_limit' "
            "wird von Basic EOQ nicht unterstützt."
        )
    ]


def test_rejects_missing_constraint_quantity(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {},
            "source_text": None,
        }
    ]

    errors = validate_basic_eoq_constraints(
        valid_inputs,
        constraints,
    )

    assert errors == [
        (
            "Für den Constraint "
            "'minimum_order_quantity' "
            "fehlt der Parameter 'quantity'."
        )
    ]


def test_rejects_duplicate_constraint_type(
    valid_inputs: dict,
) -> None:
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {
                "quantity": 200.0,
            },
            "source_text": None,
        },
        {
            "type": "minimum_order_quantity",
            "parameters": {
                "quantity": 300.0,
            },
            "source_text": None,
        },
    ]

    errors = validate_basic_eoq_constraints(
        valid_inputs,
        constraints,
    )

    assert errors == [
        (
            "Der Constraint-Typ "
            "'minimum_order_quantity' "
            "wurde mehrfach angegeben."
        )
    ]
