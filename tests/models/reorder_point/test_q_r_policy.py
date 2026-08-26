import math

import pytest

from agentic_optimization.models.reorder_point.q_r_policy import (
    QRPolicySpecification,
    build_q_r_policy_result_presentation,
    create_q_r_policy_specification,
    optimize_q_r_policy,
    solve_q_r_policy,
    validate_q_r_policy_constraints,
    validate_q_r_policy_inputs,
)


@pytest.fixture
def valid_inputs() -> dict:
    return {
        "annual_demand": 10_000.0,
        "fixed_order_cost": 100.0,
        "holding_cost": 5.0,
        "penalty_cost": 25.0,
        "lead_time_demand_mean": 400.0,
        "lead_time_demand_std": 50.0,
    }


def test_valid_basic_input_converges(valid_inputs: dict) -> None:
    assert validate_q_r_policy_inputs(valid_inputs) == []

    result = optimize_q_r_policy(valid_inputs, [])

    assert result["solver_status"] == "Optimal"
    assert result["converged"] is True
    assert result["order_quantity"] > 0
    assert result["reorder_point"] >= 0
    assert result["expected_annual_cost"] is not None
    assert isinstance(result["order_quantity"], int)
    assert isinstance(result["reorder_point"], int)
    assert isinstance(result["safety_stock"], int)
    assert result["safety_stock"] == math.ceil(
        result["reorder_point"]
        - valid_inputs["lead_time_demand_mean"]
    )


def test_rounds_operational_outputs_with_fractional_statistics(
    valid_inputs: dict,
) -> None:
    inputs = {
        **valid_inputs,
        "lead_time_demand_mean": 400.25,
        "lead_time_demand_std": 49.75,
    }

    result = optimize_q_r_policy(inputs, [])

    assert result["solver_status"] == "Optimal"
    assert isinstance(result["order_quantity"], int)
    assert isinstance(result["reorder_point"], int)
    assert isinstance(result["safety_stock"], int)
    assert result["safety_stock"] == math.ceil(
        result["reorder_point"]
        - inputs["lead_time_demand_mean"]
    )


def test_matches_cornell_q_r_reference_example() -> None:
    inputs = {
        "annual_demand": 7_300.0,
        "fixed_order_cost": 250.0,
        "holding_cost": 50.0,
        "penalty_cost": 10.0,
        "lead_time_demand_mean": 60.0,
        "lead_time_demand_std": 6.928,
    }
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {"quantity": 250},
            "source_text": None,
        },
        {
            "type": "maximum_order_quantity",
            "parameters": {"quantity": 300},
            "source_text": None,
        },
        {
            "type": "minimum_reorder_point",
            "parameters": {"reorder_point": 60},
            "source_text": None,
        },
        {
            "type": "maximum_reorder_point",
            "parameters": {"reorder_point": 80},
            "source_text": None,
        },
        {
            "type": "storage_capacity",
            "parameters": {"capacity": 400},
            "source_text": None,
        },
    ]

    result = optimize_q_r_policy(inputs, constraints)

    assert result["solver_status"] == "Optimal"
    assert result["converged"] is True
    assert result["order_quantity"] == 274
    assert result["reorder_point"] == 67
    assert result["safety_stock"] == 7
    assert result["expected_annual_cost"] == pytest.approx(14_011.35, abs=0.01)
    assert result["iterations"] == 5
    assert "1 - h Q/(p λ)" in result["model_representation"]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("annual_demand", 0.0),
        ("fixed_order_cost", -1.0),
        ("holding_cost", 0.0),
        ("penalty_cost", 0.0),
        ("lead_time_demand_mean", -1.0),
        ("lead_time_demand_std", -1.0),
        ("annual_demand", math.inf),
        ("annual_demand", True),
    ],
)
def test_rejects_missing_or_invalid_inputs(
    valid_inputs: dict,
    field_name: str,
    value: object,
) -> None:
    inputs = {**valid_inputs, field_name: value}
    assert validate_q_r_policy_inputs(inputs)

    missing_inputs = dict(valid_inputs)
    missing_inputs.pop(field_name)
    assert validate_q_r_policy_inputs(missing_inputs)


@pytest.mark.parametrize(
    ("constraint_type", "parameter_name", "value"),
    [
        ("minimum_order_quantity", "quantity", 0.0),
        ("maximum_order_quantity", "quantity", -1.0),
        ("minimum_reorder_point", "reorder_point", -1.0),
        ("maximum_reorder_point", "reorder_point", -1.0),
        ("storage_capacity", "capacity", 0.0),
    ],
)
def test_rejects_invalid_constraints(
    valid_inputs: dict,
    constraint_type: str,
    parameter_name: str,
    value: float,
) -> None:
    constraints = [
        {
            "type": constraint_type,
            "parameters": {parameter_name: value},
            "source_text": None,
        }
    ]
    assert validate_q_r_policy_constraints(valid_inputs, constraints)


def test_rejects_minimum_above_maximum(valid_inputs: dict) -> None:
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {"quantity": 200.0},
            "source_text": None,
        },
        {
            "type": "maximum_order_quantity",
            "parameters": {"quantity": 100.0},
            "source_text": None,
        },
        {
            "type": "minimum_reorder_point",
            "parameters": {"reorder_point": 500.0},
            "source_text": None,
        },
        {
            "type": "maximum_reorder_point",
            "parameters": {"reorder_point": 400.0},
            "source_text": None,
        },
    ]
    errors = validate_q_r_policy_constraints(valid_inputs, constraints)
    assert len(errors) == 2


@pytest.mark.parametrize(
    ("constraint_type", "parameter_name"),
    [
        ("minimum_order_quantity", "quantity"),
        ("maximum_order_quantity", "quantity"),
        ("minimum_reorder_point", "reorder_point"),
        ("maximum_reorder_point", "reorder_point"),
        ("storage_capacity", "capacity"),
    ],
)
def test_rejects_fractional_unit_constraints(
    valid_inputs: dict,
    constraint_type: str,
    parameter_name: str,
) -> None:
    constraints = [
        {
            "type": constraint_type,
            "parameters": {parameter_name: 10.5},
            "source_text": None,
        }
    ]

    errors = validate_q_r_policy_constraints(
        valid_inputs,
        constraints,
    )

    assert any("ganzen Einheiten" in error for error in errors)


def test_duplicate_constraints_raise_error(valid_inputs: dict) -> None:
    constraint = {
        "type": "minimum_order_quantity",
        "parameters": {"quantity": 10.0},
        "source_text": None,
    }
    with pytest.raises(ValueError, match="mehrfach"):
        create_q_r_policy_specification(valid_inputs, [constraint, constraint])


def test_unsupported_constraint_raises_error(valid_inputs: dict) -> None:
    with pytest.raises(ValueError, match="nicht unterstützt"):
        create_q_r_policy_specification(
            valid_inputs,
            [
                {
                    "type": "budget_limit",
                    "parameters": {"budget": 1_000.0},
                    "source_text": None,
                }
            ],
        )


def test_one_iteration_returns_iteration_limit(valid_inputs: dict) -> None:
    specification = QRPolicySpecification(
        **valid_inputs,
        max_iterations=1,
    )
    result = solve_q_r_policy(specification)
    assert result.solver_status == "IterationLimit"
    assert result.converged is False
    assert result.iterations == 1


def test_storage_capacity_can_make_result_infeasible(valid_inputs: dict) -> None:
    result = optimize_q_r_policy(
        valid_inputs,
        [
            {
                "type": "storage_capacity",
                "parameters": {"capacity": 10.0},
                "source_text": None,
            }
        ],
    )
    assert result["solver_status"] == "Infeasible"
    assert result["order_quantity"] is None


def test_result_presentation_separates_business_and_technical_metrics() -> None:
    presentation = build_q_r_policy_result_presentation(
        {
            "solver_status": "Optimal",
            "order_quantity": 640,
            "reorder_point": 510,
            "safety_stock": 110,
            "expected_annual_cost": 5432.1,
            "iterations": 3,
        }
    )

    assert presentation["metrics"] == [
        {
            "label": "Empfohlene Bestellmenge Q",
            "value": "640 Einheiten",
        },
        {
            "label": "Meldebestand R",
            "value": "510 Einheiten",
        },
        {
            "label": "Sicherheitsbestand",
            "value": "110 Einheiten",
        },
        {
            "label": "Erwartete jährliche Kosten",
            "value": "5.432,10 €",
        },
    ]
    assert presentation["technical_metrics"] == [
        {
            "label": "Iterationen",
            "value": "3",
        },
    ]
