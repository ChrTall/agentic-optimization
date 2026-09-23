import math

import pytest

from agentic_optimization.models.reorder_point.core import (
    CORE_DEFINITION,
    calculate_expected_shortage,
    evaluate_q_r_candidate,
    get_order_multiple,
    validate_q_r_forecast,
    validate_q_r_semantics,
    determine_reorder_point_candidates,
    find_optimal_q_r_policy,
    get_q_r_bounds,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)


def create_specification() -> ModelSpecification:
    return {
        "core_id": "continuous_review_q_r",
        "products": [
            {
                "product_id": "product_1",
                "forecast": {
                    "expected_value": 100.0,
                    "quantiles": [
                        {
                            "level": 0.1,
                            "value": 70.0,
                        },
                        {
                            "level": 0.5,
                            "value": 100.0,
                        },
                        {
                            "level": 0.9,
                            "value": 140.0,
                        },
                    ],
                    "horizon_days": 30,
                },
                "facts": {
                    "lead_time_days": 10,
                },
            }
        ],
        "global_facts": {},
        "objective": {
            "sense": "minimize",
            "component_ids": [
                "annual_ordering_cost",
                "annual_holding_cost",
                "annual_shortage_cost",
            ],
        },
        "constraints": [],
    }


def test_defines_q_r_core_quantities() -> None:
    assert set(CORE_DEFINITION["provided_quantities"]) == {
        "order_quantity",
        "reorder_point",
        "orders_per_year",
        "average_inventory",
        "expected_shortage_per_cycle",
        "annual_shortage_quantity",
        "safety_stock",
    }


def test_requires_lead_time_as_core_fact() -> None:
    assert {
        field["name"]
        for field in CORE_DEFINITION[
            "required_product_facts"
        ]
    } == {
        "lead_time_days",
    }


def test_accepts_two_sided_forecast_quantiles() -> None:
    specification = create_specification()

    errors = validate_q_r_forecast(
        specification["products"][0]["forecast"]
    )

    assert errors == []


def test_rejects_one_sided_forecast_quantiles() -> None:
    specification = create_specification()
    specification["products"][0]["forecast"][
        "quantiles"
    ] = [
        {
            "level": 0.5,
            "value": 100.0,
        },
        {
            "level": 0.9,
            "value": 140.0,
        },
    ]

    errors = validate_q_r_forecast(
        specification["products"][0]["forecast"]
    )

    assert any(
        "unterhalb und eines oberhalb" in error
        for error in errors
    )


def test_requires_cost_minimization() -> None:
    specification = create_specification()
    specification["objective"]["sense"] = "maximize"

    errors = validate_q_r_semantics(specification)

    assert any(
        "Kosten minimieren" in error
        for error in errors
    )


def test_calculates_expected_shortage_at_mean() -> None:
    shortage = calculate_expected_shortage(
        reorder_point=100.0,
        lead_time_demand_mean=100.0,
        lead_time_demand_std=10.0,
    )

    assert shortage == pytest.approx(
        10.0 / math.sqrt(2.0 * math.pi)
    )


def test_evaluates_q_r_cost_components() -> None:
    result = evaluate_q_r_candidate(
        annual_demand=1000.0,
        ordering_cost=50.0,
        holding_cost=2.0,
        shortage_cost=20.0,
        lead_time_demand_mean=100.0,
        lead_time_demand_std=10.0,
        order_quantity=250.0,
        reorder_point=100.0,
    )

    assert result.orders_per_year == pytest.approx(4.0)
    assert result.safety_stock == pytest.approx(0.0)
    assert result.average_inventory == pytest.approx(125.0)
    assert result.expected_shortage_per_cycle == pytest.approx(
        3.989422804
    )
    assert result.annual_ordering_cost == pytest.approx(200.0)
    assert result.annual_holding_cost == pytest.approx(250.0)
    assert result.annual_shortage_cost == pytest.approx(
        319.1538243
    )
    assert result.annual_relevant_cost == pytest.approx(
        769.1538243
    )


def test_rejects_non_positive_order_quantity() -> None:
    with pytest.raises(
        ValueError,
        match="größer als null",
    ):
        evaluate_q_r_candidate(
            annual_demand=1000.0,
            ordering_cost=50.0,
            holding_cost=2.0,
            shortage_cost=20.0,
            lead_time_demand_mean=100.0,
            lead_time_demand_std=10.0,
            order_quantity=0.0,
            reorder_point=100.0,
        )


def test_determines_relevant_reorder_point_candidates() -> None:
    candidates = determine_reorder_point_candidates(
        annual_demand=7300.0,
        holding_cost=50.0,
        shortage_cost=10.0,
        lead_time_demand_mean=60.0,
        lead_time_demand_std=6.928,
        order_quantity=274,
        minimum_reorder_point=60,
        maximum_reorder_point=80,
    )

    assert candidates == [
        60,
        66,
        67,
        80,
    ]


def test_respects_reorder_point_bounds() -> None:
    candidates = determine_reorder_point_candidates(
        annual_demand=7300.0,
        holding_cost=50.0,
        shortage_cost=10.0,
        lead_time_demand_mean=60.0,
        lead_time_demand_std=6.928,
        order_quantity=274,
        minimum_reorder_point=70,
        maximum_reorder_point=75,
    )

    assert candidates == [
        70,
        75,
    ]


def test_handles_deterministic_lead_time_demand() -> None:
    candidates = determine_reorder_point_candidates(
        annual_demand=1000.0,
        holding_cost=2.0,
        shortage_cost=20.0,
        lead_time_demand_mean=100.0,
        lead_time_demand_std=0.0,
        order_quantity=250,
    )

    assert candidates == [
        0,
        100,
    ]


def test_finds_cost_minimal_integer_q_r_policy() -> None:
    result = find_optimal_q_r_policy(
        annual_demand=7300.0,
        ordering_cost=250.0,
        holding_cost=50.0,
        shortage_cost=10.0,
        lead_time_demand_mean=60.0,
        lead_time_demand_std=6.928,
        minimum_order_quantity=250,
        maximum_order_quantity=300,
        minimum_reorder_point=60,
        maximum_reorder_point=80,
    )

    assert result.order_quantity == 274
    assert result.reorder_point == 66
    assert result.safety_stock == pytest.approx(6.0)
    assert result.annual_relevant_cost == pytest.approx(
        14007.7774518
    )


def test_respects_order_quantity_bounds() -> None:
    result = find_optimal_q_r_policy(
        annual_demand=10000.0,
        ordering_cost=100.0,
        holding_cost=5.0,
        shortage_cost=25.0,
        lead_time_demand_mean=400.0,
        lead_time_demand_std=50.0,
        minimum_order_quantity=700,
        maximum_order_quantity=750,
    )

    assert 700 <= result.order_quantity <= 750


def test_rejects_inconsistent_q_r_bounds() -> None:
    with pytest.raises(
        ValueError,
        match="Mindestbestellmenge",
    ):
        find_optimal_q_r_policy(
            annual_demand=1000.0,
            ordering_cost=50.0,
            holding_cost=2.0,
            shortage_cost=20.0,
            lead_time_demand_mean=100.0,
            lead_time_demand_std=10.0,
            minimum_order_quantity=300,
            maximum_order_quantity=200,
        )


def test_reads_integer_q_r_bounds_from_constraints() -> None:
    specification = create_specification()
    specification["constraints"] = [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 10.2,
            },
        },
        {
            "component_id": "maximum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 20.8,
            },
        },
        {
            "component_id": "minimum_reorder_point",
            "product_id": "product_1",
            "parameters": {
                "reorder_point": 5.2,
            },
        },
        {
            "component_id": "maximum_reorder_point",
            "product_id": "product_1",
            "parameters": {
                "reorder_point": 12.9,
            },
        },
    ]

    assert get_q_r_bounds(
        specification,
        "product_1",
    ) == (
        11,
        20,
        6,
        12,
    )


def test_q_r_bounds_are_product_specific() -> None:
    specification = create_specification()
    specification["constraints"] = [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "another_product",
            "parameters": {
                "quantity": 500.0,
            },
        }
    ]

    assert get_q_r_bounds(
        specification,
        "product_1",
    ) == (
        1,
        None,
        0,
        None,
    )


def test_rejects_conflicting_order_quantity_bounds() -> None:
    specification = create_specification()
    specification["constraints"] = [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 200.0,
            },
        },
        {
            "component_id": "maximum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 100.0,
            },
        },
    ]

    errors = validate_q_r_semantics(specification)

    assert any(
        "Mindestbestellmenge" in error
        and "Höchstbestellmenge" in error
        for error in errors
    )


def test_rejects_conflicting_reorder_point_bounds() -> None:
    specification = create_specification()
    specification["constraints"] = [
        {
            "component_id": "minimum_reorder_point",
            "product_id": "product_1",
            "parameters": {
                "reorder_point": 80.0,
            },
        },
        {
            "component_id": "maximum_reorder_point",
            "product_id": "product_1",
            "parameters": {
                "reorder_point": 60.0,
            },
        },
    ]

    errors = validate_q_r_semantics(specification)

    assert any(
        "Mindestmeldebestand" in error
        and "Höchstmeldebestand" in error
        for error in errors
    )


def test_respects_order_multiple() -> None:
    result = find_optimal_q_r_policy(
        annual_demand=7300.0,
        ordering_cost=250.0,
        holding_cost=50.0,
        shortage_cost=10.0,
        lead_time_demand_mean=60.0,
        lead_time_demand_std=6.928,
        minimum_order_quantity=250,
        maximum_order_quantity=300,
        order_multiple=50,
        minimum_reorder_point=60,
        maximum_reorder_point=80,
    )

    assert result.order_quantity == 300
    assert result.order_quantity % 50 == 0
    assert result.reorder_point == 66
    assert result.annual_relevant_cost == pytest.approx(
        14063.4367393
    )


def test_reads_order_multiple_from_constraints() -> None:
    specification = create_specification()
    specification["constraints"] = [
        {
            "component_id": "order_multiple",
            "product_id": "product_1",
            "parameters": {
                "multiple": 50,
            },
        }
    ]

    assert get_order_multiple(
        specification,
        "product_1",
    ) == 50


def test_returns_no_order_multiple_when_not_configured() -> None:
    specification = create_specification()

    assert get_order_multiple(
        specification,
        "product_1",
    ) is None


def test_rejects_order_multiple_above_maximum() -> None:
    specification = create_specification()
    specification["constraints"] = [
        {
            "component_id": "maximum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 50,
            },
        },
        {
            "component_id": "order_multiple",
            "product_id": "product_1",
            "parameters": {
                "multiple": 100,
            },
        },
    ]

    errors = validate_q_r_semantics(specification)

    assert any(
        "kein zulässiges Bestellvielfaches" in error
        for error in errors
    )