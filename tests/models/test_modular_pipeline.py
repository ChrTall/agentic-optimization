import pytest

from agentic_optimization.models.execution import (
    execute_model_specification,
)
from agentic_optimization.models.proposal_mapper import (
    build_specification_from_proposal,
)
from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ProductSpecification,
)
from agentic_optimization.models.validator import (
    validate_model_specification,
)
from agentic_optimization.ui.modular_input_builder import (
    apply_modular_input_value,
    build_modular_input_layout,
)


def test_complete_modular_newsvendor_pipeline() -> None:
    proposal: ComponentSelectionProposal = {
        "core_id": "single_period_newsvendor",
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "reason": "Verkaufserlöse wurden beschrieben.",
                "sources": ["Freitext"],
            },
            {
                "component_id": "salvage_revenue",
                "reason": "Restbestände besitzen einen Restwert.",
                "sources": ["Freitext"],
            },
            {
                "component_id": "purchase_cost",
                "reason": "Einkaufskosten wurden angegeben.",
                "sources": ["Lieferantenvertrag"],
            },
        ],
        "constraints": [
            {
                "component_id": "minimum_order_quantity",
                "product_id": "product_1",
                "parameters": {
                    "quantity": 110.0,
                },
                "reason": "Eine Mindestbestellmenge wurde erkannt.",
                "sources": ["Lieferantenvertrag"],
            }
        ],
    }

    products: list[ProductSpecification] = [
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
                "selling_price": 12.0,
                "purchase_cost": 5.0,
            },
        }
    ]

    specification = build_specification_from_proposal(
        proposal,
        products,
        {},
    )

    initial_validation = validate_model_specification(
        specification
    )

    assert initial_validation["is_valid"] is False
    assert initial_validation["errors"] == []
    assert len(initial_validation["missing_inputs"]) == 1

    missing_input = initial_validation["missing_inputs"][0]

    assert missing_input["source"] == "product_fact"
    assert missing_input["field"]["name"] == "salvage_value"

    layout = build_modular_input_layout(specification)

    assert layout["product_groups"][0][
        "fact_inputs"
    ] == [missing_input]

    completed_specification = apply_modular_input_value(
        specification,
        missing_input,
        1.5,
    )

    completed_validation = validate_model_specification(
        completed_specification
    )

    assert completed_validation["is_valid"] is True

    result = execute_model_specification(
        completed_specification
    )

    assert result["status"] == "optimal"
    assert result["solution_method"] == "linear_programming"
    assert result["decisions"] == [
        {
            "quantity_id": "order_quantity",
            "product_id": "product_1",
            "value": pytest.approx(110.0),
        }
    ]
    assert result["errors"] == []


def test_complete_modular_q_r_pipeline() -> None:
    proposal: ComponentSelectionProposal = {
        "core_id": "continuous_review_q_r",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": (
                    "Fixe Bestellkosten wurden beschrieben."
                ),
                "sources": ["Freitext"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": (
                    "Lagerhaltungskosten wurden beschrieben."
                ),
                "sources": ["Freitext"],
            },
            {
                "component_id": "annual_shortage_cost",
                "reason": (
                    "Fehlmengenkosten wurden beschrieben."
                ),
                "sources": ["Freitext"],
            },
        ],
        "constraints": [
            {
                "component_id": "minimum_order_quantity",
                "product_id": "product_1",
                "parameters": {
                    "quantity": 300.0,
                },
                "reason": (
                    "Eine Mindestbestellmenge wurde erkannt."
                ),
                "sources": ["Lieferantenvertrag"],
            }
        ],
    }

    products: list[ProductSpecification] = [
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
    ]

    specification = build_specification_from_proposal(
        proposal,
        products,
        {},
    )

    initial_validation = validate_model_specification(
        specification
    )

    assert initial_validation["is_valid"] is False
    assert initial_validation["errors"] == []

    missing_by_name = {
        missing_input["field"]["name"]: missing_input
        for missing_input
        in initial_validation["missing_inputs"]
    }

    assert set(missing_by_name) == {
        "ordering_cost",
        "holding_cost",
        "shortage_cost",
    }

    layout = build_modular_input_layout(specification)

    assert {
        missing_input["field"]["name"]
        for missing_input
        in layout["product_groups"][0]["fact_inputs"]
    } == {
        "ordering_cost",
        "holding_cost",
        "shortage_cost",
    }

    values = {
        "ordering_cost": 50.0,
        "holding_cost": 2.0,
        "shortage_cost": 20.0,
    }

    completed_specification = specification

    for field_name, value in values.items():
        completed_specification = (
            apply_modular_input_value(
                completed_specification,
                missing_by_name[field_name],
                value,
            )
        )

    completed_validation = validate_model_specification(
        completed_specification
    )

    assert completed_validation["is_valid"] is True

    result = execute_model_specification(
        completed_specification
    )

    decisions = {
        decision["quantity_id"]: decision["value"]
        for decision in result["decisions"]
    }

    assert result["status"] == "optimal"
    assert result["solution_method"] == "numerical"
    assert decisions["order_quantity"] >= 300
    assert decisions["reorder_point"] >= 0
    assert result["errors"] == []