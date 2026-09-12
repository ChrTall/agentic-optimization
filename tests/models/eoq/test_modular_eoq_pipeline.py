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


def test_complete_modular_eoq_pipeline() -> None:
    proposal: ComponentSelectionProposal = {
        "core_id": "eoq",
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
                "expected_value": 3500.0,
                "horizon_days": 365,
            },
            "facts": {
                "ordering_cost": 120.0,
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
    assert len(
        initial_validation["missing_inputs"]
    ) == 1

    missing_input = initial_validation[
        "missing_inputs"
    ][0]

    assert missing_input["source"] == "product_fact"
    assert missing_input["field"]["name"] == "holding_cost"

    layout = build_modular_input_layout(specification)

    assert layout["product_groups"][0][
        "fact_inputs"
    ] == [missing_input]

    complete_specification = apply_modular_input_value(
        specification,
        missing_input,
        12.0,
    )

    final_validation = validate_model_specification(
        complete_specification
    )
    result = execute_model_specification(
        complete_specification
    )

    assert final_validation["is_valid"] is True
    assert result["status"] == "optimal"
    assert result["solution_method"] == "analytical"
    assert result["objective_value"] == pytest.approx(
        3200.0
    )
    assert result["decisions"] == [
        {
            "quantity_id": "order_quantity",
            "product_id": "product_1",
            "value": pytest.approx(300.0),
        }
    ]
