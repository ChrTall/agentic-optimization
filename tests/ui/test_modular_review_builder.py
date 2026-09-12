import pytest

from agentic_optimization.models.proposal_mapper import (
    build_specification_from_proposal,
)
from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ProductSpecification,
)
from agentic_optimization.ui.modular_review_builder import (
    build_modular_review_data,
)


def create_proposal() -> ComponentSelectionProposal:
    return {
        "core_id": "single_period_newsvendor",
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "reason": "Verkaufserlöse wurden genannt.",
                "sources": ["source_1"],
            },
            {
                "component_id": "purchase_cost",
                "reason": "Einkaufskosten wurden genannt.",
                "sources": ["source_1"],
            },
        ],
        "constraints": [
            {
                "component_id": "minimum_order_quantity",
                "product_id": "product_1",
                "parameters": {
                    "quantity": 50.0,
                },
                "reason": "Eine Mindestmenge wurde genannt.",
                "sources": ["source_1"],
            }
        ],
    }


def create_products() -> list[ProductSpecification]:
    return [
        {
            "product_id": "product_1",
            "forecast": {
                "expected_value": 100.0,
                "quantiles": [
                    {
                        "level": 0.5,
                        "value": 100.0,
                    }
                ],
                "horizon_days": 30,
            },
            "facts": {
                "selling_price": 12.0,
                "purchase_cost": 5.0,
            },
        }
    ]


def test_builds_review_data() -> None:
    proposal = create_proposal()
    specification = build_specification_from_proposal(
        proposal,
        create_products(),
        {},
    )

    review = build_modular_review_data(
        proposal,
        specification,
    )

    assert review["core_name"] == (
        "Einperiodiges Newsvendor-Modell"
    )
    assert review["objective_sense"] == "maximize"
    assert review["ready_for_confirmation"] is True
    assert review["validation_errors"] == []
    assert review["missing_inputs"] == []

    assert review["objective_components"][0] == {
        "component_id": "sales_revenue",
        "name": "Verkaufserlös",
        "effect": "benefit",
        "reason": "Verkaufserlöse wurden genannt.",
        "sources": ["source_1"],
    }

    assert review["constraints"] == [
        {
            "component_id": "minimum_order_quantity",
            "name": "Mindestbestellmenge",
            "scope": "per_product",
            "product_id": "product_1",
            "parameters": {
                "quantity": 50.0,
            },
            "reason": "Eine Mindestmenge wurde genannt.",
            "sources": ["source_1"],
        }
    ]


def test_review_reports_missing_component_input() -> None:
    proposal = create_proposal()
    proposal["objective_components"].append(
        {
            "component_id": "salvage_revenue",
            "reason": "Restbestände besitzen einen Restwert.",
            "sources": ["source_1"],
        }
    )

    specification = build_specification_from_proposal(
        proposal,
        create_products(),
        {},
    )

    review = build_modular_review_data(
        proposal,
        specification,
    )

    assert review["ready_for_confirmation"] is False
    assert review["validation_errors"] == []
    assert len(review["missing_inputs"]) == 1
    assert review["missing_inputs"][0]["field"]["name"] == (
        "salvage_value"
    )


def test_rejects_mismatched_review_data() -> None:
    proposal = create_proposal()
    specification = build_specification_from_proposal(
        proposal,
        create_products(),
        {},
    )
    specification["objective"]["component_ids"] = [
        "purchase_cost",
    ]

    with pytest.raises(
        ValueError,
        match="Zielfunktionsbausteine",
    ):
        build_modular_review_data(
            proposal,
            specification,
        )


def test_builds_eoq_review_data() -> None:
    proposal: ComponentSelectionProposal = {
        "core_id": "eoq",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten wurden genannt.",
                "sources": ["source_1"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten wurden genannt.",
                "sources": ["source_1"],
            },
        ],
        "constraints": [],
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
                "holding_cost": 12.0,
            },
        }
    ]

    specification = build_specification_from_proposal(
        proposal,
        products,
        {},
    )

    review = build_modular_review_data(
        proposal,
        specification,
    )

    assert review["core_id"] == "eoq"
    assert review["core_name"] == (
        "Wirtschaftliche Bestellmenge (EOQ)"
    )
    assert review["objective_sense"] == "minimize"
    assert review["ready_for_confirmation"] is True
    assert review["validation_errors"] == []
    assert review["missing_inputs"] == []

    assert {
        component["component_id"]
        for component in review["objective_components"]
    } == {
        "annual_ordering_cost",
        "annual_holding_cost",
    }
