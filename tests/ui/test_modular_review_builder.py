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
    build_modular_selection_issue_data,
)


def create_proposal() -> ComponentSelectionProposal:
    return {
        "core_id": "single_period_newsvendor",
        "selection_status": "complete",
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
        "unresolved_requirements": [],
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


def test_review_shows_selected_disposal_cost() -> None:
    proposal = create_proposal()

    proposal["objective_components"].append(
        {
            "component_id": "disposal_cost",
            "reason": (
                "Unverkaufte Ware muss kostenpflichtig "
                "entsorgt werden."
            ),
            "sources": ["source_1"],
        }
    )

    products = create_products()
    products[0]["facts"]["disposal_cost"] = 20.0

    specification = build_specification_from_proposal(
        proposal,
        products,
        {},
    )

    review = build_modular_review_data(
        proposal,
        specification,
    )

    disposal_component = next(
        component
        for component in review["objective_components"]
        if component["component_id"] == "disposal_cost"
    )

    assert disposal_component == {
        "component_id": "disposal_cost",
        "name": "Entsorgungskosten",
        "effect": "cost",
        "reason": (
            "Unverkaufte Ware muss kostenpflichtig "
            "entsorgt werden."
        ),
        "sources": ["source_1"],
    }

    assert review["ready_for_confirmation"] is True
    assert review["missing_inputs"] == []


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
        "selection_status": "complete",
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
        "unresolved_requirements": [],
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


def test_builds_q_r_review_data() -> None:
    proposal: ComponentSelectionProposal = {
        "core_id": "continuous_review_q_r",
        "selection_status": "complete",
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
            {
                "component_id": "annual_shortage_cost",
                "reason": "Fehlmengenkosten wurden genannt.",
                "sources": ["source_1"],
            },
        ],
        "constraints": [
            {
                "component_id": "minimum_reorder_point",
                "product_id": "product_1",
                "parameters": {
                    "reorder_point": 50.0,
                },
                "reason": (
                    "Ein Mindestmeldebestand wurde genannt."
                ),
                "sources": ["source_1"],
            }
        ],
        "unresolved_requirements": [],
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
                "ordering_cost": 50.0,
                "holding_cost": 2.0,
                "shortage_cost": 20.0,
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

    assert review["core_id"] == "continuous_review_q_r"
    assert review["core_name"] == (
        "Kontinuierliche Bestandsüberprüfung (Q,R)"
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
        "annual_shortage_cost",
    }

    assert review["constraints"][0][
        "component_id"
    ] == "minimum_reorder_point"


def test_copies_product_and_global_review_data() -> None:
    proposal = create_proposal()
    specification = build_specification_from_proposal(
        proposal,
        create_products(),
        {
            "currency": "EUR",
        },
    )

    review = build_modular_review_data(
        proposal,
        specification,
    )

    assert review["products"] == [
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
    assert review["global_facts"] == {
        "currency": "EUR",
    }

    review["products"][0]["facts"][
        "selling_price"
    ] = 99.0
    review["global_facts"]["currency"] = "USD"

    assert specification["products"][0]["facts"][
        "selling_price"
    ] == 12.0
    assert specification["global_facts"][
        "currency"
    ] == "EUR"


def test_review_shows_order_multiple_constraint() -> None:
    proposal = create_proposal()

    proposal["constraints"].append(
        {
            "component_id": "order_multiple",
            "product_id": "product_1",
            "parameters": {
                "multiple": 60,
            },
            "reason": (
                "Bestellungen sind nur in vollständigen "
                "Verpackungseinheiten möglich."
            ),
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

    order_multiple = next(
        constraint
        for constraint in review["constraints"]
        if constraint["component_id"] == "order_multiple"
    )

    assert order_multiple == {
        "component_id": "order_multiple",
        "name": "Verpackungseinheit",
        "scope": "per_product",
        "product_id": "product_1",
        "parameters": {
            "multiple": 60,
        },
        "reason": (
            "Bestellungen sind nur in vollständigen "
            "Verpackungseinheiten möglich."
        ),
        "sources": ["source_1"],
    }

    assert review["ready_for_confirmation"] is True


def test_builds_unsupported_selection_issue_data() -> None:
    proposal = create_proposal()
    proposal["selection_status"] = "unsupported"
    proposal["unresolved_requirements"] = [
        {
            "description": "Mengenrabatt",
            "resolution": "unsupported",
            "reason": (
                "Für Mengenrabatte ist kein "
                "kompatibler Baustein verfügbar."
            ),
            "sources": ["source_1"],
        }
    ]

    issue_data = (
        build_modular_selection_issue_data(
            proposal
        )
    )

    assert issue_data == {
        "core_id": "single_period_newsvendor",
        "core_name": (
            "Einperiodiges Newsvendor-Modell"
        ),
        "selection_status": "unsupported",
        "unresolved_requirements": [
            {
                "description": "Mengenrabatt",
                "resolution": "unsupported",
                "reason": (
                    "Für Mengenrabatte ist kein "
                    "kompatibler Baustein verfügbar."
                ),
                "sources": ["source_1"],
            }
        ],
    }
    issue_data["unresolved_requirements"][0][
        "description"
    ] = "geändert"

    assert proposal["unresolved_requirements"][0][
        "description"
    ] == "Mengenrabatt"