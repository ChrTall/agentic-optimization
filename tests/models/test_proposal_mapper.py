import pytest

from agentic_optimization.models.proposal_mapper import (
    build_specification_from_proposal,
    validate_component_selection_proposal,
)
from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ProductSpecification,
)


def create_proposal() -> ComponentSelectionProposal:
    return {
        "core_id": "single_period_newsvendor",
        "selection_status": "complete",
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "reason": "Verkaufserlöse sollen berücksichtigt werden.",
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
                    "quantity": 50.0,
                },
                "reason": "Eine Mindestmenge wurde erkannt.",
                "sources": ["Lieferantenvertrag"],
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


def test_builds_specification_from_agent_proposal() -> None:
    specification = build_specification_from_proposal(
        create_proposal(),
        create_products(),
        {
            "currency": "EUR",
        },
    )

    assert specification["core_id"] == (
        "single_period_newsvendor"
    )
    assert specification["objective"] == {
        "sense": "maximize",
        "component_ids": [
            "sales_revenue",
            "purchase_cost",
        ],
    }
    assert specification["constraints"] == [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 50.0,
            },
        }
    ]
    assert specification["global_facts"] == {
        "currency": "EUR",
    }


def test_does_not_mutate_agent_or_extracted_data() -> None:
    proposal = create_proposal()
    products = create_products()
    global_facts = {
        "currency": "EUR",
    }

    specification = build_specification_from_proposal(
        proposal,
        products,
        global_facts,
    )

    specification["products"][0]["facts"][
        "selling_price"
    ] = 99.0
    specification["constraints"][0]["parameters"][
        "quantity"
    ] = 500.0
    specification["global_facts"]["currency"] = "USD"

    assert products[0]["facts"]["selling_price"] == 12.0
    assert proposal["constraints"][0]["parameters"][
        "quantity"
    ] == 50.0
    assert global_facts["currency"] == "EUR"


def test_accepts_complete_agent_proposal() -> None:
    errors = validate_component_selection_proposal(
        create_proposal()
    )

    assert errors == []


def test_rejects_non_object_agent_proposal() -> None:
    errors = validate_component_selection_proposal(
        "invalid"
    )

    assert errors == [
        "Der Bausteinvorschlag muss als Objekt vorliegen."
    ]


def test_rejects_unknown_objective_component() -> None:
    proposal = create_proposal()
    proposal["objective_components"][0][
        "component_id"
    ] = "invented_component"

    errors = validate_component_selection_proposal(
        proposal
    )

    assert any(
        "invented_component" in error
        and "nicht registriert" in error
        for error in errors
    )


def test_mapper_rejects_constraint_without_parameters() -> None:
    proposal = create_proposal()
    del proposal["constraints"][0]["parameters"]

    with pytest.raises(
        ValueError,
        match="Parameter der Nebenbedingung",
    ):
        build_specification_from_proposal(
            proposal,
            create_products(),
            {},
        )


def test_rejects_objective_incompatible_with_core() -> None:
    proposal = create_proposal()
    proposal["objective_components"][0][
        "component_id"
    ] = "annual_ordering_cost"

    errors = validate_component_selection_proposal(
        proposal
    )

    assert any(
        "annual_ordering_cost" in error
        and "single_period_newsvendor" in error
        and "nicht mit" in error
        for error in errors
    )


def test_rejects_constraint_incompatible_with_core() -> None:
    proposal: ComponentSelectionProposal = {
        "core_id": "eoq",
        "objective_sense": "minimize",
        "selection_status": "complete",
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
        "constraints": [
            {
                "component_id": "budget_limit",
                "parameters": {
                    "budget": 1000.0,
                },
                "reason": "Ein Budget wurde genannt.",
                "sources": ["source_1"],
            }
        ],
        "unresolved_requirements": [],
    }

    errors = validate_component_selection_proposal(
        proposal
    )

    assert any(
        "budget_limit" in error
        and "eoq" in error
        and "nicht mit" in error
        for error in errors
    )


def test_rejects_missing_required_eoq_component() -> None:
    proposal: ComponentSelectionProposal = {
        "core_id": "eoq",
        "objective_sense": "minimize",
        "selection_status": "complete",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten wurden genannt.",
                "sources": ["source_1"],
            }
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }

    errors = validate_component_selection_proposal(
        proposal
    )

    assert any(
        "annual_holding_cost" in error
        and "benötigt" in error
        for error in errors
    )


def test_accepts_consistent_unsupported_proposal() -> None:
    proposal = create_proposal()
    proposal["selection_status"] = "unsupported"
    proposal["unresolved_requirements"] = [
        {
            "description": (
                "Mengenabhängiger Einkaufspreis"
            ),
            "resolution": "unsupported",
            "reason": (
                "Ein Mengenrabattmodell ist für diesen "
                "Grundmodellkern nicht registriert."
            ),
            "sources": ["Lieferantenvertrag"],
        }
    ]

    errors = validate_component_selection_proposal(
        proposal
    )

    assert errors == []


def test_rejects_complete_status_with_unresolved_requirement() -> None:
    proposal = create_proposal()
    proposal["unresolved_requirements"] = [
        {
            "description": (
                "Rückgabemöglichkeit ist unklar"
            ),
            "resolution": "clarification",
            "reason": (
                "Es ist nicht bekannt, ob Restbestand "
                "zurückgegeben werden kann."
            ),
            "sources": ["Freitext"],
        }
    ]

    errors = validate_component_selection_proposal(
        proposal
    )

    assert any(
        "Auswahlstatus 'complete'" in error
        and "needs_clarification" in error
        for error in errors
    )


def test_mapper_rejects_unsupported_proposal() -> None:
    proposal = create_proposal()
    proposal["selection_status"] = "unsupported"
    proposal["unresolved_requirements"] = [
        {
            "description": "Mengenrabatt",
            "resolution": "unsupported",
            "reason": (
                "Kein kompatibler Baustein verfügbar."
            ),
            "sources": ["Lieferantenvertrag"],
        }
    ]

    with pytest.raises(
        ValueError,
        match="nicht vollständigen",
    ):
        build_specification_from_proposal(
            proposal,
            create_products(),
            {},
        )


def test_mapper_rejects_proposal_needing_clarification() -> None:
    proposal = create_proposal()
    proposal["selection_status"] = (
        "needs_clarification"
    )
    proposal["unresolved_requirements"] = [
        {
            "description": (
                "Umgang mit unverkaufter Ware"
            ),
            "resolution": "clarification",
            "reason": (
                "Rückgabe oder Entsorgung ist unklar."
            ),
            "sources": ["Freitext"],
        }
    ]

    with pytest.raises(
        ValueError,
        match="nicht vollständigen",
    ):
        build_specification_from_proposal(
            proposal,
            create_products(),
            {},
        )


def test_rejects_malformed_unresolved_requirement() -> None:
    proposal = create_proposal()
    proposal["selection_status"] = (
        "needs_clarification"
    )
    proposal["unresolved_requirements"] = [
        {
            "description": "",
            "resolution": "clarification",
            "reason": "",
            "sources": [],
        }
    ]

    errors = validate_component_selection_proposal(
        proposal
    )

    assert any(
        "Beschreibung" in error
        for error in errors
    )
    assert any(
        "Begründung" in error
        for error in errors
    )
    assert any(
        "gültige Quelle" in error
        for error in errors
    )


def test_unsupported_status_dominates_mixed_requirements() -> None:
    proposal = create_proposal()
    proposal["selection_status"] = (
        "needs_clarification"
    )
    proposal["unresolved_requirements"] = [
        {
            "description": "Rückgabewert unklar",
            "resolution": "clarification",
            "reason": (
                "Der Rückgabewert wurde nicht angegeben."
            ),
            "sources": ["Freitext"],
        },
        {
            "description": "Mengenrabatt",
            "resolution": "unsupported",
            "reason": (
                "Kein kompatibles Mengenrabattmodell "
                "ist registriert."
            ),
            "sources": ["Lieferantenvertrag"],
        },
    ]

    errors = validate_component_selection_proposal(
        proposal
    )

    assert any(
        "erwartet wird 'unsupported'" in error
        for error in errors
    )
