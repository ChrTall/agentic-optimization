import pytest

from agentic_optimization.models.proposal_mapper import (
    build_specification_from_proposal,
    exclude_unsupported_requirements,
    validate_component_selection_proposal,
    build_component_selection_proposal,
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


def test_builds_proposal_from_semantic_decision() -> None:
    decision = {
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "reason": (
                    "Verkaufserlöse sollen berücksichtigt "
                    "werden."
                ),
                "sources": ["Freitext"],
            },
            {
                "component_id": "purchase_cost",
                "reason": (
                    "Einkaufskosten wurden angegeben."
                ),
                "sources": ["Lieferantenvertrag"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }

    proposal = build_component_selection_proposal(
        core_id="single_period_newsvendor",
        decision=decision,
    )

    assert proposal["core_id"] == (
        "single_period_newsvendor"
    )
    assert proposal["selection_status"] == "complete"
    assert proposal["objective_sense"] == "maximize"
    assert proposal["objective_components"] == (
        decision["objective_components"]
    )
    assert proposal["constraints"] == []
    assert proposal["unresolved_requirements"] == []


def test_profit_goal_adds_required_cost_component_without_value() -> None:
    proposal = build_component_selection_proposal(
        core_id="single_period_newsvendor",
        objective_goal="maximize_profit",
        decision={
            "objective_sense": "maximize",
            "objective_components": [{
                "component_id": "sales_revenue",
                "reason": "Erlöse berücksichtigen.",
                "sources": ["raw_input"],
            }],
            "constraints": [],
            "unresolved_requirements": [],
        },
    )

    assert {item["component_id"] for item in proposal[
        "objective_components"
    ]} == {"sales_revenue", "purchase_cost"}
    assert next(item for item in proposal["objective_components"]
                if item["component_id"] == "purchase_cost")[
        "sources"
    ] == ["core_definition"]


def test_adds_required_core_components_deterministically(
) -> None:
    decision = {
        "objective_sense": "minimize",
        "objective_components": [],
        "constraints": [],
        "unresolved_requirements": [],
    }

    proposal = build_component_selection_proposal(
        core_id="eoq",
        decision=decision,
    )

    assert [
        component["component_id"]
        for component in proposal[
            "objective_components"
        ]
    ] == [
        "annual_ordering_cost",
        "annual_holding_cost",
    ]

    assert all(
        component["reason"]
        == (
            "Vom ausgewählten Grundmodell "
            "verpflichtend benötigt."
        )
        for component in proposal[
            "objective_components"
        ]
    )

    assert all(
        component["sources"] == ["core_definition"]
        for component in proposal[
            "objective_components"
        ]
    )


def test_fixed_core_sets_its_objective_sense() -> None:
    decision = {
        "objective_sense": "maximize",
        "objective_components": [],
        "constraints": [],
        "unresolved_requirements": [],
    }

    for core_id in ("eoq", "continuous_review_q_r"):
        proposal = build_component_selection_proposal(
            core_id=core_id,
            decision=decision,
        )
        assert proposal["objective_sense"] == "minimize"
        assert validate_component_selection_proposal(proposal) == []


def test_rejects_external_proposal_with_wrong_fixed_core_sense() -> None:
    proposal = build_component_selection_proposal(
        core_id="eoq",
        decision={
            "objective_sense": "minimize",
            "objective_components": [],
            "constraints": [],
            "unresolved_requirements": [],
        },
    )
    proposal["objective_sense"] = "maximize"

    assert any(
        "erfordert die Zielrichtung 'minimize'" in error
        for error in validate_component_selection_proposal(
            proposal
        )
    )


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
            "candidate_component_ids": [],
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


@pytest.mark.parametrize(
    (
        "candidate_component_id",
        "expected_error",
    ),
    [
        (
            "unknown_component",
            "nicht registriert",
        ),
        (
            "annual_ordering_cost",
            "nicht mit",
        ),
        (
            "sales_revenue",
            "bereits ausgewählt",
        ),
    ],
)
def test_rejects_invalid_clarification_candidate(
    candidate_component_id: str,
    expected_error: str,
) -> None:
    proposal = create_proposal()
    proposal["selection_status"] = (
        "needs_clarification"
    )
    proposal["unresolved_requirements"] = [
        {
            "description": (
                "Fachlich mehrdeutige Anforderung"
            ),
            "resolution": "clarification",
            "candidate_component_ids": [
                candidate_component_id,
            ],
            "reason": (
                "Es muss zwischen möglichen "
                "Bausteinen gewählt werden."
            ),
            "sources": ["Freitext"],
        }
    ]

    errors = validate_component_selection_proposal(
        proposal
    )

    assert any(
        expected_error in error
        for error in errors
    )


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
            "candidate_component_ids": [
                "salvage_revenue",
                "disposal_cost",
            ],
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
            "candidate_component_ids": [],
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
            "candidate_component_ids": [
                "salvage_revenue",
                "disposal_cost",
            ],
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
            "candidate_component_ids": [],
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
            "description": (
                "Umgang mit unverkaufter Ware unklar"
            ),
            "resolution": "clarification",
            "reason": (
                "Es ist unklar, ob Restbestand "
                "zurückgegeben oder entsorgt wird."
            ),
            "sources": ["Freitext"],
            "candidate_component_ids": [
                "salvage_revenue",
                "disposal_cost",
            ],
        },
        {
            "description": "Mengenrabatt",
            "resolution": "unsupported",
            "candidate_component_ids": [],
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


def test_excludes_user_confirmed_unsupported_requirement(
) -> None:
    proposal = create_proposal()
    proposal["selection_status"] = "unsupported"
    proposal["unresolved_requirements"] = [
        {
            "description": (
                "Die Lieferung muss an einem Dienstag "
                "erfolgen."
            ),
            "resolution": "unsupported",
            "candidate_component_ids": [],
            "reason": (
                "Für feste Liefertage ist kein "
                "kompatibler Baustein verfügbar."
            ),
            "sources": ["raw_input"],
        }
    ]

    updated, excluded = (
        exclude_unsupported_requirements(
            proposal,
            confirmation_source=(
                "user_review_confirmation"
            ),
        )
    )

    assert updated["selection_status"] == "complete"
    assert updated["unresolved_requirements"] == []

    assert excluded == [
        {
            "description": (
                "Die Lieferung muss an einem Dienstag "
                "erfolgen."
            ),
            "reason": (
                "Für feste Liefertage ist kein "
                "kompatibler Baustein verfügbar."
            ),
            "sources": ["raw_input"],
            "confirmation_source": (
                "user_review_confirmation"
            ),
        }
    ]

    assert proposal["selection_status"] == "unsupported"
    assert len(proposal["unresolved_requirements"]) == 1


def test_rejects_exclusion_without_confirmation_source(
) -> None:
    proposal = create_proposal()
    proposal["selection_status"] = "unsupported"
    proposal["unresolved_requirements"] = [
        {
            "description": "Fester Liefertag",
            "resolution": "unsupported",
            "candidate_component_ids": [],
            "reason": "Kein kompatibler Baustein.",
            "sources": ["raw_input"],
        }
    ]

    with pytest.raises(
        ValueError,
        match="Bestätigungsquelle",
    ):
        exclude_unsupported_requirements(
            proposal,
            confirmation_source="",
        )


def test_copies_confirmed_exclusions_into_specification(
) -> None:
    excluded_requirements = [
        {
            "description": (
                "Die Lieferung muss an einem Dienstag "
                "erfolgen."
            ),
            "reason": (
                "Für feste Liefertage ist kein "
                "kompatibler Baustein verfügbar."
            ),
            "sources": ["raw_input"],
            "confirmation_source": (
                "user_review_confirmation"
            ),
        }
    ]

    specification = build_specification_from_proposal(
        create_proposal(),
        create_products(),
        {},
        excluded_requirements=excluded_requirements,
    )

    assert specification[
        "excluded_requirements"
    ] == excluded_requirements

    specification["excluded_requirements"][0][
        "description"
    ] = "Geändert"

    assert excluded_requirements[0]["description"] == (
        "Die Lieferung muss an einem Dienstag erfolgen."
    )
