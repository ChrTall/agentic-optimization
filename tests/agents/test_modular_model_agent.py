import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentic_optimization.agents.modular_model_agent import (
    create_modular_model_agent,
    execute_modular_model_draft,
    update_modular_model_draft,
    create_preliminary_component_selection_agent,
)


def test_selects_preliminary_components_without_products(
) -> None:
    response = {
        "core_id": "single_period_newsvendor",
        "selection_status": "complete",
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "reason": "Verkaufserlöse berücksichtigen.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "purchase_cost",
                "reason": "Einkaufskosten berücksichtigen.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(response)
    )

    agent = (
        create_preliminary_component_selection_agent(
            llm
        )
    )

    proposal = agent(
        core_id="single_period_newsvendor",
        requirements={
            "objective": "Gewinn maximieren",
        },
        global_facts={},
        evidence=[
            {
                "source_id": "raw_input",
                "content": (
                    "Der erwartete Gewinn soll maximiert "
                    "werden."
                ),
            }
        ],
    )

    assert proposal == response
    assert llm.invoke.call_count == 1

    messages = llm.invoke.call_args.args[0]
    user_message = json.loads(
        messages[1].content
    )

    assert user_message["products"] == []


def test_falls_back_to_safe_newsvendor_product_columns(
) -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content="kein gültiges JSON"
    )

    agent = (
        create_preliminary_component_selection_agent(
            llm
        )
    )

    proposal = agent(
        core_id="single_period_newsvendor",
        requirements={
            "objective": "maximize_profit",
            "product_scope": "multiple_products",
        },
        global_facts={},
        evidence=[
            {
                "source_id": "raw_input",
                "content": (
                    "Der erwartete Gewinn aus Verkaufserlösen "
                    "abzüglich Einkaufskosten soll maximiert "
                    "werden."
                ),
            }
        ],
    )

    assert proposal["selection_status"] == "complete"
    assert proposal["objective_sense"] == "maximize"
    assert [
        component["component_id"]
        for component in proposal["objective_components"]
    ] == [
        "sales_revenue",
        "purchase_cost",
    ]
    assert proposal["constraints"] == []
    assert llm.invoke.call_count == 2


def test_prepares_modular_q_r_model_draft() -> None:
    response = {
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
        "constraints": [],
        "unresolved_requirements": [],
    }

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content=json.dumps(
                {
                    "products": [],
                    "global_facts": {},
                }
            )
        ),
        AIMessage(
            content=json.dumps(response)
        ),
        AIMessage(
            content=json.dumps(
                {
                    "uncovered_requirements": [],
                }
            )
        ),
    ]

    agent = create_modular_model_agent(llm)

    draft = agent(
        core_id="continuous_review_q_r",
        requirements={
            "objective": "Jährliche Kosten minimieren",
        },
        products=[
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
        global_facts={},
        evidence=[
            {
                "source_id": "source_1",
                "content": (
                    "Bestellkosten, Lagerkosten und "
                    "Fehlmengenkosten sollen berücksichtigt "
                    "werden."
                ),
            }
        ],
        preliminary_proposal=response,
    )

    assert draft["proposal"] == response
    assert draft["specification"]["core_id"] == (
        "continuous_review_q_r"
    )
    assert draft["specification"]["objective"] == {
        "sense": "minimize",
        "component_ids": [
            "annual_ordering_cost",
            "annual_holding_cost",
            "annual_shortage_cost",
        ],
    }

    assert draft["validation"]["is_valid"] is False
    assert draft["validation"]["errors"] == []

    assert {
        missing_input["field"]["name"]
        for missing_input
        in draft["validation"]["missing_inputs"]
    } == {
        "ordering_cost",
        "holding_cost",
        "shortage_cost",
    }

    with pytest.raises(
        ValueError,
        match="nicht vollständig",
    ):
        execute_modular_model_draft(
            draft,
            confirmed=True,
        )

    values = {
        "ordering_cost": 50.0,
        "holding_cost": 2.0,
        "shortage_cost": 20.0,
    }

    for missing_input in draft[
        "validation"
    ]["missing_inputs"]:
        field_name = missing_input["field"]["name"]

        draft = update_modular_model_draft(
            draft,
            missing_input,
            values[field_name],
        )

    assert draft["validation"]["is_valid"] is True
    assert draft["validation"]["errors"] == []
    assert draft["validation"]["missing_inputs"] == []

    with pytest.raises(
        ValueError,
        match="bestätigt",
    ):
        execute_modular_model_draft(
            draft,
            confirmed=False,
        )

    result = execute_modular_model_draft(
        draft,
        confirmed=True,
    )

    assert result["status"] == "optimal"
    assert result["solution_method"] == "numerical"

    assert llm.invoke.call_count == 3

    selection_messages = (
        llm.invoke.call_args_list[1].args[0]
    )
    selection_payload = json.loads(
        selection_messages[1].content
    )

    assert selection_payload[
        "locked_objective_selection"
    ] == {
        "objective_sense": "minimize",
        "objective_component_ids": [
            "annual_ordering_cost",
            "annual_holding_cost",
            "annual_shortage_cost",
        ],
    }


def test_uses_extracted_facts_in_model_specification() -> None:
    fact_response = {
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
                    "selling_price": 12.0,
                    "purchase_cost": 5.0,
                },
            }
        ],
        "global_facts": {},
    }

    component_response = {
        "core_id": "single_period_newsvendor",
        "selection_status": "complete",
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "reason": "Verkaufspreis wurde genannt.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "purchase_cost",
                "reason": "Einkaufspreis wurde genannt.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content=json.dumps(fact_response)
        ),
        AIMessage(
            content=json.dumps(component_response)
        ),
        AIMessage(
            content=json.dumps(
                {
                    "uncovered_requirements": [],
                }
            )
        ),
    ]

    agent = create_modular_model_agent(llm)

    draft = agent(
        core_id="single_period_newsvendor",
        requirements={
            "objective": "maximize_profit",
        },
        products=[
            {
                "product_id": "product_1",
                "forecast": {},
                "facts": {},
            }
        ],
        global_facts={},
        evidence=[
            {
                "source_id": "raw_input",
                "content": (
                    "Die erwartete Nachfrage beträgt 100 "
                    "Einheiten für 30 Tage. Der Verkaufspreis "
                    "beträgt 12 Euro und der Einkaufspreis "
                    "5 Euro. Die Nachfragequantile sind "
                    "70, 100 und 140 Einheiten."
                ),
            }
        ],
    )

    product = draft["specification"]["products"][0]

    assert product["forecast"] == (
        fact_response["products"][0]["forecast"]
    )
    assert product["facts"] == {
        "selling_price": 12.0,
        "purchase_cost": 5.0,
    }
    assert draft["validation"]["is_valid"] is True
    assert draft["validation"]["errors"] == []
    assert draft["validation"]["missing_inputs"] == []
    assert llm.invoke.call_count == 3


def test_returns_unsupported_proposal_without_specification() -> None:
    response = {
        "core_id": "eoq",
        "selection_status": "unsupported",
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
        "unresolved_requirements": [
            {
                "description": "Mengenrabatt",
                "resolution": "unsupported",
                "candidate_component_ids": [],
                "reason": (
                    "Für Mengenrabatte ist kein "
                    "kompatibler Baustein verfügbar."
                ),
                "sources": ["source_1"],
            }
        ],
    }

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content=json.dumps(
                {
                    "products": [],
                    "global_facts": {},
                }
            )
        ),
        AIMessage(content=json.dumps(response)),
        AIMessage(
            content=json.dumps(
                {
                    "uncovered_requirements": [],
                }
            )
        ),
    ]

    agent = create_modular_model_agent(llm)

    draft = agent(
        core_id="eoq",
        requirements={
            "objective": "Kosten minimieren",
        },
        products=[
            {
                "product_id": "product_1",
                "forecast": {},
                "facts": {},
            }
        ],
        global_facts={},
        evidence=[
            {
                "source_id": "source_1",
                "content": (
                    "Ab 1.000 Stück gilt ein Mengenrabatt."
                ),
            }
        ],
    )

    assert draft["proposal"] == response
    assert draft["products"] == [
        {
            "product_id": "product_1",
            "forecast": {},
            "facts": {},
        }
    ]
    assert draft["global_facts"] == {}
    assert "specification" not in draft
    assert "validation" not in draft
    assert llm.invoke.call_count == 3


def test_coverage_audit_detects_overlooked_requirement(
) -> None:
    component_response = {
        "core_id": "eoq",
        "selection_status": "complete",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten wurden genannt.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten wurden genannt.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }

    uncovered_requirement = {
        "description": (
            "Jede Lieferung muss an einem "
            "Dienstag eintreffen."
        ),
        "resolution": "unsupported",
        "candidate_component_ids": [],
        "reason": (
            "Es ist kein Baustein für einen "
            "bestimmten Liefertag vorhanden."
        ),
        "sources": ["raw_input"],
    }

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content=json.dumps(
                {
                    "products": [],
                    "global_facts": {},
                }
            )
        ),
        AIMessage(
            content=json.dumps(component_response)
        ),
        AIMessage(
            content=json.dumps(
                {
                    "uncovered_requirements": [
                        uncovered_requirement
                    ],
                }
            )
        ),
    ]

    agent = create_modular_model_agent(llm)

    draft = agent(
        core_id="eoq",
        requirements={
            "objective": "Gesamtkosten minimieren",
        },
        products=[
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
        ],
        global_facts={},
        evidence=[
            {
                "source_id": "raw_input",
                "content": (
                    "Bestell- und Lagerhaltungskosten sollen "
                    "minimiert werden. Jede Lieferung muss "
                    "an einem Dienstag eintreffen."
                ),
            }
        ],
    )

    assert draft["proposal"]["selection_status"] == (
        "unsupported"
    )
    assert draft["proposal"]["constraints"] == []
    assert draft["proposal"][
        "unresolved_requirements"
    ] == [
        uncovered_requirement
    ]
    assert "specification" not in draft
    assert "validation" not in draft
    assert llm.invoke.call_count == 3
