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
from agentic_optimization.ui.modular_input_builder import (
    build_dynamic_product_table_schema,
)


def test_selects_preliminary_components_without_products(
) -> None:
    llm = Mock()

    agent = (
        create_preliminary_component_selection_agent(
            llm
        )
    )

    proposal = agent(
        core_id="single_period_newsvendor",
        requirements={
            "objective": "maximize_profit",
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

    assert proposal["selection_status"] == "complete"
    assert proposal["objective_sense"] == "maximize"
    assert [
        item["component_id"] for item in proposal["objective_components"]
    ] == [
        "sales_revenue", "purchase_cost"
    ]
    assert all(
        item["sources"] == ["core_definition"]
        for item in proposal["objective_components"]
    )
    assert proposal["constraints"] == []
    assert proposal["unresolved_requirements"] == []
    llm.invoke.assert_not_called()


def test_uses_registered_newsvendor_product_columns_without_llm(
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
    llm.invoke.assert_not_called()


@pytest.mark.parametrize(
    ("core_id", "objective", "expected_components", "expected_facts"),
    [
        (
            "eoq", "minimize_cost",
            ["annual_ordering_cost", "annual_holding_cost"],
            {"ordering_cost", "holding_cost"},
        ),
        (
            "continuous_review_q_r", "minimize_cost",
            [
                "annual_ordering_cost",
                "annual_holding_cost",
                "annual_shortage_cost",
            ],
            {"lead_time_days", "ordering_cost", "holding_cost", "shortage_cost"},
        ),
    ],
)
def test_preliminary_form_uses_only_registered_core_requirements(
    core_id: str,
    objective: str,
    expected_components: list[str],
    expected_facts: set[str],
) -> None:
    llm = Mock()
    proposal = create_preliminary_component_selection_agent(llm)(
        core_id=core_id,
        requirements={"objective": objective},
        global_facts={},
        evidence=[{"source_id": "raw_input", "content": "Beliebige Beschreibung"}],
    )

    assert [
        item["component_id"] for item in proposal["objective_components"]
    ] == expected_components
    schema = build_dynamic_product_table_schema(core_id, proposal)
    assert {field["name"] for field in schema["fact_fields"]} == expected_facts
    assert proposal["unresolved_requirements"] == []
    llm.invoke.assert_not_called()


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

    assert selection_payload["locked_objective_selection"] is None


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


def test_preliminary_revenue_only_does_not_block_purchase_cost_input() -> None:
    preliminary = {
        "core_id": "single_period_newsvendor",
        "selection_status": "complete",
        "objective_sense": "maximize",
        "objective_components": [{
            "component_id": "sales_revenue",
            "reason": "Erlöse berücksichtigen.",
            "sources": ["raw_input"],
        }],
        "constraints": [],
        "unresolved_requirements": [],
    }
    final_decision = {
        "objective_sense": "maximize",
        "objective_components": list(preliminary["objective_components"]),
        "constraints": [],
        "unresolved_requirements": [],
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({
            "products": [], "global_facts": {},
        })),
        AIMessage(content=json.dumps(final_decision)),
    ]

    draft = create_modular_model_agent(
        llm, enable_coverage_audit=False
    )(
        core_id="single_period_newsvendor",
        requirements={"objective": "maximize_profit"},
        products=[{
            "product_id": "product_1", "forecast": {}, "facts": {},
        }],
        global_facts={},
        evidence=[{
            "source_id": "raw_input",
            "content": "Den erwarteten Gewinn maximieren.",
        }],
        preliminary_proposal=preliminary,
    )

    assert draft["proposal"]["selection_status"] == "complete"
    assert {item["component_id"] for item in draft["proposal"][
        "objective_components"
    ]} == {"sales_revenue", "purchase_cost"}
    assert {item["field"]["name"] for item in draft["validation"][
        "missing_inputs"
    ]} >= {"selling_price", "purchase_cost"}
    selection_payload = json.loads(
        llm.invoke.call_args_list[1].args[0][1].content
    )
    assert selection_payload["locked_objective_selection"] is None


def test_qr_missing_forecast_is_input_not_model_clarification() -> None:
    selection = {
        "objective_components": [],
        "constraints": [{
            "component_id": "minimum_reorder_point",
            "product_id": "product_1",
            "parameters": {},
            "reason": "Meldebestand bestimmen.",
            "sources": ["raw_input"],
        }],
        "unresolved_requirements": [],
    }
    audit = {
        "rejected_constraints": [{
            "index": 0,
            "reason": "Keine zusätzliche Untergrenze gewünscht.",
        }],
        "input_only_requirements": [],
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({
            "products": [], "global_facts": {},
        })),
        AIMessage(content=json.dumps(selection)),
        AIMessage(content=json.dumps(audit)),
    ]

    draft = create_modular_model_agent(
        llm, enable_coverage_audit=False
    )(
        core_id="continuous_review_q_r",
        requirements={"objective": "minimize_cost"},
        products=[{
            "product_id": "product_1",
            "forecast": {},
            "facts": {"lead_time_days": 5},
        }],
        global_facts={},
        evidence=[{
            "source_id": "raw_input",
            "content": (
                "Der Lagerbestand wird laufend überwacht. "
                "Wir möchten Bestellmenge und Meldebestand bestimmen. "
                "Die Nachfrage ist unsicher."
            ),
        }],
    )

    assert draft["proposal"]["selection_status"] == "complete"
    assert draft["proposal"]["constraints"] == []
    assert draft["proposal"]["unresolved_requirements"] == []
    assert {item["field"]["name"] for item in draft[
        "validation"
    ]["missing_inputs"]} >= {
        "expected_value", "quantiles", "horizon_days"
    }


def test_complete_eoq_inputs_are_not_blocked_by_unquoted_missing_claims() -> None:
    mistaken_selection = {
        "objective_sense": "minimize",
        "objective_components": [],
        "constraints": [],
        "unresolved_requirements": [
            {
                "description": description,
                "resolution": "unsupported",
                "candidate_component_ids": [],
                "reason": "Ein Eingabewert sei nicht vorhanden.",
                "sources": ["raw_input"],
            }
            for description in (
                "Erwartete Nachfrage fehlt.",
                "Prognosezeitraum fehlt.",
                "Bestellkosten fehlen.",
                "Lagerhaltungskosten fehlen.",
            )
        ],
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"products": [], "global_facts": {}})),
        AIMessage(content=json.dumps(mistaken_selection)),
        AIMessage(content=json.dumps({
            **mistaken_selection,
            "unresolved_requirements": [],
        })),
    ]

    draft = create_modular_model_agent(
        llm, enable_coverage_audit=False
    )(
        core_id="eoq",
        requirements={"objective": "minimize_cost"},
        products=[{
            "product_id": "product_1",
            "forecast": {"expected_value": 3500, "horizon_days": 365},
            "facts": {"ordering_cost": 120, "holding_cost": 12},
        }],
        global_facts={},
        evidence=[{
            "source_id": "raw_input",
            "content": (
                "Für einen Artikel planen wir eine Bestellung und möchten "
                "die jährlichen Bestell- und Lagerkosten minimieren. "
                "Wie oft bestellt wird, müssen wir noch klären."
            ),
        }],
    )

    assert draft["proposal"]["selection_status"] == "complete"
    assert draft["proposal"]["unresolved_requirements"] == []
    assert draft["validation"]["is_valid"] is True
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
    reported_response = json.loads(json.dumps(response))
    reported_response["additional_requirements"] = reported_response.pop(
        "unresolved_requirements"
    )
    reported_response["additional_requirements"][0]["evidence_excerpt"] = (
        "Ab 1.000 Stück gilt ein Mengenrabatt."
    )

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
        AIMessage(content=json.dumps(reported_response)),
        AIMessage(content=json.dumps({
            "rejected_constraints": [],
            "input_only_requirements": [],
        })),
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
    assert llm.invoke.call_count == 4


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
        AIMessage(content=json.dumps({
            "rejected_constraints": [],
            "input_only_requirements": [],
        })),
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
    assert llm.invoke.call_count == 4


@pytest.mark.parametrize("misclassification_stage", [
    "selection", "coverage",
])
def test_missing_newsvendor_forecast_is_input_after_all_audits(
    misclassification_stage: str,
) -> None:
    missing_forecast = [
        {
            "description": description,
            "resolution": "unsupported",
            "candidate_component_ids": [],
            "reason": description,
            "sources": ["raw_input"],
        }
        for description in (
            "Erwartete Nachfrage ist nicht angegeben.",
            "Nachfragequantile fehlen.",
        )
    ]
    selection = {
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": component_id,
                "reason": "Für den erwarteten Gewinn erforderlich.",
                "sources": ["raw_input"],
            }
            for component_id in ("sales_revenue", "purchase_cost")
        ],
        "constraints": [],
        "unresolved_requirements": (
            missing_forecast if misclassification_stage == "selection" else []
        ),
    }
    input_audit = {
        "rejected_constraints": [],
        "input_only_requirements": [
            {"index": 0, "field_names": ["expected_value"]},
            {"index": 1, "field_names": ["quantiles"]},
        ],
    }
    responses = [
        {"products": [], "global_facts": {}},
        selection,
    ]
    if misclassification_stage == "selection":
        responses.append({
            **selection,
            "unresolved_requirements": [],
        })
    responses.append({
        "uncovered_requirements": (
            missing_forecast if misclassification_stage == "coverage" else []
        ),
    })
    if misclassification_stage == "coverage":
        responses.append(input_audit)
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(response)) for response in responses
    ]

    draft = create_modular_model_agent(llm)(
        core_id="single_period_newsvendor",
        requirements={"objective": "maximize_profit"},
        products=[{
            "product_id": "product_1",
            "forecast": {},
            "facts": {"selling_price": 20.0, "purchase_cost": 10.0},
        }],
        global_facts={},
        evidence=[{
            "source_id": "raw_input",
            "content": (
                "Wir bestellen einmal vor der Veranstaltung und können "
                "nicht nachbestellen. Die Nachfrage ist unsicher. "
                "Wir möchten den erwarteten Gewinn maximieren."
            ),
        }],
    )

    assert draft["proposal"]["selection_status"] == "complete"
    assert draft["proposal"]["unresolved_requirements"] == []
    assert {item["field"]["name"] for item in draft["validation"][
        "missing_inputs"
    ]} >= {"expected_value", "quantiles", "horizon_days"}
    assert llm.invoke.call_count == 4
