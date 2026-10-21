import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentic_optimization.agents.component_selection_agent import (
    create_component_selection_agent,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)


def create_mock_llm(response: dict) -> Mock:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(response)
    )

    return llm


def create_response() -> dict:
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


def call_agent(
    llm: Mock,
    evidence_content: str | None = None,
    locked_objective_proposal: dict | None = None,
    previous_proposal: dict | None = None,
    critic_feedback: list[dict] | None = None,
):
    agent = create_component_selection_agent(llm)

    return agent(
        core_id="single_period_newsvendor",
        requirements={
            "objective": "Gewinn maximieren",
        },
        products=[
            {
                "product_id": "product_1",
                "facts": {
                    "selling_price": 12.0,
                    "purchase_cost": 5.0,
                },
            }
        ],
        global_facts={},
        evidence=[
            {
                "source_id": "source_1",
                "content": (
                    evidence_content
                    if evidence_content is not None
                    else (
                        "Der Einkaufspreis beträgt 5 Euro. "
                        "Die Mindestbestellmenge beträgt "
                        "50 Stück. Unverkaufte Ware muss "
                        "kostenpflichtig entsorgt werden. "
                        "Bestellungen sind nur in "
                        "vollständigen Verpackungseinheiten "
                        "zu je 60 Stück möglich."
                    )
                ),
            }
        ],
        locked_objective_proposal=(
            locked_objective_proposal
        ),
        previous_proposal=previous_proposal,
        critic_feedback=critic_feedback,
    )


def test_selects_registered_components() -> None:
    response = create_response()
    llm = create_mock_llm(response)

    result = call_agent(llm)

    assert result == response
    llm.invoke.assert_called_once()


def test_accepts_semantic_decision_without_technical_fields(
) -> None:
    decision = create_response()
    del decision["core_id"]
    del decision["selection_status"]

    llm = create_mock_llm(decision)

    result = call_agent(llm)

    assert result["core_id"] == (
        "single_period_newsvendor"
    )
    assert result["selection_status"] == "complete"
    assert result["objective_sense"] == "maximize"
    assert result["objective_components"] == (
        decision["objective_components"]
    )
    assert result["constraints"] == (
        decision["constraints"]
    )
    assert result["unresolved_requirements"] == []
    assert llm.invoke.call_count == 1


def test_repairs_only_missing_unresolved_requirements_list() -> None:
    decision = create_response()
    decision["unresolved_requirements"] = None
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(decision)),
        AIMessage(content=json.dumps({"unresolved_requirements": []})),
    ]

    result = call_agent(llm)

    assert result["unresolved_requirements"] == []
    assert result["objective_components"] == decision["objective_components"]
    assert result["constraints"] == decision["constraints"]
    assert llm.invoke.call_count == 2


def test_repair_preserves_reported_unsupported_requirement() -> None:
    decision = create_response()
    decision.pop("unresolved_requirements")
    requirement = {
        "description": "Lieferung nur dienstags",
        "resolution": "unsupported",
        "candidate_component_ids": [],
        "reason": "Ein bestimmter Liefertag ist nicht abbildbar.",
        "sources": ["source_1"],
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(decision)),
        AIMessage(content=json.dumps({
            "unresolved_requirements": [requirement],
        })),
    ]

    result = call_agent(llm)

    assert result["selection_status"] == "unsupported"
    assert result["unresolved_requirements"] == [requirement]


def test_repairs_redundant_eoq_questions_about_mandatory_components() -> None:
    decision = {
        "objective_components": [],
        "constraints": [],
        "unresolved_requirements": [
            {
                "description": description,
                "resolution": "clarification",
                "candidate_component_ids": [
                    "annual_ordering_cost", "annual_holding_cost"
                ],
                "reason": "Welche Kosten sind relevant?",
                "sources": ["raw_input"],
            }
            for description in (
                "Bestellkosten berücksichtigen",
                "Lagerhaltungskosten berücksichtigen",
            )
        ],
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(decision)),
        AIMessage(content=json.dumps({"unresolved_requirements": []})),
    ]

    result = create_component_selection_agent(llm)(
        core_id="eoq",
        requirements={"objective": "Jährliche Kosten minimieren"},
        products=[],
        global_facts={},
        evidence=[{
            "source_id": "raw_input",
            "content": (
                "Bestell- und Lagerhaltungskosten sollen "
                "minimiert werden."
            ),
        }],
    )

    assert result["selection_status"] == "complete"
    assert result["unresolved_requirements"] == []
    assert {
        item["component_id"] for item in result["objective_components"]
    } == {"annual_ordering_cost", "annual_holding_cost"}
    assert llm.invoke.call_count == 2
    initial_messages = llm.invoke.call_args_list[0].args[0]
    initial_task = json.loads(initial_messages[1].content)
    assert initial_task["already_selected_by_system_component_ids"] == [
        "annual_holding_cost", "annual_ordering_cost"
    ]
    repair_messages = llm.invoke.call_args_list[1].args[0]
    repair_task = json.loads(repair_messages[1].content)
    assert repair_task["forbidden_clarification_candidate_ids"] == [
        "annual_holding_cost", "annual_ordering_cost"
    ]
    assert [item["index"] for item in repair_task[
        "overlapping_questions"
    ]] == [0, 1]
    assert {item["component_id"] for item in repair_task[
        "selected_components"
    ]} == {"annual_ordering_cost", "annual_holding_cost"}
    assert not set(repair_task[
        "available_clarification_candidate_ids"
    ]).intersection(repair_task[
        "forbidden_clarification_candidate_ids"
    ])


def test_deduplication_preserves_other_unsupported_requirement() -> None:
    duplicated = {
        "description": "Bestellkosten berücksichtigen",
        "resolution": "clarification",
        "candidate_component_ids": ["annual_ordering_cost"],
        "reason": "Kostenart unklar.",
        "sources": ["raw_input"],
    }
    unsupported = {
        "description": "Lieferung nur dienstags",
        "resolution": "unsupported",
        "candidate_component_ids": [],
        "reason": "Kein passender Baustein.",
        "sources": ["raw_input"],
    }
    decision = {
        "objective_components": [],
        "constraints": [],
        "unresolved_requirements": [duplicated, unsupported],
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(decision)),
        AIMessage(content=json.dumps({
            "unresolved_requirements": [unsupported],
        })),
    ]

    result = create_component_selection_agent(llm)(
        core_id="eoq",
        requirements={"objective": "Kosten minimieren"},
        products=[],
        global_facts={},
        evidence=[{
            "source_id": "raw_input",
            "content": "Bestellkosten und Lieferung nur dienstags.",
        }],
    )

    assert result["selection_status"] == "unsupported"
    assert result["unresolved_requirements"] == [unsupported]
    assert llm.invoke.call_count == 2


def test_full_retry_receives_selected_ids_after_invalid_focused_repair() -> None:
    repeated_question = {
        "description": "Welche Kosten sollen berücksichtigt werden?",
        "resolution": "clarification",
        "candidate_component_ids": [
            "annual_ordering_cost", "annual_holding_cost",
        ],
        "reason": "Die Kostenarten seien noch offen.",
        "sources": ["raw_input"],
    }
    invalid = {
        "objective_components": [],
        "constraints": [],
        "unresolved_requirements": [repeated_question],
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(invalid)),
        AIMessage(content=json.dumps({
            "unresolved_requirements": [repeated_question],
        })),
        AIMessage(content=json.dumps({
            **invalid, "unresolved_requirements": [],
        })),
    ]

    result = create_component_selection_agent(llm)(
        core_id="eoq",
        requirements={"objective": "minimize_cost"},
        products=[], global_facts={},
        evidence=[{
            "source_id": "raw_input",
            "content": "Jährliche Bestell- und Lagerkosten minimieren.",
        }],
    )

    assert result["selection_status"] == "complete"
    assert llm.invoke.call_count == 3
    retry_messages = llm.invoke.call_args_list[2].args[0]
    feedback = json.loads(retry_messages[-1].content)
    assert feedback["forbidden_clarification_candidate_ids"] == [
        "annual_holding_cost", "annual_ordering_cost"
    ]


def test_nonmandatory_overlap_still_needs_focused_repair() -> None:
    decision = create_response()
    decision["unresolved_requirements"] = [{
        "description": "Ungeklärte Erlösart",
        "resolution": "clarification",
        "candidate_component_ids": ["sales_revenue"],
        "reason": "Erlösart noch offen.",
        "sources": ["source_1"],
    }]
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(decision)),
        AIMessage(content=json.dumps({"unresolved_requirements": []})),
    ]

    result = call_agent(llm)

    assert result["unresolved_requirements"] == []
    assert llm.invoke.call_count == 2


def test_profit_goal_collects_missing_purchase_price_in_form() -> None:
    response = {
        "objective_sense": "maximize",
        "objective_components": [{
            "component_id": "sales_revenue",
            "reason": "Erlöse berücksichtigen.",
            "sources": ["raw_input"],
        }],
        "constraints": [],
        "unresolved_requirements": [{
            "description": "Einkaufskosten fehlen.",
            "resolution": "clarification",
            "candidate_component_ids": ["purchase_cost"],
            "reason": "Ein Einkaufspreis wurde nicht angegeben.",
            "sources": ["raw_input"],
        }],
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(response)),
        AIMessage(content=json.dumps({"unresolved_requirements": []})),
    ]

    result = create_component_selection_agent(llm)(
        core_id="single_period_newsvendor",
        requirements={"objective": "maximize_profit"},
        products=[],
        global_facts={},
        evidence=[{
            "source_id": "raw_input",
            "content": "Den erwarteten Gewinn maximieren.",
        }],
    )

    assert result["selection_status"] == "complete"
    assert result["unresolved_requirements"] == []
    assert {item["component_id"] for item in result[
        "objective_components"
    ]} == {"sales_revenue", "purchase_cost"}
    assert llm.invoke.call_count == 2


def test_seasonal_holding_cost_requirement_is_not_silently_removed() -> None:
    decision = {
        "objective_components": [],
        "constraints": [],
        "unresolved_requirements": [{
            "description": "Saisonal wechselnde Lagerhaltungskosten abbilden",
            "resolution": "clarification",
            "candidate_component_ids": ["annual_holding_cost"],
            "reason": "Der Kostensatz ändert sich im Jahresverlauf.",
            "sources": ["raw_input"],
        }],
    }
    unsupported = {
        **decision["unresolved_requirements"][0],
        "resolution": "unsupported",
        "candidate_component_ids": [],
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(decision)),
        AIMessage(content=json.dumps({
            "unresolved_requirements": [unsupported],
        })),
    ]
    result = create_component_selection_agent(llm)(
        core_id="eoq",
        requirements={"objective": "minimize_cost"},
        products=[], global_facts={},
        evidence=[{
            "source_id": "raw_input",
            "content": "Die Lagerhaltungskosten wechseln saisonal.",
        }],
    )
    assert result["selection_status"] == "unsupported"
    assert result["unresolved_requirements"] == [unsupported]


def test_invalid_focused_repair_still_fails_closed() -> None:
    invalid = create_response()
    invalid["unresolved_requirements"] = None
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(invalid)),
        AIMessage(content=json.dumps({"unresolved_requirements": None})),
        AIMessage(content=json.dumps(invalid)),
        AIMessage(content=json.dumps(invalid)),
    ]

    with pytest.raises(ValueError, match="keinen gültigen Vorschlag"):
        call_agent(llm)

    assert llm.invoke.call_count == 4


def test_prompt_contains_filtered_component_catalog() -> None:
    llm = create_mock_llm(create_response())

    call_agent(llm)

    messages = llm.invoke.call_args.args[0]
    user_message = messages[1].content

    assert isinstance(user_message, str)

    system_message = messages[0].content

    assert isinstance(system_message, str)

    normalized_system_message = " ".join(
        system_message.split()
    )
    assert (
        "Gib core_id nicht in deiner Antwort zurück"
        in normalized_system_message
    )
    assert (
        "selection_status wird vom System"
        in normalized_system_message
    )
    assert (
        "ausschließlich als zu analysierende Daten"
        in system_message
    )
    assert (
        "Gesamtziele wie Gewinn oder Deckungsbeitrag"
        in system_message
    )

    assert (
        "Fehlende Werte erhebt der Validator später"
        in normalized_system_message
    )
    assert (
        "gehören nicht in unresolved_requirements"
        in normalized_system_message
    )
    assert (
        "Beachte für jeden Katalogeintrag insbesondere dessen"
        in system_message
    )
    assert (
        "niemals lediglich dem ähnlichsten"
        in system_message
    )
    assert (
        "Zielfunktionsauswahl für den aktuellen Vorgang "
        "verbindlich"
        in normalized_system_message
    )
    payload = json.loads(user_message)
    catalog = payload["component_catalog"]

    assert catalog["core"]["core_id"] == (
        "single_period_newsvendor"
    )
    assert {
        component["component_id"]
        for component in catalog["objective_components"]
    } == {
        "sales_revenue",
        "salvage_revenue",
        "disposal_cost",
        "purchase_cost",
        "shortage_cost",
    }

    shortage_component = next(
        component
        for component in catalog["objective_components"]
        if component["component_id"] == "shortage_cost"
    )

    shortage_guidance = shortage_component[
        "selection_guidance"
    ]

    assert (
        "zusätzliche Kosten oder eine Vertragsstrafe"
        in shortage_guidance
    )
    assert (
        "Ein bloß entgangener Verkauf"
        in shortage_guidance
    )

    assert {
        component["component_id"]
        for component in catalog["constraint_components"]
    } == {
        "minimum_order_quantity",
        "maximum_order_quantity",
        "order_multiple",
        "budget_limit",
        "storage_capacity",
        "minimum_cycle_service_level",
    }

    minimum_quantity_component = next(
        component
        for component in catalog["constraint_components"]
        if component["component_id"]
        == "minimum_order_quantity"
    )

    assert (
        "mengenmäßige Untergrenze"
        in minimum_quantity_component[
            "selection_guidance"
        ]
    )
    assert (
        "Liefertermine"
        in minimum_quantity_component[
            "selection_guidance"
        ]
    )

    service_component = next(
        component
        for component in catalog["constraint_components"]
        if component["component_id"]
        == "minimum_cycle_service_level"
    )

    assert (
        "Wahrscheinlichkeit"
        in service_component["selection_guidance"]
    )

    assert "builder" not in user_message
    assert (
        "Pflichtbausteine des Grundmodells und des angegebenen "
        "Optimierungsziels werden vom "
        "System ergänzt"
        in normalized_system_message
    )


def test_accepts_json_code_fence() -> None:
    response = create_response()
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=(
            "```json\n"
            f"{json.dumps(response)}\n"
            "```"
        )
    )

    result = call_agent(llm)

    assert result == response


def test_classifies_invalid_json_as_format_error() -> None:
    corrected_response = create_response()

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content="kein gültiges JSON"),
        AIMessage(
            content=json.dumps(corrected_response)
        ),
    ]

    result = call_agent(llm)

    assert result == corrected_response
    assert llm.invoke.call_count == 2

    correction_messages = (
        llm.invoke.call_args_list[1].args[0]
    )
    feedback = json.loads(
        correction_messages[-1].content
    )

    assert feedback["error_category"] == (
        "response_format"
    )
    assert feedback["required_action"] == (
        "return_valid_json"
    )


def test_rejects_unknown_source() -> None:
    response = create_response()
    response["objective_components"][0]["sources"] = [
        "invented_source"
    ]
    llm = create_mock_llm(response)

    with pytest.raises(
        ValueError,
        match="invented_source",
    ):
        call_agent(llm)


def test_corrects_unknown_source_with_structured_feedback(
) -> None:
    invalid_response = create_response()
    invalid_response["objective_components"][0][
        "sources"
    ] = ["invented_source"]

    corrected_response = create_response()

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content=json.dumps(invalid_response)
        ),
        AIMessage(
            content=json.dumps(corrected_response)
        ),
    ]

    result = call_agent(llm)

    assert result == corrected_response
    assert llm.invoke.call_count == 2

    correction_messages = (
        llm.invoke.call_args_list[1].args[0]
    )
    feedback = json.loads(
        correction_messages[-1].content
    )

    assert feedback["error_category"] == (
        "source_reference"
    )
    assert feedback["required_action"] == (
        "use_allowed_source_ids"
    )
    assert feedback["allowed_source_ids"] == [
        "source_1"
    ]


def test_normalizes_numbered_source_placeholder_when_unambiguous(
) -> None:
    response = create_response()
    llm = create_mock_llm(response)
    agent = create_component_selection_agent(llm)

    result = agent(
        core_id="single_period_newsvendor",
        requirements={
            "objective": "maximize_profit",
        },
        products=[],
        global_facts={},
        evidence=[
            {
                "source_id": "raw_input",
                "content": (
                    "Der erwartete Gewinn soll maximiert "
                    "werden. Eine Mindestbestellmenge gilt."
                ),
            }
        ],
    )

    selected_items = [
        *result["objective_components"],
        *result["constraints"],
    ]

    assert all(
        item["sources"] == ["raw_input"]
        for item in selected_items
    )
    assert llm.invoke.call_count == 1


def test_discards_constraint_number_without_evidence() -> None:
    response = create_response()
    llm = create_mock_llm(response)
    agent = create_component_selection_agent(llm)

    result = agent(
        core_id="single_period_newsvendor",
        requirements={
            "objective": "maximize_profit",
        },
        products=[],
        global_facts={},
        evidence=[
            {
                "source_id": "raw_input",
                "content": (
                    "Der Lieferant verlangt eine "
                    "Mindestbestellmenge."
                ),
            }
        ],
    )

    assert result["constraints"][0]["component_id"] == (
        "minimum_order_quantity"
    )
    assert result["constraints"][0]["parameters"] == {}


def test_keeps_constraint_number_with_evidence() -> None:
    response = create_response()
    llm = create_mock_llm(response)

    result = call_agent(llm)

    assert result["constraints"][0]["parameters"] == {
        "quantity": 50.0,
    }


def test_rejects_unknown_component() -> None:
    response = create_response()
    response["objective_components"][0][
        "component_id"
    ] = "invented_component"
    llm = create_mock_llm(response)

    with pytest.raises(
        ValueError,
        match="invented_component",
    ):
        call_agent(llm)


def test_rejects_different_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        CORE_REGISTRY,
        "other_core",
        {},
    )

    response = create_response()
    response["core_id"] = "other_core"
    llm = create_mock_llm(response)

    with pytest.raises(
        ValueError,
        match="anderes Grundmodell",
    ):
        call_agent(llm)


def test_builds_eoq_selection_request() -> None:
    response = {
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
    llm = create_mock_llm(response)
    agent = create_component_selection_agent(llm)

    result = agent(
        core_id="eoq",
        requirements={
            "objective": "Gesamtkosten minimieren",
        },
        products=[
            {
                "product_id": "product_1",
                "facts": {
                    "ordering_cost": 120.0,
                    "holding_cost": 12.0,
                },
            }
        ],
        global_facts={},
        evidence=[
            {
                "source_id": "source_1",
                "content": (
                    "Bestellkosten und Lagerkosten "
                    "sollen berücksichtigt werden."
                ),
            }
        ],
    )

    messages = llm.invoke.call_args.args[0]
    user_message = messages[1].content

    assert isinstance(user_message, str)

    payload = json.loads(user_message)
    catalog = payload["component_catalog"]

    assert {
        component["component_id"]
        for component in catalog["objective_components"]
    } == {
        "annual_ordering_cost",
        "annual_holding_cost",
    }
    assert {
        component["component_id"]
        for component in catalog["constraint_components"]
    } == {
        "minimum_order_quantity",
        "maximum_order_quantity",
        "order_multiple",
    }
    assert result == response


def test_adds_required_eoq_components_without_llm_echo(
) -> None:
    decision = {
        "objective_sense": "minimize",
        "objective_components": [],
        "constraints": [],
        "unresolved_requirements": [],
    }

    llm = create_mock_llm(decision)
    agent = create_component_selection_agent(llm)

    result = agent(
        core_id="eoq",
        requirements={
            "objective": "Gesamtkosten minimieren",
        },
        products=[],
        global_facts={},
        evidence=[
            {
                "source_id": "raw_input",
                "content": (
                    "Bestell- und Lagerhaltungskosten "
                    "sollen minimiert werden."
                ),
            }
        ],
    )

    assert [
        component["component_id"]
        for component in result[
            "objective_components"
        ]
    ] == [
        "annual_ordering_cost",
        "annual_holding_cost",
    ]
    assert all(
        component["sources"] == ["core_definition"]
        for component in result[
            "objective_components"
        ]
    )
    assert llm.invoke.call_count == 1


def test_eoq_selection_cannot_reverse_fixed_cost_objective() -> None:
    llm = create_mock_llm({
        "objective_sense": "maximize",
        "objective_components": [],
        "constraints": [],
        "unresolved_requirements": [],
    })

    result = create_component_selection_agent(llm)(
        core_id="eoq",
        requirements={"objective": "Gesamtkosten minimieren"},
        products=[],
        global_facts={},
        evidence=[{
            "source_id": "raw_input",
            "content": "Bestell- und Lagerkosten minimieren.",
        }],
    )

    assert result["objective_sense"] == "minimize"
    assert llm.invoke.call_count == 1


def test_builds_q_r_selection_request() -> None:
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

    llm = create_mock_llm(response)
    agent = create_component_selection_agent(llm)

    result = agent(
        core_id="continuous_review_q_r",
        requirements={
            "objective": "Jährliche Kosten minimieren",
        },
        products=[
            {
                "product_id": "product_1",
                "facts": {
                    "lead_time_days": 10,
                    "ordering_cost": 50.0,
                    "holding_cost": 2.0,
                    "shortage_cost": 20.0,
                },
            }
        ],
        global_facts={},
        evidence=[
            {
                "source_id": "source_1",
                "content": (
                    "Bestell-, Lager- und Fehlmengenkosten "
                    "sollen berücksichtigt werden. Der "
                    "Meldebestand muss mindestens 50 betragen."
                ),
            }
        ],
    )

    messages = llm.invoke.call_args.args[0]
    user_message = messages[1].content

    assert isinstance(user_message, str)

    payload = json.loads(user_message)
    catalog = payload["component_catalog"]

    assert {
        component["component_id"]
        for component in catalog["objective_components"]
    } == {
        "annual_ordering_cost",
        "annual_holding_cost",
        "annual_shortage_cost",
    }
    assert {
        component["component_id"]
        for component in catalog["constraint_components"]
    } == {
        "minimum_order_quantity",
        "maximum_order_quantity",
        "order_multiple",
        "minimum_reorder_point",
        "maximum_reorder_point",
        "maximum_inventory_position",
        "minimum_cycle_service_level",
    }
    assert result == response


def test_accepts_disposal_cost_component_proposal() -> None:
    response = create_response()

    response["objective_components"].append(
        {
            "component_id": "disposal_cost",
            "reason": (
                "Unverkaufte Ware muss kostenpflichtig "
                "entsorgt werden."
            ),
            "sources": ["source_1"],
        }
    )

    llm = create_mock_llm(response)

    result = call_agent(llm)

    selected_component_ids = {
        component["component_id"]
        for component in result["objective_components"]
    }

    assert "disposal_cost" in selected_component_ids


def test_accepts_order_multiple_constraint_proposal() -> None:
    response = create_response()

    response["constraints"].append(
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

    llm = create_mock_llm(response)

    result = call_agent(llm)

    order_multiple = next(
        constraint
        for constraint in result["constraints"]
        if constraint["component_id"] == "order_multiple"
    )

    assert order_multiple["product_id"] == "product_1"
    assert order_multiple["parameters"] == {
        "multiple": 60,
    }
    assert order_multiple["sources"] == ["source_1"]


def test_accepts_cycle_service_level_proposal() -> None:
    response = create_response()

    response["constraints"].append(
        {
            "component_id": (
                "minimum_cycle_service_level"
            ),
            "product_id": "product_1",
            "parameters": {
                "service_level_percent": 90.0,
            },
            "reason": (
                "Mindestens 90 Prozent der Perioden "
                "sollen ohne Fehlbestand verlaufen."
            ),
            "sources": ["source_1"],
        }
    )

    llm = create_mock_llm(response)

    result = call_agent(
        llm,
        evidence_content=(
            "Die Wahrscheinlichkeit eines vollständigen "
            "Nachfrageausgleichs soll mindestens "
            "90 Prozent betragen."
        ),
    )

    service_constraint = next(
        constraint
        for constraint in result["constraints"]
        if constraint["component_id"]
        == "minimum_cycle_service_level"
    )

    assert service_constraint["product_id"] == "product_1"
    assert service_constraint["parameters"] == {
        "service_level_percent": 90.0,
    }
    assert service_constraint["sources"] == ["source_1"]


def test_returns_unsupported_requirement() -> None:
    response = create_response()
    response["selection_status"] = "unsupported"
    response["unresolved_requirements"] = [
        {
            "description": (
                "Mengenabhängiger Einkaufspreis"
            ),
            "resolution": "unsupported",
            "candidate_component_ids": [],
            "reason": (
                "Für Mengenrabatte ist kein "
                "kompatibler Baustein verfügbar."
            ),
            "sources": ["source_1"],
        }
    ]

    llm = create_mock_llm(response)

    result = call_agent(llm)

    assert result["selection_status"] == "unsupported"
    assert result["unresolved_requirements"] == (
        response["unresolved_requirements"]
    )


def test_derives_selection_status_deterministically() -> None:
    response = create_response()
    response["selection_status"] = "unsupported"
    response["unresolved_requirements"] = []

    llm = create_mock_llm(response)

    result = call_agent(llm)

    assert result["selection_status"] == "complete"
    assert llm.invoke.call_count == 1


def test_rejects_unknown_source_for_unresolved_requirement() -> None:
    response = create_response()
    response["selection_status"] = "unsupported"
    response["unresolved_requirements"] = [
        {
            "description": "Mengenrabatt",
            "resolution": "unsupported",
            "candidate_component_ids": [],
            "reason": (
                "Kein kompatibler Baustein verfügbar."
            ),
            "sources": ["invented_source"],
        }
    ]

    llm = create_mock_llm(response)

    with pytest.raises(
        ValueError,
        match="invented_source",
    ):
        call_agent(llm)


def test_corrects_missing_input_reported_as_clarification() -> None:
    invalid_response = create_response()
    invalid_response["selection_status"] = (
        "needs_clarification"
    )
    invalid_response["unresolved_requirements"] = [
        {
            "description": (
                "Verkaufspreis pro Einheit"
            ),
            "resolution": "clarification",
            "reason": (
                "Der Baustein sales_revenue ist passend, "
                "aber selling_price wurde nicht angegeben."
            ),
            "sources": ["source_1"],
        }
    ]

    corrected_response = create_response()

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content=json.dumps(invalid_response)
        ),
        AIMessage(
            content=json.dumps(corrected_response)
        ),
    ]

    result = call_agent(llm)

    assert result == corrected_response
    assert llm.invoke.call_count == 2


def test_final_selection_rejects_unquoted_missing_input_claim() -> None:
    decision = {
        "objective_sense": "minimize",
        "objective_components": [],
        "constraints": [],
        "additional_requirements": [{
            "description": "Bestellkosten fehlen.",
            "resolution": "unsupported",
            "candidate_component_ids": [],
            "reason": "Ein Eingabewert sei unbekannt.",
            "sources": ["raw_input"],
        }],
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(decision)),
        AIMessage(content=json.dumps({
            **decision, "additional_requirements": [],
        })),
    ]

    result = create_component_selection_agent(
        llm, require_explicit_requirement_evidence=True
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
                "Für einen Artikel möchten wir die jährlichen Bestell- "
                "und Lagerkosten minimieren."
            ),
        }],
    )

    assert result["selection_status"] == "complete"
    assert result["unresolved_requirements"] == []
    assert llm.invoke.call_count == 2
    system_prompt = llm.invoke.call_args_list[0].args[0][0].content
    assert "additional_requirements" in system_prompt
    assert "unresolved_requirements" not in system_prompt
    feedback = json.loads(llm.invoke.call_args_list[1].args[0][-1].content)
    assert "wörtlichen Textbeleg" in feedback["validation_errors"][0]


def test_final_selection_preserves_quoted_additional_requirement() -> None:
    quote = "Jede Lieferung muss an einem Dienstag eintreffen."
    decision = {
        "objective_sense": "minimize",
        "objective_components": [],
        "constraints": [],
        "additional_requirements": [{
            "description": quote,
            "resolution": "unsupported",
            "candidate_component_ids": [],
            "reason": "Kein registrierter Baustein bildet den Liefertag ab.",
            "sources": ["raw_input"],
        }],
    }
    llm = create_mock_llm(decision)

    result = create_component_selection_agent(
        llm, require_explicit_requirement_evidence=True
    )(
        core_id="eoq",
        requirements={"objective": "minimize_cost"},
        products=[],
        global_facts={},
        evidence=[{
            "source_id": "raw_input",
            "content": "Wir bestellen regelmäßig. " + quote,
        }],
    )

    assert result["selection_status"] == "unsupported"
    assert result["unresolved_requirements"] == decision[
        "additional_requirements"
    ]
    llm.invoke.assert_called_once()


def test_corrects_invalid_response_once() -> None:
    invalid_response = create_response()
    invalid_response["selection_status"] = "unsupported"
    invalid_response["unresolved_requirements"] = [
        {}
    ]

    corrected_response = create_response()

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content=json.dumps(invalid_response)
        ),
        AIMessage(
            content=json.dumps(corrected_response)
        ),
    ]

    result = call_agent(llm)

    assert result == corrected_response
    assert llm.invoke.call_count == 2

    correction_messages = (
        llm.invoke.call_args_list[1].args[0]
    )

    feedback = json.loads(
        correction_messages[-1].content
    )

    assert feedback["task"] == (
        "correct_component_selection_proposal"
    )
    assert feedback["attempt_number"] == 2
    assert feedback["maximum_attempts"] == 3

    assert len(feedback["validation_errors"]) >= 3
    assert any(
        "Beschreibung" in error
        for error in feedback["validation_errors"]
    )
    assert any(
        "Begründung" in error
        for error in feedback["validation_errors"]
    )
    assert any(
        "Quelle" in error
        for error in feedback["validation_errors"]
    )

    assert feedback["error_category"] == (
        "proposal_validation"
    )
    assert feedback["required_action"] == (
        "correct_model_draft"
    )

    assert "Validierungsfehler" in (
        correction_messages[-1].content
    )

    assert (
        "darf nicht gleichzeitig bereits ausgewählt sein"
        in correction_messages[-1].content
    )
    assert (
        "entferne sie aus unresolved_requirements"
        in correction_messages[-1].content
    )


def test_corrects_response_after_multiple_attempts() -> None:
    first_invalid_response = create_response()
    first_invalid_response["selection_status"] = "unsupported"
    first_invalid_response["unresolved_requirements"] = [{}]

    second_invalid_response = create_response()
    second_invalid_response["objective_components"][0][
        "component_id"
    ] = "invented_component"

    corrected_response = create_response()

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content=json.dumps(first_invalid_response)
        ),
        AIMessage(
            content=json.dumps(second_invalid_response)
        ),
        AIMessage(
            content=json.dumps(corrected_response)
        ),
    ]

    result = call_agent(llm)

    assert result == corrected_response
    assert llm.invoke.call_count == 3

    third_messages = llm.invoke.call_args_list[2].args[0]

    assert "Validierungsfehler" in (
        third_messages[-1].content
    )
    assert "invented_component" in (
        third_messages[-1].content
    )


def test_revises_existing_proposal_from_critic_feedback(
) -> None:
    previous_proposal = create_response()

    uncovered_requirement = {
        "description": (
            "Jede Lieferung muss an einem Dienstag "
            "eintreffen."
        ),
        "resolution": "unsupported",
        "candidate_component_ids": [],
        "reason": (
            "Für einen bestimmten Liefertag ist kein "
            "Baustein vorhanden."
        ),
        "sources": ["source_1"],
    }

    corrected_response = create_response()
    corrected_response["selection_status"] = "unsupported"
    corrected_response["unresolved_requirements"] = [
        uncovered_requirement
    ]

    llm = create_mock_llm(corrected_response)

    result = call_agent(
        llm,
        previous_proposal=previous_proposal,
        critic_feedback=[uncovered_requirement],
    )

    assert result == corrected_response
    assert llm.invoke.call_count == 1

    messages = llm.invoke.call_args.args[0]

    assert len(messages) == 4

    feedback = json.loads(messages[-1].content)

    assert feedback["task"] == (
        "review_component_selection_proposal"
    )
    assert feedback["critic_feedback"] == [
        uncovered_requirement
    ]
    assert (
        "Prüfe die Hinweise eigenständig"
        in feedback["instruction"]
    )


def test_rejects_invalid_critic_feedback_before_llm(
) -> None:
    previous_proposal = create_response()

    invalid_feedback = {
        "description": (
            "Eine angeblich offene Anforderung."
        ),
        "resolution": "unsupported",
        "candidate_component_ids": [],
        "reason": (
            "Kein kompatibler Baustein vorhanden."
        ),
        "sources": ["invented_source"],
    }

    llm = create_mock_llm(create_response())

    with pytest.raises(
        ValueError,
        match="Ungültige Kritikerhinweise",
    ):
        call_agent(
            llm,
            previous_proposal=previous_proposal,
            critic_feedback=[invalid_feedback],
        )

    llm.invoke.assert_not_called()


def test_rejects_malformed_critic_feedback_before_llm(
) -> None:
    llm = create_mock_llm(create_response())

    with pytest.raises(
        ValueError,
        match="Ungültige Kritikerhinweise",
    ):
        call_agent(
            llm,
            previous_proposal=create_response(),
            critic_feedback=[
                "kein gültiges Objekt"  # type: ignore[list-item]
            ],
        )

    llm.invoke.assert_not_called()


def test_rejects_response_after_failed_correction() -> None:
    invalid_response = create_response()
    invalid_response["selection_status"] = "unsupported"
    invalid_response["unresolved_requirements"] = [
        {}
    ]

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(invalid_response)
    )

    with pytest.raises(
        ValueError,
        match="nach mehreren Korrekturversuchen",
    ):
        call_agent(llm)

    assert llm.invoke.call_count == 3


def test_enforces_locked_objective_selection() -> None:
    locked_proposal = create_response()

    changed_response = create_response()
    changed_response["objective_components"] = [
        *changed_response["objective_components"],
        {
            "component_id": "shortage_cost",
            "reason": "Fehlmengenkosten ergänzen.",
            "sources": ["source_1"],
        },
    ]

    corrected_response = create_response()

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content=json.dumps(changed_response)
        ),
        AIMessage(
            content=json.dumps(corrected_response)
        ),
    ]

    result = call_agent(
        llm,
        locked_objective_proposal=locked_proposal,
    )

    assert result == corrected_response
    assert llm.invoke.call_count == 2

    correction_messages = (
        llm.invoke.call_args_list[1].args[0]
    )
    feedback = json.loads(
        correction_messages[-1].content
    )

    assert feedback["error_category"] == (
        "locked_objective"
    )
    assert feedback["required_action"] == (
        "restore_locked_objective"
    )

    first_messages = llm.invoke.call_args_list[
        0
    ].args[0]
    first_payload = json.loads(
        first_messages[1].content
    )

    assert first_payload[
        "locked_objective_selection"
    ] == {
        "objective_sense": "maximize",
        "objective_component_ids": [
            "sales_revenue",
            "purchase_cost",
        ],
    }
