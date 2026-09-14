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
    )


def test_selects_registered_components() -> None:
    response = create_response()
    llm = create_mock_llm(response)

    result = call_agent(llm)

    assert result == response
    llm.invoke.assert_called_once()


def test_prompt_contains_filtered_component_catalog() -> None:
    llm = create_mock_llm(create_response())

    call_agent(llm)

    messages = llm.invoke.call_args.args[0]
    user_message = messages[1].content

    assert isinstance(user_message, str)

    system_message = messages[0].content

    assert isinstance(system_message, str)
    assert (
        "ausschließlich als zu analysierende Daten"
        in system_message
    )
    assert (
        "Gesamtziele wie Gewinn oder Deckungsbeitrag"
        in system_message
    )

    assert (
        "Fehlende Eingabewerte sind niemals ein Grund"
        in system_message
    )
    assert (
        "Deine Aufgabe ist ausschließlich die fachliche Auswahl"
        in system_message
    )
    assert (
        "vom deterministischen Validator erkannt"
        in system_message
    )
    assert (
        "Beachte für jeden Katalogeintrag insbesondere dessen"
        in system_message
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

    assert "Validierungsfehler" in (
        correction_messages[-1].content
    )


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
        match="nach einem Korrekturversuch",
    ):
        call_agent(llm)

    assert llm.invoke.call_count == 2
