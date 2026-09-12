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


def call_agent(llm: Mock):
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
                    "Der Einkaufspreis beträgt 5 Euro. "
                    "Die Mindestbestellmenge beträgt 50 Stück."
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
        "purchase_cost",
        "shortage_cost",
    }
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
    }
    assert result == response
