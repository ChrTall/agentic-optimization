import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentic_optimization.agents.modular_fact_extraction_agent import (
    create_modular_fact_extraction_agent,
)


def create_products() -> list[dict]:
    return [
        {
            "product_id": "product_1",
            "forecast": {},
            "facts": {},
        }
    ]


def test_extracts_registered_modular_facts() -> None:
    response = {
        "products": [
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
        ],
        "global_facts": {},
    }

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(response)
    )

    agent = create_modular_fact_extraction_agent(llm)

    products, global_facts = agent(
        core_id="single_period_newsvendor",
        requirements={
            "objective": "maximize_profit",
        },
        products=create_products(),
        global_facts={},
        evidence=[
            {
                "source_id": "raw_input",
                "content": (
                    "Erwartete Nachfrage 100, "
                    "Verkaufspreis 12, Einkaufspreis 5."
                ),
            }
        ],
    )

    assert products[0]["facts"] == {
        "selling_price": 12.0,
        "purchase_cost": 5.0,
    }
    assert products[0]["forecast"][
        "expected_value"
    ] == 100.0
    assert global_facts == {}

    llm.invoke.assert_called_once()


def test_rejects_unregistered_extracted_fact() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "products": [
                    {
                        "product_id": "product_1",
                        "forecast": {},
                        "facts": {
                            "invented_cost": 10.0,
                        },
                    }
                ],
                "global_facts": {},
            }
        )
    )

    agent = create_modular_fact_extraction_agent(llm)

    with pytest.raises(
        ValueError,
        match="Nicht erlaubte Produktfelder",
    ):
        agent(
            core_id="single_period_newsvendor",
            requirements={},
            products=create_products(),
            global_facts={},
            evidence=[],
        )


def test_accepts_json_code_block() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=(
            "```json\n"
            '{"products": [], "global_facts": {}}\n'
            "```"
        )
    )

    agent = create_modular_fact_extraction_agent(llm)

    products, global_facts = agent(
        core_id="single_period_newsvendor",
        requirements={},
        products=create_products(),
        global_facts={},
        evidence=[],
    )

    assert products == create_products()
    assert global_facts == {}


def test_extracts_disposal_cost_from_evidence() -> None:
    response = {
        "products": [
            {
                "product_id": "product_1",
                "forecast": {},
                "facts": {
                    "disposal_cost": 20.0,
                },
            }
        ],
        "global_facts": {},
    }

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(response)
    )

    agent = create_modular_fact_extraction_agent(llm)

    products, global_facts = agent(
        core_id="single_period_newsvendor",
        requirements={
            "objective": "maximize_profit",
        },
        products=create_products(),
        global_facts={},
        evidence=[
            {
                "source_id": "raw_input",
                "content": (
                    "Unverkaufte Ware muss entsorgt werden. "
                    "Die Entsorgung kostet 20 Euro pro Stück."
                ),
            }
        ],
    )

    assert products[0]["facts"]["disposal_cost"] == 20.0
    assert global_facts == {}

    messages = llm.invoke.call_args.args[0]
    user_message = messages[1].content

    assert isinstance(user_message, str)
    assert "disposal_cost" in user_message
