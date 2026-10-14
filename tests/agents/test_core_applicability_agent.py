import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentic_optimization.agents.core_applicability_agent import (
    create_core_applicability_agent,
)


def _response(findings: list[dict[str, str]]) -> AIMessage:
    return AIMessage(content=json.dumps({"findings": findings}))


def test_extracts_only_grounded_explicit_finding() -> None:
    llm = Mock()
    llm.invoke.return_value = _response([{
        "condition_id": "steady_demand",
        "status": "supported",
        "source_id": "raw_input",
        "excerpt": "Der Bedarf ist gleichmäßig",
    }])

    result = create_core_applicability_agent(llm)(
        core_id="eoq",
        evidence=[{
            "source_id": "raw_input",
            "content": "Der Bedarf ist gleichmäßig über das Jahr.",
        }],
    )

    assert result["findings"] == {"steady_demand": "supported"}
    assert result["assessment"]["status"] == "needs_clarification"
    task = json.loads(llm.invoke.call_args.args[0][1].content)
    assert task["core_id"] == "eoq"
    assert {item["condition_id"] for item in task["conditions"]} == {
        "recurring_orders",
        "steady_demand", "no_quantity_discounts", "no_shortages"
    }


def test_silence_keeps_all_assumptions_open() -> None:
    llm = Mock()
    llm.invoke.return_value = _response([])

    result = create_core_applicability_agent(llm)(
        core_id="eoq",
        evidence=[{
            "source_id": "raw_input",
            "content": "Der Jahresbedarf beträgt 3500 Einheiten.",
        }],
    )

    assert result["findings"] == {}
    assert result["assessment"]["status"] == "needs_clarification"
    assert len(result["assessment"]["unresolved"]) == 4


def test_retries_after_invented_excerpt() -> None:
    llm = Mock()
    llm.invoke.side_effect = [
        _response([{
            "condition_id": "steady_demand",
            "status": "supported",
            "source_id": "raw_input",
            "excerpt": "Der Bedarf ist gleichmäßig",
        }]),
        _response([]),
    ]

    result = create_core_applicability_agent(llm)(
        core_id="eoq",
        evidence=[{
            "source_id": "raw_input",
            "content": "Wir bestellen regelmäßig nach.",
        }],
    )

    assert result["findings"] == {}
    assert llm.invoke.call_count == 2
    feedback = json.loads(
        llm.invoke.call_args_list[1].args[0][-1].content
    )
    assert feedback["task"] == "correct_applicability_findings"


def test_conflicting_sources_are_not_silently_resolved() -> None:
    llm = Mock()
    llm.invoke.return_value = _response([
        {
            "condition_id": "no_quantity_discounts",
            "status": "supported",
            "source_id": "raw_input",
            "excerpt": "Es gibt keine Mengenrabatte",
        },
        {
            "condition_id": "no_quantity_discounts",
            "status": "contradicted",
            "source_id": "clarification_1",
            "excerpt": "Der Lieferant bietet Mengenrabatte",
        },
    ])

    result = create_core_applicability_agent(llm)(
        core_id="eoq",
        evidence=[
            {
                "source_id": "raw_input",
                "content": "Es gibt keine Mengenrabatte.",
            },
            {
                "source_id": "clarification_1",
                "content": "Der Lieferant bietet Mengenrabatte.",
            },
        ],
    )

    assert result["findings"] == {
        "no_quantity_discounts": "conflict"
    }
    assert result["assessment"]["status"] == "needs_clarification"
    assert result["assessment"]["conflicted"] == [
        "no_quantity_discounts"
    ]


def test_rejects_unknown_condition_after_bounded_attempts() -> None:
    llm = Mock()
    llm.invoke.return_value = _response([{
        "condition_id": "invented_condition",
        "status": "supported",
        "source_id": "raw_input",
        "excerpt": "Wir bestellen regelmäßig",
    }])

    with pytest.raises(ValueError, match="nicht zuverlässig"):
        create_core_applicability_agent(llm, max_attempts=2)(
            core_id="eoq",
            evidence=[{
                "source_id": "raw_input",
                "content": "Wir bestellen regelmäßig nach.",
            }],
        )

    assert llm.invoke.call_count == 2
