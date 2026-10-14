import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentic_optimization.agents.core_candidate_agent import (
    create_core_candidate_agent,
    validate_core_candidate_proposal,
)


EVIDENCE = [{
    "source_id": "raw_input",
    "content": "Wir bestellen regelmäßig nach und minimieren Kosten.",
}]


def proposal(core_id: str = "eoq") -> dict:
    return {
        "candidates": [{
            "core_id": core_id,
            "source_id": "raw_input",
            "excerpt": "Wir bestellen regelmäßig nach",
        }],
        "reason": "Wiederkehrende Bestellungen sind plausibel.",
    }


def test_validates_grounded_registered_candidate() -> None:
    result = validate_core_candidate_proposal(
        json.dumps(proposal()), evidence=EVIDENCE
    )

    assert result["candidates"][0]["core_id"] == "eoq"


@pytest.mark.parametrize("change", [
    {"core_id": "invented"},
    {"source_id": "invented"},
    {"excerpt": "nicht in der Quelle"},
    {"open_condition_ids": ["steady_demand"]},
])
def test_rejects_invalid_candidate(change: dict) -> None:
    response = proposal()
    response["candidates"][0].update(change)

    with pytest.raises(ValueError):
        validate_core_candidate_proposal(
            json.dumps(response), evidence=EVIDENCE
        )


def test_no_candidate_is_valid_outcome() -> None:
    result = validate_core_candidate_proposal(
        json.dumps({"candidates": [], "reason": "Kein Kern passt."}),
        evidence=EVIDENCE,
    )

    assert result["candidates"] == []


def test_agent_passes_registry_cards_and_repairs_format() -> None:
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content="kein JSON"),
        AIMessage(content=json.dumps(proposal())),
    ]

    result = create_core_candidate_agent(llm)(evidence=EVIDENCE)

    assert result["candidates"][0]["core_id"] == "eoq"
    assert llm.invoke.call_count == 2
    first_messages = llm.invoke.call_args_list[0].args[0]
    task = json.loads(first_messages[1].content)
    assert {card["core_id"] for card in task["model_cards"]} == {
        "single_period_newsvendor", "eoq", "continuous_review_q_r"
    }
    assert task["evidence"] == EVIDENCE


def test_agent_stops_after_bounded_attempts() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="kein JSON")

    with pytest.raises(ValueError, match="nicht zuverlässig"):
        create_core_candidate_agent(llm, max_attempts=2)(
            evidence=EVIDENCE
        )

    assert llm.invoke.call_count == 2
