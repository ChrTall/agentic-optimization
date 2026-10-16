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


@pytest.mark.parametrize("change, message", [
    ({"core_id": "invented"}, "core_id 'invented' ist nicht registriert"),
    ({"source_id": "invented"}, "source_id 'invented' ist unbekannt"),
    ({"excerpt": "nicht in der Quelle"}, "nicht wörtlich vor"),
    ({"excerpt": ""}, "nicht leerer Textbeleg"),
    ({"open_condition_ids": ["steady_demand"]}, "genau core_id"),
])
def test_rejects_invalid_candidate(change: dict, message: str) -> None:
    response = proposal()
    response["candidates"][0].update(change)

    with pytest.raises(ValueError, match=message):
        validate_core_candidate_proposal(
            json.dumps(response), evidence=EVIDENCE
        )


def test_rejects_duplicate_candidate_with_specific_feedback() -> None:
    response = proposal()
    response["candidates"].append(response["candidates"][0].copy())

    with pytest.raises(ValueError, match="Kandidat 2: core_id 'eoq' wurde bereits"):
        validate_core_candidate_proposal(json.dumps(response), evidence=EVIDENCE)


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


def test_agent_retries_unverifiable_excerpt_with_actionable_feedback() -> None:
    invalid = proposal()
    invalid["candidates"][0]["excerpt"] = "Wir bestellen immer nach"
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(invalid)),
        AIMessage(content=json.dumps(proposal())),
    ]

    result = create_core_candidate_agent(llm)(evidence=EVIDENCE)

    assert result["candidates"][0]["excerpt"] == "Wir bestellen regelmäßig nach"
    retry_messages = llm.invoke.call_args_list[1].args[0]
    feedback = json.loads(retry_messages[-1].content)
    assert "nicht wörtlich vor" in feedback["validation_error"]
    assert feedback["allowed_source_ids"] == ["raw_input"]
    assert "eoq" in feedback["allowed_core_ids"]
    assert "vollständigen Vorschlag" in feedback["instruction"]


def test_agent_stops_after_bounded_attempts() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="kein JSON")

    with pytest.raises(ValueError, match="nicht zuverlässig"):
        create_core_candidate_agent(llm, max_attempts=2)(
            evidence=EVIDENCE
        )

    assert llm.invoke.call_count == 2
