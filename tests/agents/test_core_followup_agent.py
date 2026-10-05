import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentic_optimization.agents.core_followup_agent import (
    create_core_followup_agent,
    validate_core_followup_question,
)


def test_generates_question_for_registered_condition() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({
        "question": "Wird die Menge einmal vor dem Zeitraum festgelegt?"
    }))

    question = create_core_followup_agent(llm)(
        core_id="single_period_newsvendor",
        condition_id="single_decision_period",
        raw_input="Wir planen eine Bestellung.",
    )

    assert question.startswith("Wird die Menge")
    payload = json.loads(llm.invoke.call_args.args[0][1].content)
    assert payload["condition_id"] == "single_decision_period"
    assert payload["description"] == "Wir planen eine Bestellung."


@pytest.mark.parametrize("response", [
    {"question": "Zu kurz?"},
    {"question": "Eine längere Aussage ohne Fragezeichen"},
    {"question": "Ist dies erlaubt?", "core_id": "eoq"},
])
def test_rejects_bad_question_shape(response) -> None:
    with pytest.raises(ValueError):
        validate_core_followup_question(json.dumps(response))


def test_rejects_condition_not_belonging_to_core() -> None:
    with pytest.raises(ValueError, match="Unbekannte"):
        create_core_followup_agent(Mock())(
            core_id="eoq",
            condition_id="single_decision_period",
            raw_input="Bestellung",
        )
