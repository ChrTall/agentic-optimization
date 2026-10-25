from copy import deepcopy
from unittest.mock import Mock
import json

from langchain_core.messages import AIMessage

from agentic_optimization.agents.chat_intake_validation import validate_chat_intake
from agentic_optimization.ui.business_problem_form import BUSINESS_QUESTIONS
from agentic_optimization.ui.chat_diagnostics import intake_snapshot


def test_snapshot_is_independent_and_excludes_unrelated_settings():
    graph = {"business_intake": {"profile": {"goal": "old"}, "secret": "hidden"},
             "llm_config": {"api_key": "hidden"}}
    snapshot = intake_snapshot(graph)
    graph["business_intake"]["profile"]["goal"] = "new"
    assert snapshot["intake"]["profile"]["goal"] == "old"
    assert "hidden" not in json.dumps(snapshot)


def test_trace_explains_rejected_request_without_changing_acceptance_or_call_count():
    answer = {"question_id": "recommendation", "option_id": "both"}
    response = {"findings": [{**answer, "excerpt": "Eine feste Menge wird nachbestellt.",
        "verdict": "explicit", "intent": "answer", "evidence_basis": "direct_statement"}]}
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps(response))
    events = []
    kwargs = dict(message="Eine feste Menge wird nachbestellt.",
                  questions=[BUSINESS_QUESTIONS["recommendation"]], proposed_answers=[answer],
                  profile={}, current_question_id=None)
    without_trace = validate_chat_intake(llm, **kwargs)
    llm.reset_mock()
    with_trace = validate_chat_intake(llm, **kwargs,
                                    diagnostic=lambda event: events.append(deepcopy(event)))
    assert without_trace == with_trace == []
    assert llm.invoke.call_count == 1
    assert json.loads(events[0]["response"]) == response
    assert events[-1] == {"stage": "accepted_answers", "answers": []}
