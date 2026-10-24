from copy import deepcopy
from unittest.mock import Mock

import pytest

import agentic_optimization.agents.modular_model_agent as module
from agentic_optimization.models.preparation_recovery import (
    ModelPreparationFailure, failed_preparation_state, preparation_failure_details,
)


def setup_agent(monkeypatch):
    product = {"product_id": "product_1", "forecast": {}, "facts": {}}
    proposal = {"core_id": "eoq", "selection_status": "complete", "objective_sense": "minimize",
                "objective_components": [{"component_id": cid, "reason": "Kosten", "sources": ["raw_input"]}
                                         for cid in ["annual_ordering_cost", "annual_holding_cost"]],
                "constraints": [], "unresolved_requirements": []}
    facts = Mock(return_value=([deepcopy(product)], {}))
    selection = Mock(side_effect=[ValueError("invalid JSON with secret input"), proposal])
    integrity = Mock(return_value=proposal)
    monkeypatch.setattr(module, "create_modular_fact_extraction_agent", lambda llm: facts)
    monkeypatch.setattr(module, "create_component_selection_agent", lambda llm, **kwargs: selection)
    monkeypatch.setattr(module, "create_component_selection_integrity_agent", lambda llm: integrity)
    args = {"core_id": "eoq", "products": [product], "global_facts": {},
            "requirements": {"objective": "minimize_cost"}, "evidence": []}
    return module.create_modular_model_agent(Mock(), enable_coverage_audit=False), args, facts, selection, integrity


def test_retry_resumes_failed_selection_without_repeating_completed_extraction(monkeypatch):
    agent, args, facts, selection, integrity = setup_agent(monkeypatch)
    with pytest.raises(ModelPreparationFailure) as failure:
        agent(**args)
    assert failure.value.stage == "selection"
    checkpoint = deepcopy(failure.value.checkpoint)
    draft = agent(**args, checkpoint=checkpoint)
    assert facts.call_count == 1
    assert selection.call_count == 2
    assert integrity.call_count == 1
    assert draft["specification"]["core_id"] == "eoq"
    assert "selection" not in checkpoint["completed"]


def test_changed_inputs_invalidate_cached_preparation_steps(monkeypatch):
    agent, args, facts, selection, integrity = setup_agent(monkeypatch)
    with pytest.raises(ModelPreparationFailure) as failure:
        agent(**args)
    args["products"][0]["facts"]["ordering_cost"] = 50
    agent(**args, checkpoint=failure.value.checkpoint)
    assert facts.call_count == 2


def test_error_details_never_echo_raw_exception_or_erase_user_data():
    raw = "Authorization: Bearer SECRET / private user input"
    details = preparation_failure_details(ValueError(raw), "selection")
    assert raw not in str(details)
    assert details["kind"] == "invalid_response"
    assert details["reference"]
    state = {"raw_input": "Meine Beschreibung", "business_intake": {"pending_answers": ["Antwort"]},
             "modular_model": {"products": [{"product_id": "product_1"}],
                               "specification": {}, "proposal": {}, "validation": {}}}
    result = failed_preparation_state(state, ValueError(raw))
    assert result["raw_input"] == state["raw_input"]
    assert result["business_intake"] == state["business_intake"]
    assert result["modular_model"]["products"] == state["modular_model"]["products"]
    assert "specification" not in result["modular_model"]
    assert "specification" in state["modular_model"]


def test_provider_timeout_is_reported_without_exception_payload():
    import httpx
    from openai import APITimeoutError

    error = APITimeoutError(request=httpx.Request("POST", "https://example.invalid/private"))
    details = preparation_failure_details(error, "facts")
    assert details["kind"] == "timeout"
    assert "nicht rechtzeitig" in details["message"]
    assert "private" not in str(details)


def test_failure_chat_only_offers_recovery_and_never_asks_for_model_inputs():
    from agentic_optimization.ui.chat_assistant_builder import build_chat_context, build_follow_up_messages

    state = failed_preparation_state({"modular_model": {"products": []}}, ValueError("bad response"))
    context = build_chat_context("preparation_failed", state)
    assert context["current_question"] is None
    assert context["allowed_outputs"] == []
    assert "retry_preparation" in context["allowed_actions"]
    messages, key = build_follow_up_messages("preparation_failed", state, product_rows=[],
                                            previous_step="model_input_form", previous_question_key=None)
    assert "technisch unterbrochen" in messages[0]
    repeated, _ = build_follow_up_messages("preparation_failed", state, product_rows=[],
                                          previous_step="preparation_failed", previous_question_key=key)
    assert repeated == []
