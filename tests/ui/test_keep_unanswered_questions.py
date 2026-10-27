"""Bestätigungen in der Übersicht überspringen keine offene Rückfrage."""

from agentic_optimization.state import (
    WORKFLOW_BUSINESS_CORE_SELECTED,
    WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
)
from agentic_optimization.ui.requirements_panel import keep_unanswered_questions


def _result(profile: dict, answers: dict | None = None) -> dict:
    return {
        "business_intake": {
            "profile": profile,
            "applicability_user_answers": answers or {},
            "pending_question_ids": [],
        },
        "control": {"workflow_status": WORKFLOW_BUSINESS_CORE_SELECTED},
    }


PREVIOUS = {
    "pending_question_ids": ["product_scope", "core_condition:recurring_orders"],
    "core_question_plan": {"question_id": "core_condition:recurring_orders"},
}


def test_open_question_keeps_clarification_step() -> None:
    result = keep_unanswered_questions(
        PREVIOUS,
        _result({"product_scope": {"status": "missing"}}),
        step="business_clarification",
    )

    assert result["control"]["workflow_status"] == (
        WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    )
    assert result["business_intake"]["pending_question_ids"] == [
        "product_scope", "core_condition:recurring_orders",
    ]
    assert result["business_intake"]["core_question_plan"] == (
        PREVIOUS["core_question_plan"]
    )


def test_answered_questions_let_the_workflow_continue() -> None:
    result_state = _result(
        {"product_scope": {"status": "confirmed"}},
        {"recurring_orders": "supported"},
    )

    assert keep_unanswered_questions(
        PREVIOUS, result_state, step="business_clarification"
    ) is result_state


def test_other_steps_are_unchanged() -> None:
    result_state = _result({"product_scope": {"status": "missing"}})

    assert keep_unanswered_questions(
        PREVIOUS, result_state, step="model_input_form"
    ) is result_state
