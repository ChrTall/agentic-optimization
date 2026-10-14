from agentic_optimization.models.core_question_planner import (
    plan_next_core_question,
)


def selection(status="needs_clarification", candidates=None):
    candidates = candidates or ["eoq", "single_period_newsvendor"]
    return {
        "status": status,
        "selected_core_id": None,
        "candidate_core_ids": candidates,
        "reason": "Test",
    }


def plan(core_selection=None, *, applicability_plan=None, core_id=None,
         answers=None, asked=()):
    return plan_next_core_question(
        core_selection or selection(),
        applicability_plan=applicability_plan,
        applicability_core_id=core_id,
        condition_answers=answers or {},
        asked_question_ids=asked,
    )


def test_asks_structural_candidate_condition() -> None:
    question = plan()
    assert question == {
        "question_id": "core_condition:single_decision_period",
        "kind": "candidate_condition",
        "core_id": "single_period_newsvendor",
        "condition_id": "single_decision_period",
        "reason": None,
    }


def test_falls_back_to_core_choice_after_structural_questions() -> None:
    question = plan(asked={
        "core_condition:single_decision_period",
        "core_condition:no_replenishment_within_period",
    })
    assert question["kind"] == "core_choice"
    assert question["question_id"] == "core_choice"


def test_unknown_answer_never_becomes_implicit_core_choice() -> None:
    question = plan(
        answers={"single_decision_period": "unknown"},
        asked={"core_condition:no_replenishment_within_period"},
    )
    assert question is None


def test_applicability_question_carries_conflict_reason() -> None:
    question = plan(
        selection("needs_clarification", ["eoq"]),
        applicability_plan={
            "status": "needs_clarification",
            "question_condition_id": "no_quantity_discounts",
            "question_reason": "conflict",
            "pending_confirmation": [],
            "contradicted": [],
        },
        core_id="eoq",
    )
    assert question == {
        "question_id": "core_condition:no_quantity_discounts",
        "kind": "applicability_condition",
        "core_id": "eoq",
        "condition_id": "no_quantity_discounts",
        "reason": "conflict",
    }


def test_deferred_unknown_needs_no_question() -> None:
    assert plan(
        selection("needs_clarification", ["eoq"]),
        applicability_plan={
            "status": "needs_clarification",
            "question_condition_id": None,
            "question_reason": None,
            "pending_confirmation": [],
            "contradicted": [],
        },
        core_id="eoq",
    ) is None
