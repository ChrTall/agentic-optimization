from agentic_optimization.models.core_candidate_selection import (
    choose_candidate_condition,
    select_from_core_candidates,
)
from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
    create_profile_field,
)


def _proposal(*core_ids):
    return {
        "candidates": [
            {
                "core_id": core_id,
                "source_id": "raw_input",
                "excerpt": "Bestellung",
            }
            for core_id in core_ids
        ],
        "reason": "Die Beschreibung ist vorläufig passend.",
    }


def _confirm(profile, field, value):
    profile[field] = create_profile_field(
        status="confirmed", value=value,
        evidence=[{"source_id": "form", "source_type": "user_selection"}],
    )


def _infer(profile, field, value):
    profile[field] = create_profile_field(
        status="inferred", value=value,
        evidence=[{"source_id": "raw_input", "source_type": "raw_input"}],
    )


def test_one_candidate_can_be_selected_without_fixed_answer_pattern():
    result = select_from_core_candidates(
        _proposal("eoq"), create_empty_business_problem_profile()
    )
    assert result["status"] == "selected"
    assert result["selected_core_id"] == "eoq"


def test_two_candidates_need_user_disambiguation():
    profile = create_empty_business_problem_profile()
    result = select_from_core_candidates(
        _proposal("eoq", "single_period_newsvendor"), profile
    )
    assert result["status"] == "needs_clarification"
    assert result["selected_core_id"] is None

    chosen = select_from_core_candidates(
        _proposal("eoq", "single_period_newsvendor"), profile,
        chosen_core_id="single_period_newsvendor",
    )
    assert chosen["selected_core_id"] == "single_period_newsvendor"


def test_confirmed_profit_goal_rejects_eoq():
    profile = create_empty_business_problem_profile()
    _confirm(profile, "primary_goal", "maximize_profit")
    result = select_from_core_candidates(_proposal("eoq"), profile)
    assert result["status"] == "unsupported"


def test_inferred_event_context_vetoes_recurring_candidate():
    profile = create_empty_business_problem_profile()
    _infer(profile, "procurement_pattern", "one_time")
    _infer(profile, "primary_goal", "maximize_profit")

    result = select_from_core_candidates(
        _proposal("single_period_newsvendor", "continuous_review_q_r"),
        profile,
    )

    assert result["status"] == "selected"
    assert result["selected_core_id"] == "single_period_newsvendor"


def test_confirmed_reorder_point_rejects_eoq():
    profile = create_empty_business_problem_profile()
    _confirm(profile, "recommendation", "reorder_point")
    result = select_from_core_candidates(_proposal("eoq"), profile)
    assert result["status"] == "unsupported"


def test_qr_accepts_order_quantity_as_requested_decision():
    profile = create_empty_business_problem_profile()
    _confirm(profile, "recommendation", "order_quantity")
    _confirm(profile, "procurement_pattern", "recurring")
    _confirm(profile, "inventory_review", "continuous")
    _confirm(profile, "primary_goal", "minimize_cost")

    result = select_from_core_candidates(
        _proposal("continuous_review_q_r"), profile
    )

    assert result["status"] == "selected"
    assert result["selected_core_id"] == "continuous_review_q_r"


def test_explicitly_denied_condition_rejects_candidate():
    result = select_from_core_candidates(
        _proposal("eoq"), create_empty_business_problem_profile(),
        condition_answers={"no_shortages": "contradicted"},
    )
    assert result["status"] == "unsupported"


def test_structural_question_is_asked_once_before_core_choice():
    candidates = ["eoq", "single_period_newsvendor"]
    first = choose_candidate_condition(
        candidates,
        answered_condition_ids=set(),
        asked_question_ids=set(),
    )
    assert first == ("single_period_newsvendor", "single_decision_period")

    second = choose_candidate_condition(
        candidates,
        answered_condition_ids={"single_decision_period"},
        asked_question_ids={"core_condition:single_decision_period"},
    )
    assert second == (
        "single_period_newsvendor", "no_replenishment_within_period"
    )

    final = choose_candidate_condition(
        candidates,
        answered_condition_ids={
            "single_decision_period", "no_replenishment_within_period"
        },
        asked_question_ids=set(),
    )
    assert final is None
