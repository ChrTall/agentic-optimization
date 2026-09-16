from agentic_optimization.models.core_selector import (
    select_model_core,
)
from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
    create_profile_field,
)


def confirmed(value):
    return create_profile_field(
        status="confirmed",
        value=value,
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )


def test_selects_complete_newsvendor_profile() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity"
    )
    profile["procurement_pattern"] = confirmed(
        "one_time"
    )
    profile["replenishment_during_period"] = confirmed(
        "not_possible"
    )

    decision = select_model_core(profile)

    assert decision["status"] == "selected"
    assert decision["selected_core_id"] == (
        "single_period_newsvendor"
    )
    assert decision["candidate_core_ids"] == [
        "single_period_newsvendor"
    ]


def test_selects_complete_eoq_profile() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity"
    )
    profile["procurement_pattern"] = confirmed(
        "recurring"
    )

    decision = select_model_core(profile)

    assert decision["status"] == "selected"
    assert decision["selected_core_id"] == "eoq"


def test_selects_complete_q_r_profile() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity_and_reorder_point"
    )
    profile["procurement_pattern"] = confirmed(
        "recurring"
    )
    profile["inventory_review"] = confirmed(
        "continuous"
    )

    decision = select_model_core(profile)

    assert decision["status"] == "selected"
    assert decision["selected_core_id"] == (
        "continuous_review_q_r"
    )


def test_does_not_select_incomplete_single_candidate() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity"
    )
    profile["procurement_pattern"] = confirmed(
        "one_time"
    )

    decision = select_model_core(profile)

    assert decision["status"] == "needs_clarification"
    assert decision["selected_core_id"] is None
    assert decision["candidate_core_ids"] == [
        "single_period_newsvendor"
    ]


def test_empty_profile_needs_clarification() -> None:
    profile = create_empty_business_problem_profile()

    decision = select_model_core(profile)

    assert decision["status"] == "needs_clarification"
    assert decision["selected_core_id"] is None
    assert set(decision["candidate_core_ids"]) == {
        "single_period_newsvendor",
        "eoq",
        "continuous_review_q_r",
    }


def test_rejects_unsupported_periodic_review_case() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity_and_reorder_point"
    )
    profile["procurement_pattern"] = confirmed(
        "recurring"
    )
    profile["inventory_review"] = confirmed(
        "periodic"
    )

    decision = select_model_core(profile)

    assert decision["status"] == "unsupported"
    assert decision["selected_core_id"] is None
    assert decision["candidate_core_ids"] == []