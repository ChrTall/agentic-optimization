from unittest.mock import Mock


from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
    create_profile_field,
)
from agentic_optimization.agents.business_intake import (
    create_initial_business_intake_state,
    prepare_business_intake,
)


def test_skips_text_analysis_without_description() -> None:
    profile = create_empty_business_problem_profile()
    profile_agent = Mock()

    result = prepare_business_intake(
        profile=profile,
        profile_agent=profile_agent,
    )

    profile_agent.assert_not_called()

    assert result["profile"] == profile
    assert result["profile"] is not profile

    assert [
        question["question_id"]
        for question in result["questions"]
    ] == [
        "recommendation",
        "procurement_pattern",
        "product_scope",
        "primary_goal",
    ]
    assert result["core_selection"]["status"] == (
        "needs_clarification"
    )
    assert result["core_selection"][
        "selected_core_id"
    ] is None


def test_analyzes_description_before_selecting_questions() -> None:
    profile = create_empty_business_problem_profile()
    analyzed_profile = (
        create_empty_business_problem_profile()
    )

    analyzed_profile["recommendation"] = (
        create_profile_field(
            status="inferred",
            value="order_quantity",
            evidence=[
                {
                    "source_id": "raw_input",
                    "source_type": "free_text",
                    "excerpt": "wie viel bestellt werden soll",
                }
            ],
        )
    )
    analyzed_profile["procurement_pattern"] = (
        create_profile_field(
            status="inferred",
            value="recurring",
            evidence=[
                {
                    "source_id": "raw_input",
                    "source_type": "free_text",
                    "excerpt": "regelmäßig nachbestellt",
                }
            ],
        )
    )

    profile_agent = Mock(
        return_value=analyzed_profile
    )
    raw_input = (
        "Das Produkt wird regelmäßig nachbestellt. "
        "Wir möchten wissen, wie viel bestellt werden soll."
    )

    result = prepare_business_intake(
        profile=profile,
        profile_agent=profile_agent,
        raw_input=raw_input,
        source_id="raw_input",
    )

    profile_agent.assert_called_once_with(
        profile=profile,
        raw_input=raw_input,
        source_id="raw_input",
    )

    assert [
        question["question_id"]
        for question in result["questions"]
    ] == [
        "product_scope",
        "primary_goal",
    ]
    assert result["core_selection"]["status"] == (
        "selected"
    )
    assert result["core_selection"][
        "selected_core_id"
    ] == "eoq"


def test_forwards_previously_asked_questions() -> None:
    profile = create_empty_business_problem_profile()
    profile["procurement_pattern"] = create_profile_field(
        status="user_unknown",
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )

    profile_agent = Mock()

    result = prepare_business_intake(
        profile=profile,
        profile_agent=profile_agent,
        asked_question_ids=[
            "procurement_pattern",
        ],
    )

    question_ids = [
        question["question_id"]
        for question in result["questions"]
    ]

    assert "procurement_pattern" not in question_ids
    assert (
        "procurement_pattern_follow_up"
        in question_ids
    )


def test_reports_unsupported_business_case() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = create_profile_field(
        status="confirmed",
        value="order_quantity_and_reorder_point",
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )
    profile["procurement_pattern"] = create_profile_field(
        status="confirmed",
        value="recurring",
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )
    profile["inventory_review"] = create_profile_field(
        status="confirmed",
        value="periodic",
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )

    result = prepare_business_intake(
        profile=profile,
        profile_agent=Mock(),
    )

    assert result["core_selection"]["status"] == (
        "unsupported"
    )
    assert result["core_selection"][
        "selected_core_id"
    ] is None


def test_creates_initial_business_intake_state() -> None:
    intake_state = (
        create_initial_business_intake_state()
    )

    assert intake_state["profile"] == (
        create_empty_business_problem_profile()
    )
    assert intake_state["asked_question_ids"] == []
    assert intake_state["pending_question_ids"] == [
        "recommendation",
        "procurement_pattern",
        "product_scope",
        "primary_goal",
    ]
    assert intake_state["pending_answers"] == []
    assert intake_state["analyzed_raw_input"] == ""
    assert intake_state["analysis_error"] is None
    assert "core_selection" not in intake_state
