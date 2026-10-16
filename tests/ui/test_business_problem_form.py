import pytest

from agentic_optimization.models.problem_profile import (
    PROFILE_FIELD_ALLOWED_VALUES,
    BusinessProblemProfile,
    create_empty_business_problem_profile,
    create_profile_field,
)
from agentic_optimization.ui.business_problem_form import (
    BUSINESS_QUESTIONS,
    apply_business_question_answer,
    get_relevant_business_questions,
)


def create_profile() -> BusinessProblemProfile:
    return create_empty_business_problem_profile()


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


def get_question_ids(
    profile: BusinessProblemProfile,
) -> list[str]:
    return [
        question["question_id"]
        for question in get_relevant_business_questions(
            profile
        )
    ]


def test_empty_profile_starts_with_general_questions() -> None:
    question_ids = get_question_ids(create_profile())

    assert question_ids == [
        "recommendation",
        "procurement_pattern",
        "product_scope",
        "primary_goal",
    ]


def test_inferred_recommendation_still_requires_direct_choice() -> None:
    profile = create_profile()
    profile["procurement_pattern"] = create_profile_field(
        status="ambiguous",
        candidate_values=["one_time", "recurring"],
        evidence=[{
            "source_id": "raw_input",
            "source_type": "free_text",
            "excerpt": "Wie oft bestellt wird, ist noch offen",
        }],
    )
    profile["recommendation"] = create_profile_field(
        status="inferred",
        value="order_quantity",
        evidence=[{
            "source_id": "raw_input",
            "source_type": "free_text",
            "excerpt": "eine Bestellung",
        }],
    )

    assert "recommendation" in get_question_ids(profile)
    profile["recommendation"] = confirmed("order_quantity")
    assert "recommendation" not in get_question_ids(profile)


def test_one_time_procurement_asks_about_replenishment() -> None:
    profile = create_profile()
    profile["recommendation"] = confirmed(
        "order_quantity"
    )
    profile["procurement_pattern"] = confirmed("one_time")

    question_ids = get_question_ids(profile)

    assert "replenishment_during_period" in question_ids
    assert "inventory_review" not in question_ids


def test_inferred_one_time_procurement_does_not_ask_replenishment_yet() -> None:
    profile = create_profile()
    profile["procurement_pattern"] = create_profile_field(
        status="inferred",
        value="one_time",
        evidence=[{
            "source_id": "raw_input",
            "source_type": "free_text",
            "excerpt": "eine Bestellung",
        }],
    )
    profile["replenishment_during_period"] = create_profile_field(
        status="ambiguous",
        candidate_values=["possible", "not_possible"],
        evidence=[{
            "source_id": "raw_input",
            "source_type": "free_text",
            "excerpt": "später nachbestellen",
        }],
    )

    assert "replenishment_during_period" not in get_question_ids(profile)

    profile["procurement_pattern"] = confirmed("one_time")
    assert "replenishment_during_period" in get_question_ids(profile)


def test_recurring_quantity_only_skips_inventory_review() -> None:
    profile = create_profile()
    profile["recommendation"] = confirmed(
        "order_quantity"
    )
    profile["procurement_pattern"] = confirmed("recurring")

    assert "inventory_review" not in get_question_ids(
        profile
    )


def test_recurring_reorder_decision_asks_inventory_review() -> None:
    profile = create_profile()
    profile["recommendation"] = confirmed(
        "order_quantity_and_reorder_point"
    )
    profile["procurement_pattern"] = confirmed("recurring")

    assert "inventory_review" in get_question_ids(profile)


def test_known_product_ids_skip_product_scope_question() -> None:
    profile = create_profile()
    profile["product_ids"] = create_profile_field(
        status="inferred",
        value=["product_1", "product_2"],
        evidence=[
            {
                "source_id": "products.xlsx",
                "source_type": "document",
            }
        ],
    )

    assert "product_scope" not in get_question_ids(profile)


def test_user_unknown_remains_open_for_adaptive_clarification() -> None:
    profile = create_profile()
    profile["procurement_pattern"] = create_profile_field(
        status="user_unknown",
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )

    assert "procurement_pattern" in get_question_ids(profile)


def test_unknown_primary_goal_is_not_asked_again_immediately() -> None:
    profile = create_profile()
    profile["primary_goal"] = create_profile_field(
        status="user_unknown",
        evidence=[{
            "source_id": "form_1",
            "source_type": "user_selection",
        }],
    )

    questions = get_relevant_business_questions(
        profile, asked_question_ids=["primary_goal"]
    )

    assert "primary_goal" not in [
        question["question_id"] for question in questions
    ]


def test_user_unknown_uses_concrete_procurement_follow_up() -> None:
    profile = create_profile()
    profile["procurement_pattern"] = create_profile_field(
        status="user_unknown",
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )

    questions = get_relevant_business_questions(
        profile,
        asked_question_ids=[
            "procurement_pattern",
        ],
    )

    question_ids = [
        question["question_id"]
        for question in questions
    ]

    assert "procurement_pattern" not in question_ids
    assert (
        "procurement_pattern_follow_up"
        in question_ids
    )

    follow_up = next(
        question
        for question in questions
        if question["question_id"]
        == "procurement_pattern_follow_up"
    )

    assert (
        follow_up["target_field"]
        == "procurement_pattern"
    )
    assert (
        "später erneut bestellt"
        in follow_up["prompt"]
    )


def test_does_not_repeat_exhausted_procurement_questions() -> None:
    profile = create_profile()
    profile["procurement_pattern"] = create_profile_field(
        status="user_unknown",
        evidence=[
            {
                "source_id": "form_1_follow_up",
                "source_type": "user_selection",
            }
        ],
    )

    questions = get_relevant_business_questions(
        profile,
        asked_question_ids=[
            "procurement_pattern",
            "procurement_pattern_follow_up",
        ],
    )

    question_ids = {
        question["question_id"]
        for question in questions
    }

    assert "procurement_pattern" not in question_ids
    assert (
        "procurement_pattern_follow_up"
        not in question_ids
    )


def test_questions_do_not_expose_model_names() -> None:
    visible_parts: list[str] = []

    for question in BUSINESS_QUESTIONS.values():
        visible_parts.append(question["prompt"])
        visible_parts.extend(
            option["label"]
            for option in question["options"]
        )

    visible_text = " ".join(visible_parts)

    for technical_term in (
        "Newsvendor",
        "EOQ",
        "Q,R",
        "stochastisch",
    ):
        assert technical_term not in visible_text


def test_applies_confirmed_business_answer() -> None:
    profile = create_profile()

    updated_profile = apply_business_question_answer(
        profile,
        question_id="recommendation",
        option_id="both",
        source_id="form_1",
    )

    assert updated_profile["recommendation"]["value"] == (
        "order_quantity_and_reorder_point"
    )
    assert (
        updated_profile["recommendation"]["status"]
        == "confirmed"
    )
    assert updated_profile["recommendation"]["evidence"] == [
        {
            "source_id": "form_1",
            "source_type": "user_selection",
            "reference": "recommendation",
        }
    ]


def test_applies_user_unknown_without_domain_value() -> None:
    profile = create_profile()

    updated_profile = apply_business_question_answer(
        profile,
        question_id="procurement_pattern",
        option_id="user_unknown",
        source_id="form_1",
    )

    assert (
        updated_profile["procurement_pattern"]["value"]
        is None
    )
    assert (
        updated_profile["procurement_pattern"]["status"]
        == "user_unknown"
    )


def test_follow_up_updates_original_profile_field() -> None:
    profile = create_profile()

    updated_profile = apply_business_question_answer(
        profile,
        question_id="procurement_pattern_follow_up",
        option_id="recurring",
        source_id="form_1_follow_up",
    )

    assert updated_profile["procurement_pattern"]["value"] == (
        "recurring"
    )
    assert (
        updated_profile["procurement_pattern"]["status"]
        == "confirmed"
    )


def test_business_answer_does_not_mutate_original_profile() -> None:
    profile = create_profile()

    apply_business_question_answer(
        profile,
        question_id="recommendation",
        option_id="order_quantity",
        source_id="form_1",
    )

    assert profile["recommendation"]["value"] is None
    assert profile["recommendation"]["status"] == "missing"


@pytest.mark.parametrize(
    ("question_id", "option_id", "expected_error"),
    [
        (
            "unknown_question",
            "order_quantity",
            "nicht registriert",
        ),
        (
            "recommendation",
            "unknown_option",
            "gehört nicht",
        ),
    ],
)
def test_rejects_invalid_business_answer(
    question_id: str,
    option_id: str,
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        apply_business_question_answer(
            create_profile(),
            question_id=question_id,
            option_id=option_id,
            source_id="form_1",
        )


def test_question_values_match_profile_domain() -> None:
    for question in BUSINESS_QUESTIONS.values():
        target_field = question["target_field"]

        question_values = {
            option["profile_value"]
            for option in question["options"]
            if "profile_value" in option
        }

        assert question_values <= (
            PROFILE_FIELD_ALLOWED_VALUES[target_field]
        )
