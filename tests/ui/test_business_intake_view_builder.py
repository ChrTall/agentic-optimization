import pytest

from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
    create_profile_field,
)
from agentic_optimization.ui.business_intake_view_builder import (
    build_business_intake_view_data,
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


def test_builds_pending_questions_in_state_order() -> None:
    profile = create_empty_business_problem_profile()

    view = build_business_intake_view_data(
        {
            "profile": profile,
            "pending_question_ids": [
                "primary_goal",
                "recommendation",
            ],
        }
    )

    assert [
        question["question_id"]
        for question in view["questions"]
    ] == [
        "primary_goal",
        "recommendation",
    ]
    assert view["resolved_field_count"] == 0
    assert view["total_field_count"] == 6
    assert view["message_type"] == "none"


def test_reports_completed_business_intake() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity"
    )
    profile["procurement_pattern"] = confirmed(
        "recurring"
    )
    profile["product_scope"] = confirmed(
        "single_product"
    )
    profile["primary_goal"] = confirmed(
        "minimize_cost"
    )

    view = build_business_intake_view_data(
        {
            "profile": profile,
            "pending_question_ids": [],
            "core_selection": {
                "status": "selected",
                "selected_core_id": "eoq",
                "candidate_core_ids": ["eoq"],
                "reason": "Eindeutige Auswahl.",
            },
        }
    )

    assert view["resolved_field_count"] == 4
    assert view["message_type"] == "success"
    assert view["message"] == (
        "Die betriebliche Ausgangssituation wurde "
        "vollständig erfasst."
    )


def test_hides_technical_model_name_for_unsupported_case() -> None:
    profile = create_empty_business_problem_profile()

    view = build_business_intake_view_data(
        {
            "profile": profile,
            "pending_question_ids": [],
            "core_selection": {
                "status": "unsupported",
                "selected_core_id": None,
                "candidate_core_ids": [],
                "reason": "Technische Auswahlbegründung.",
            },
        }
    )

    assert view["message_type"] == "error"
    assert "Grundmodellen" in view["message"]
    assert "Newsvendor" not in view["message"]
    assert "EOQ" not in view["message"]
    assert "Q,R" not in view["message"]


def test_core_choice_describes_situation_without_model_names() -> None:
    view = build_business_intake_view_data({
        "profile": create_empty_business_problem_profile(),
        "pending_question_ids": ["core_choice"],
        "candidate_proposal": {
            "candidates": [
                {
                    "core_id": core_id,
                    "source_id": "raw_input",
                    "excerpt": "Bestellung",
                }
                for core_id in ("eoq", "single_period_newsvendor")
            ],
            "reason": "Zwei Kandidaten.",
        },
    })

    question = view["questions"][0]
    assert question["question_id"] == "core_choice"
    assert "Bestellsituation" in question["prompt"]
    assert {option["option_id"] for option in question["options"]} == {
        "eoq", "single_period_newsvendor"
    }
    visible = " ".join(
        option["label"] + " " + option["help"]
        for option in question["options"]
    )
    assert "EOQ" not in visible
    assert "Newsvendor" not in visible
    assert "bei jeder Bestellung" in visible
    assert "vor Beginn des Verkaufszeitraums" in visible


def test_candidate_question_ignores_stale_inverted_llm_wording() -> None:
    question = build_business_intake_view_data({
        "profile": create_empty_business_problem_profile(),
        "pending_question_ids": [
            "core_condition:no_replenishment_within_period"
        ],
        "core_question_plan": {
            "question_id": "core_condition:no_replenishment_within_period",
            "kind": "candidate_condition",
            "core_id": "single_period_newsvendor",
            "condition_id": "no_replenishment_within_period",
            "reason": None,
        },
        "core_question_prompt": "Kann währenddessen nachbestellt werden?",
    })["questions"][0]

    assert question["prompt"] == (
        "Sind Nachbestellungen während des Verkaufszeitraums "
        "ausgeschlossen?"
    )
    assert question["options"][0]["label"].startswith("Ja — ")


def test_applicability_conflict_uses_fixed_question() -> None:
    question = build_business_intake_view_data({
        "profile": create_empty_business_problem_profile(),
        "pending_question_ids": ["core_condition:no_quantity_discounts"],
        "core_question_plan": {
            "question_id": "core_condition:no_quantity_discounts",
            "kind": "applicability_condition",
            "core_id": "eoq",
            "condition_id": "no_quantity_discounts",
            "reason": "conflict",
        },
    })["questions"][0]

    assert question["prompt"].startswith(
        "Ihre Angaben dazu widersprechen sich. "
    )
    assert question["options"][0]["label"] == "Ja"


def test_rejects_unknown_pending_question() -> None:
    with pytest.raises(
        ValueError,
        match="nicht registriert",
    ):
        build_business_intake_view_data(
            {
                "profile": (
                    create_empty_business_problem_profile()
                ),
                "pending_question_ids": [
                    "invented_question",
                ],
            }
        )
