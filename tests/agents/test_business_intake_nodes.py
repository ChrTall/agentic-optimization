import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage
import pytest

from agentic_optimization.agents.business_intake_nodes import (
    _profile_question_covers_core_condition,
    create_business_intake_node,
    initialize_modular_core_node,
)
from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
    create_profile_field,
)
from agentic_optimization.state import (
    WORKFLOW_BUSINESS_CASE_UNSUPPORTED,
    WORKFLOW_BUSINESS_CORE_SELECTED,
    WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED,
    WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
    get_workflow_status,
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


@pytest.mark.parametrize(
    ("question_id", "core_id", "condition_id", "covered"),
    [
        ("procurement_pattern", "eoq", "recurring_orders", True),
        ("procurement_pattern", "continuous_review_q_r", "recurring_orders", True),
        ("procurement_pattern", "single_period_newsvendor", "single_decision_period", True),
        ("replenishment_during_period", "single_period_newsvendor", "no_replenishment_within_period", True),
        ("inventory_review", "continuous_review_q_r", "continuous_inventory_review", True),
        ("procurement_pattern", "eoq", "steady_demand", False),
    ],
)
def test_profile_question_covers_registered_core_condition(
    question_id: str, core_id: str, condition_id: str, covered: bool,
) -> None:
    assert _profile_question_covers_core_condition(
        [question_id], core_id, condition_id,
    ) is covered


def test_direct_conditions_override_old_profile_in_model_requirements() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed("order_quantity")
    profile["procurement_pattern"] = create_profile_field(
        status="inferred", value="one_time",
        evidence=[{"source_id": "raw_input", "source_type": "free_text"}],
    )
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")
    raw = "Wir bestellen einmal; der Lieferant bietet Mengenrabatte."
    result = initialize_modular_core_node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "core_selection": {
                "status": "selected", "selected_core_id": "eoq",
                "candidate_core_ids": ["eoq"], "reason": "Bestätigt.",
            },
            "applicability_user_answers": {
                "recurring_orders": "supported",
                "no_quantity_discounts": "supported",
            },
        },
    })
    assert result["requirements"]["planning_horizon"] == "repeated_cycles"
    assert result["requirements"]["explicit_model_conditions"] == {
        "recurring_orders": "supported",
        "no_quantity_discounts": "supported",
    }
    assert profile["procurement_pattern"]["value"] == "one_time"
    answer_source = next(
        item for item in result["modular_model"]["evidence"]
        if item["source_id"] == "business_answers"
    )
    assert "no_quantity_discounts" in answer_source["content"]
    assert "supported" in answer_source["content"]


def test_reconsiders_cached_candidates_after_answer_rules_them_out() -> None:
    raw = "Wir planen die Bestellmenge und den Meldebestand."
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity_and_reorder_point"
    )
    profile["primary_goal"] = confirmed("minimize_cost")
    profile["product_scope"] = confirmed("single_product")
    profile["procurement_pattern"] = create_profile_field(
        status="inferred", value="one_time",
        evidence=[{"source_id": "raw_input", "source_type": "free_text"}],
    )
    previous = {
        "candidates": [{
            "core_id": core_id, "source_id": "raw_input",
            "excerpt": "Bestellmenge",
        } for core_id in ("eoq", "single_period_newsvendor")],
        "reason": "Frühere, noch unvollständige Kandidatenliste.",
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({
            "candidates": [{
                "core_id": "continuous_review_q_r",
                "source_id": "business_answers",
                "excerpt": "continuous_inventory_review",
            }],
            "reason": "Die Antworten beschreiben eine laufende Prüfung.",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]
    result = create_business_intake_node(llm)({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "analyzed_raw_input": raw,
            "candidate_proposal": previous,
            "candidate_analyzed_raw_input": raw,
            "asked_question_ids": [],
            "pending_question_ids": [
                "core_condition:continuous_inventory_review"
            ],
            "pending_answers": [{
                "question_id": "core_condition:continuous_inventory_review",
                "option_id": "yes",
                "source_id": "business_clarification_form",
            }],
            "applicability_user_answers": {
                "single_decision_period": "contradicted",
            },
        },
    })
    assert get_workflow_status(result) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert result["business_intake"]["applicability_core_id"] == (
        "continuous_review_q_r"
    )
    assert result["business_intake"]["pending_question_ids"] == [
        "core_condition:recurring_orders"
    ]
    candidate_payload = json.loads(llm.invoke.call_args_list[0].args[0][1].content)
    assert any(
        item["source_id"] == "business_answers"
        and "continuous_inventory_review" in item["content"]
        for item in candidate_payload["evidence"]
    )


def test_node_selects_core_from_analyzed_description() -> None:
    raw_input = (
        "Wir möchten nur die Bestellmenge bestimmen. "
        "Das Produkt wird regelmäßig nachbestellt. "
        "Es geht um ein Produkt. "
        "Wir möchten die Gesamtkosten reduzieren. "
        "Es gibt keine Mengenrabatte."
    )

    response = {
        "inferences": [
            {
                "field_id": "recommendation",
                "value": "order_quantity",
                "excerpt": (
                    "nur die Bestellmenge bestimmen"
                ),
            },
            {
                "field_id": "procurement_pattern",
                "value": "recurring",
                "excerpt": (
                    "Das Produkt wird regelmäßig nachbestellt"
                ),
            },
            {
                "field_id": "product_scope",
                "value": "single_product",
                "excerpt": "Es geht um ein Produkt",
            },
            {
                "field_id": "primary_goal",
                "value": "minimize_cost",
                "excerpt": (
                    "Wir möchten die Gesamtkosten reduzieren"
                ),
            },
        ],
        "ambiguities": [],
    }

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(response)),
        AIMessage(content=json.dumps({
            "candidates": [{
                "core_id": "eoq",
                "source_id": "raw_input",
                "excerpt": "Das Produkt wird regelmäßig nachbestellt",
            }],
            "reason": "Wiederkehrende Bestellmengenplanung.",
        })),
        AIMessage(content=json.dumps({"findings": [{
            "condition_id": "no_quantity_discounts",
            "status": "supported",
            "source_id": "raw_input",
            "excerpt": "Es gibt keine Mengenrabatte.",
        }]})),
    ]

    node = create_business_intake_node(llm)

    state = {
        "raw_input": raw_input,
        "business_intake": {
            "profile": (
                create_empty_business_problem_profile()
            ),
            "asked_question_ids": [],
        },
    }

    result = node(state)
    intake_result = result["business_intake"]

    assert get_workflow_status(result) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert intake_result["core_selection"]["status"] == (
        "needs_clarification"
    )
    assert intake_result["core_selection"][
        "selected_core_id"
    ] is None
    assert intake_result["pending_question_ids"] == [
        "core_condition:recurring_orders"
    ]
    assert intake_result["analysis_error"] is None
    assert result["raw_input"] == raw_input
    assert llm.invoke.call_count == 3
    assert intake_result["applicability_plan"]["status"] == "needs_clarification"
    assert intake_result["applicability_text_evidence"] == [{
        "condition_id": "no_quantity_discounts",
        "status": "supported",
        "source_id": "raw_input",
        "excerpt": "Es gibt keine Mengenrabatte.",
    }]

    confirmed = node({
        **result,
        "business_intake": {
            **intake_result,
            "pending_answers": [{
                "question_id": "core_condition:recurring_orders",
                "option_id": "yes",
                "source_id": "business_clarification_form",
            }],
        },
    })
    assert get_workflow_status(confirmed) == WORKFLOW_BUSINESS_CORE_SELECTED
    initialized = initialize_modular_core_node(confirmed)
    assert initialized["modular_model"]["core_assumption_evidence"] == (
        intake_result["applicability_text_evidence"]
    )


def test_event_description_requires_explicit_structural_answers():
    raw_input = (
        "Für eine dreitägige Veranstaltung bestellen wir einen Artikel "
        "einmal vor Beginn. Während der Veranstaltung können wir nicht "
        "nachbestellen. Die Nachfrage ist unsicher. Wir möchten die "
        "Bestellmenge festlegen und den erwarteten Gewinn maximieren."
    )
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({
            "inferences": [
                {"field_id": "procurement_pattern", "value": "one_time",
                 "excerpt": "einmal vor Beginn"},
                {"field_id": "replenishment_during_period",
                 "value": "not_possible",
                 "excerpt": "können wir nicht nachbestellen"},
                {"field_id": "product_scope", "value": "single_product",
                 "excerpt": "einen Artikel"},
                {"field_id": "primary_goal", "value": "maximize_profit",
                 "excerpt": "den erwarteten Gewinn maximieren"},
            ],
            "ambiguities": [],
        })),
        AIMessage(content=json.dumps({
            "candidates": [
                {
                    "core_id": core_id,
                    "source_id": "raw_input",
                    "excerpt": "Bestellmenge festlegen",
                }
                for core_id in (
                    "single_period_newsvendor",
                    "continuous_review_q_r",
                )
            ],
            "reason": "Beide Kandidaten zunächst prüfen.",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]

    node = create_business_intake_node(llm)
    result = node({
        "raw_input": raw_input,
        "business_intake": {
            "profile": create_empty_business_problem_profile(),
            "asked_question_ids": [],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert result["business_intake"]["pending_question_ids"] == [
        "core_condition:single_decision_period"
    ]

    for condition_id in (
        "single_decision_period",
        "no_replenishment_within_period",
    ):
        intake = result["business_intake"]
        assert intake["pending_question_ids"] == [
            f"core_condition:{condition_id}"
        ]
        result = node({
            **result,
            "business_intake": {
                **intake,
                "pending_answers": [{
                    "question_id": f"core_condition:{condition_id}",
                    "option_id": "yes",
                    "source_id": "business_clarification_form",
                }],
            },
        })

    assert get_workflow_status(result) == WORKFLOW_BUSINESS_CORE_SELECTED
    assert result["business_intake"]["pending_question_ids"] == []
    assert result["business_intake"]["core_selection"][
        "selected_core_id"
    ] == "single_period_newsvendor"
    assert llm.invoke.call_count == 3


def test_two_cost_candidates_need_structural_clarification() -> None:
    raw = "Wir planen eine Bestellung für einen Artikel."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": [
                {
                    "core_id": core_id,
                    "source_id": "raw_input",
                    "excerpt": "eine Bestellung",
                }
                for core_id in ("eoq", "continuous_review_q_r")
            ],
            "reason": "Der zeitliche Ablauf ist unklar.",
        })),
        AIMessage(content=json.dumps({
            "question": "Wird der Lagerbestand laufend überwacht?",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]
    node = create_business_intake_node(llm)

    first = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "pending_answers": [],
        },
    })
    assert get_workflow_status(first) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    intake = first["business_intake"]
    assert intake["pending_question_ids"] == [
        "core_condition:continuous_inventory_review"
    ]
    assert intake["core_question_plan"]["kind"] == (
        "candidate_condition"
    )
    question = build_business_intake_view_data(intake)["questions"][0]
    assert {item["option_id"] for item in question["options"]} == {
        "yes", "no", "user_unknown"
    }
    assert "laufend überwacht" in question["prompt"]
    assert "laufend" in (
        question["options"][0]["label"]
    )

    second = node({
        **first,
        "business_intake": {
            **intake,
            "pending_answers": [{
                "question_id": "core_condition:continuous_inventory_review",
                "option_id": "no",
                "source_id": "business_clarification_form",
            }],
        },
    })
    assert second["business_intake"]["pending_question_ids"] == [
        "core_condition:recurring_orders"
    ]
    third = node({
        **second,
        "business_intake": {
            **second["business_intake"],
            "pending_answers": [{
                "question_id": "core_condition:recurring_orders",
                "option_id": "yes",
                "source_id": "business_clarification_form",
            }],
        },
    })
    assert get_workflow_status(third) == WORKFLOW_BUSINESS_CORE_SELECTED
    assert third["business_intake"]["core_selection"][
        "selected_core_id"
    ] == "eoq"
    assert llm.invoke.call_count == 4


@pytest.mark.parametrize("candidate_ids", [
    ("eoq", "continuous_review_q_r"),
    ("continuous_review_q_r", "eoq"),
])
def test_open_profile_questions_precede_candidate_specific_questions(
    candidate_ids,
) -> None:
    raw = (
        "Für einen Artikel planen wir eine Bestellung und möchten die "
        "jährlichen Bestell- und Lagerkosten minimieren. Wie oft bestellt "
        "wird, müssen wir noch klären."
    )
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")
    for field_id, values in (
        ("recommendation", ["order_quantity", "reorder_point"]),
        ("procurement_pattern", ["one_time", "recurring"]),
    ):
        profile[field_id] = create_profile_field(
            status="ambiguous", candidate_values=values,
            evidence=[{
                "source_id": "raw_input", "source_type": "free_text",
                "excerpt": "eine Bestellung",
            }],
        )
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": [{
                "core_id": core_id, "source_id": "raw_input",
                "excerpt": "Bestell- und Lagerkosten minimieren",
            } for core_id in candidate_ids],
            "reason": "Die Bestellstruktur ist noch offen.",
        })),
        AIMessage(content=json.dumps({
            "question": "Wird der Lagerbestand laufend überwacht?",
        })),
    ]

    result = create_business_intake_node(llm)({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert result["business_intake"]["pending_question_ids"] == [
        "recommendation", "procurement_pattern",
    ]
    assert result["business_intake"]["core_question_plan"] is None

    next_result = create_business_intake_node(llm)({
        **result,
        "business_intake": {
            **result["business_intake"],
            "pending_answers": [
                {
                    "question_id": "recommendation",
                    "option_id": "order_quantity",
                    "source_id": "business_clarification_form",
                },
                {
                    "question_id": "procurement_pattern",
                    "option_id": "recurring",
                    "source_id": "business_clarification_form",
                },
            ],
        },
    })
    assert next_result["business_intake"]["pending_question_ids"] == [
        "core_condition:continuous_inventory_review"
    ]


@pytest.mark.parametrize("recommendation_status", ["missing", "inferred"])
def test_inferred_recommendation_does_not_change_first_form(
    recommendation_status,
) -> None:
    raw = (
        "Für einen Artikel planen wir eine Bestellung und möchten die "
        "jährlichen Bestell- und Lagerkosten minimieren. Wie oft bestellt "
        "wird, müssen wir noch klären."
    )
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")
    if recommendation_status == "inferred":
        profile["recommendation"] = create_profile_field(
            status="inferred", value="order_quantity",
            evidence=[{
                "source_id": "raw_input", "source_type": "free_text",
                "excerpt": "eine Bestellung",
            }],
        )
    profile["procurement_pattern"] = create_profile_field(
        status="ambiguous", candidate_values=["one_time", "recurring"],
        evidence=[{
            "source_id": "raw_input", "source_type": "free_text",
            "excerpt": "Wie oft bestellt wird, müssen wir noch klären",
        }],
    )
    result = create_business_intake_node(Mock())({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "analyzed_raw_input": raw,
            "candidate_proposal": {
                "candidates": [{
                    "core_id": core_id, "source_id": "raw_input",
                    "excerpt": "Bestell- und Lagerkosten minimieren",
                } for core_id in ("eoq", "continuous_review_q_r")],
                "reason": "Die Bestellstruktur ist noch offen.",
            },
            "candidate_analyzed_raw_input": raw,
            "asked_question_ids": [],
        },
    })

    assert result["business_intake"]["pending_question_ids"] == [
        "recommendation", "procurement_pattern",
    ]


def test_unknown_structural_answers_do_not_force_a_model_choice() -> None:
    raw = "Wir möchten für einen Artikel eine Bestellmenge bestimmen."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": [
                {
                    "core_id": core_id,
                    "source_id": "raw_input",
                    "excerpt": "Bestellmenge bestimmen",
                }
                for core_id in ("eoq", "continuous_review_q_r")
            ],
            "reason": "Die Entscheidungsstruktur ist offen.",
        })),
        AIMessage(content=json.dumps({
            "question": "Wird der Lagerbestand laufend überwacht?"
        })),
    ]
    node = create_business_intake_node(llm)
    state = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
        },
    })

    for condition_id in ("continuous_inventory_review",):
        intake = state["business_intake"]
        assert intake["pending_question_ids"] == [
            f"core_condition:{condition_id}"
        ]
        state = node({
            **state,
            "business_intake": {
                **intake,
                "pending_answers": [{
                    "question_id": f"core_condition:{condition_id}",
                    "option_id": "user_unknown",
                    "source_id": "business_clarification_form",
                }],
            },
        })

    intake = state["business_intake"]
    assert get_workflow_status(state) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert intake["pending_question_ids"] == []
    assert intake["core_selection"]["selected_core_id"] is None
    assert intake["applicability_user_answers"] == {
        "continuous_inventory_review": "unknown",
    }
    view = build_business_intake_view_data(intake)
    assert view["message_type"] == "warning"
    assert "nicht bekannt" in view["message"]


def test_new_description_reopens_previously_deferred_condition() -> None:
    old_raw = "Wir planen eine Bestellung für eine Veranstaltung."
    raw = old_raw + "\nDie Menge wird einmal vor Beginn festgelegt."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("maximize_profit")
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": [{
                "core_id": "single_period_newsvendor",
                "source_id": "raw_input", "excerpt": "Veranstaltung",
            }],
            "reason": "Einmaliger Verkaufszeitraum.",
        })),
        AIMessage(content=json.dumps({"findings": [{
            "condition_id": "single_decision_period",
            "status": "supported",
            "source_id": "raw_input",
            "excerpt": "Die Menge wird einmal vor Beginn festgelegt.",
        }]})),
    ]
    result = create_business_intake_node(llm)({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": ["core_condition:single_decision_period"],
            "pending_question_ids": [],
            "applicability_user_answers": {
                "single_decision_period": "unknown",
            },
            "analyzed_raw_input": old_raw,
        },
    })
    intake = result["business_intake"]
    assert get_workflow_status(result) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert intake["pending_question_ids"] == [
        "core_condition:single_decision_period"
    ]
    assert "single_decision_period" not in intake["applicability_user_answers"]


def test_unknown_primary_goal_is_not_repeated_or_assumed() -> None:
    raw = "Wir möchten eine Bestellmenge für einen Artikel bestimmen."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": [{
                "core_id": "eoq",
                "source_id": "raw_input",
                "excerpt": "Bestellmenge für einen Artikel",
            }],
            "reason": "Bestellmengenplanung.",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]
    node = create_business_intake_node(llm)
    first = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
        },
    })
    assert first["business_intake"]["pending_question_ids"] == [
        "primary_goal"
    ]

    result = node({
        **first,
        "business_intake": {
            **first["business_intake"],
            "pending_answers": [{
                "question_id": "primary_goal",
                "option_id": "user_unknown",
                "source_id": "business_clarification_form",
            }],
        },
    })

    intake = result["business_intake"]
    assert get_workflow_status(result) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert intake["pending_question_ids"] == []
    assert intake["profile"]["primary_goal"]["status"] == "user_unknown"
    assert intake["core_selection"]["selected_core_id"] is None
    assert "Optimierungsziel" in build_business_intake_view_data(
        intake
    )["message"]


def test_form_only_unknown_goal_does_not_select_core() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed("order_quantity")
    profile["procurement_pattern"] = confirmed("recurring")
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = create_profile_field(
        status="user_unknown",
        evidence=[{"source_id": "form", "source_type": "user_selection"}],
    )
    result = create_business_intake_node(Mock())({
        "raw_input": "",
        "business_intake": {
            "profile": profile,
            "asked_question_ids": ["primary_goal"],
        },
    })
    assert get_workflow_status(result) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert result["business_intake"]["core_selection"]["selected_core_id"] is None


@pytest.mark.parametrize("status", ["conflict", "ambiguous"])
def test_candidate_keeps_unresolved_recommendation_question(status) -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = create_profile_field(
        status=status,
        evidence=[{"source_id": "raw_input", "source_type": "free_text"}],
        candidate_values=["order_quantity", "reorder_point"],
    )
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")
    result = create_business_intake_node(Mock())({
        "raw_input": "",
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "candidate_proposal": {
                "candidates": [{"core_id": "eoq", "source_id": "raw_input", "excerpt": "Bestellung"}],
                "reason": "Zu prüfen.",
            },
            "candidate_analyzed_raw_input": "",
        },
    })
    assert get_workflow_status(result) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert "recommendation" in result["business_intake"]["pending_question_ids"]


def test_new_text_keeps_answer_submitted_in_same_turn() -> None:
    old_raw = "Wir planen eine Bestellung."
    new_raw = "Wir planen eine Bestellung für eine Veranstaltung."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("maximize_profit")
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": [{
                "core_id": "single_period_newsvendor",
                "source_id": "raw_input", "excerpt": "für eine Veranstaltung",
            }], "reason": "Einmalige Veranstaltung.",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]
    result = create_business_intake_node(llm)({
        "raw_input": new_raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "pending_question_ids": ["core_condition:single_decision_period"],
            "pending_answers": [{
                "question_id": "core_condition:single_decision_period",
                "option_id": "yes", "source_id": "business_clarification_form",
            }],
            "analyzed_raw_input": old_raw,
        },
    })
    intake = result["business_intake"]
    assert intake["applicability_user_answers"]["single_decision_period"] == "supported"
    assert intake["pending_question_ids"] == [
        "core_condition:no_replenishment_within_period"
    ]


def test_new_text_keeps_core_choice_submitted_in_same_turn() -> None:
    raw = "Wir möchten Bestellmenge und Meldebestand bestimmen."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")
    candidates = [{
        "core_id": core_id, "source_id": "raw_input", "excerpt": "Bestellmenge",
    } for core_id in ("eoq", "continuous_review_q_r")]
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": candidates, "reason": "Zwei denkbare Modelle.",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]
    result = create_business_intake_node(llm)({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "pending_question_ids": ["core_choice"],
            "pending_answers": [{
                "question_id": "core_choice",
                "option_id": "continuous_review_q_r",
                "source_id": "business_clarification_form",
            }],
            "candidate_proposal": {
                "candidates": candidates, "reason": "Bisherige Kandidaten.",
            },
            "analyzed_raw_input": "Ältere Beschreibung",
        },
    })
    intake = result["business_intake"]
    assert intake["chosen_core_id"] == "continuous_review_q_r"
    assert intake["applicability_core_id"] == "continuous_review_q_r"


def test_applicability_failure_retains_answers_for_retry() -> None:
    raw = "Bestellmenge für eine Veranstaltung."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("maximize_profit")
    proposal = {
        "candidates": [{
            "core_id": "single_period_newsvendor",
            "source_id": "raw_input", "excerpt": "Veranstaltung",
        }], "reason": "Zu prüfen.",
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content="{}"), AIMessage(content="{}"),
        AIMessage(content="{}"),
    ]
    node = create_business_intake_node(llm)
    failed = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "pending_question_ids": ["core_condition:single_decision_period"],
            "pending_answers": [{
                "question_id": "core_condition:single_decision_period",
                "option_id": "yes", "source_id": "business_clarification_form",
            }],
            "analyzed_raw_input": raw,
            "candidate_proposal": proposal,
            "candidate_analyzed_raw_input": raw,
        },
    })
    intake = failed["business_intake"]
    assert get_workflow_status(failed) == WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED
    assert "core_condition:single_decision_period" in intake["asked_question_ids"]
    assert intake["applicability_user_answers"]["single_decision_period"] == "supported"
    assert intake["candidate_proposal"] == proposal

    llm.invoke.side_effect = None
    llm.invoke.return_value = AIMessage(content=json.dumps({"findings": []}))
    retried = node(failed)
    assert retried["business_intake"]["applicability_user_answers"][
        "single_decision_period"
    ] == "supported"
    assert retried["business_intake"]["pending_question_ids"] == [
        "core_condition:no_replenishment_within_period"
    ]


def test_candidate_failure_replays_answer_on_retry() -> None:
    raw = "Wir planen eine einmalige Bestellung für eine Veranstaltung."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("maximize_profit")
    answer = {
        "question_id": "core_condition:single_decision_period",
        "option_id": "yes",
        "source_id": "business_clarification_form",
    }
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        *[AIMessage(content="{}") for _ in range(3)],
    ]
    node = create_business_intake_node(llm)
    failed = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "pending_question_ids": [answer["question_id"]],
            "pending_answers": [answer],
            "analyzed_raw_input": "Vorherige Beschreibung",
        },
    })

    assert get_workflow_status(failed) == WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED
    assert failed["business_intake"]["pending_answers"] == [answer]

    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": [{
                "core_id": "single_period_newsvendor",
                "source_id": "raw_input",
                "excerpt": "einmalige Bestellung",
            }],
            "reason": "Einmalige Bestellung vor einer Veranstaltung.",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]
    retried = node(failed)
    intake = retried["business_intake"]
    assert intake["applicability_user_answers"]["single_decision_period"] == "supported"
    assert intake["pending_question_ids"] == [
        "core_condition:no_replenishment_within_period"
    ]


def test_successive_chat_clarifications_keep_earlier_confirmation() -> None:
    raw = "Wir planen eine Bestellung für eine Veranstaltung."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("maximize_profit")
    candidate = AIMessage(content=json.dumps({
        "candidates": [{
            "core_id": "single_period_newsvendor",
            "source_id": "raw_input", "excerpt": "Veranstaltung",
        }],
        "reason": "Einmalige Veranstaltung.",
    }))
    llm = Mock()
    llm.invoke.side_effect = [
        response
        for _ in range(3)
        for response in (
            AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
            candidate,
            AIMessage(content=json.dumps({"findings": []})),
        )
    ]
    node = create_business_intake_node(llm)
    state = node({
        "raw_input": raw,
        "business_intake": {"profile": profile, "asked_question_ids": []},
    })
    assert state["business_intake"]["pending_question_ids"] == [
        "core_condition:single_decision_period"
    ]

    for condition_id, reply in (
        ("single_decision_period", "Ja, wir bestellen einmal vor Beginn."),
        ("no_replenishment_within_period", "Ja, Nachbestellen ist ausgeschlossen."),
    ):
        intake = state["business_intake"]
        state = node({
            **state,
            "raw_input": state["raw_input"] + "\n" + reply,
            "business_intake": {
                **intake,
                "pending_answers": [{
                    "question_id": f"core_condition:{condition_id}",
                    "option_id": "yes", "source_id": "chat_assistant",
                }],
            },
        })

    intake = state["business_intake"]
    assert intake["applicability_user_answers"] == {
        "single_decision_period": "supported",
        "no_replenishment_within_period": "supported",
    }
    assert get_workflow_status(state) == WORKFLOW_BUSINESS_CORE_SELECTED


def test_replacing_description_discards_old_condition_answers() -> None:
    raw = "Neuer Fall: Bestellung für eine Veranstaltung."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("maximize_profit")
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": [{
                "core_id": "single_period_newsvendor",
                "source_id": "raw_input", "excerpt": "Veranstaltung",
            }], "reason": "Zu prüfen.",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]
    result = create_business_intake_node(llm)({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "analyzed_raw_input": "Alter, unabhängiger Fall.",
            "applicability_user_answers": {
                "single_decision_period": "supported",
            },
        },
    })
    assert result["business_intake"]["applicability_user_answers"] == {}
    assert result["business_intake"]["pending_question_ids"] == [
        "core_condition:single_decision_period"
    ]


def test_rejected_candidate_stays_rejected_during_other_core_questions() -> None:
    raw = "Wir planen die Bestandssteuerung für einen Artikel."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")
    proposal = {
        "candidates": [{
            "core_id": core_id,
            "source_id": "raw_input", "excerpt": "einen Artikel",
        } for core_id in (
            "single_period_newsvendor", "continuous_review_q_r",
        )],
        "reason": "Beide Modelle sind zunächst zu prüfen.",
    }
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({"findings": []}))
    node = create_business_intake_node(llm)
    state = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "pending_question_ids": ["core_condition:single_decision_period"],
            "pending_answers": [{
                "question_id": "core_condition:single_decision_period",
                "option_id": "no", "source_id": "business_clarification_form",
            }],
            "analyzed_raw_input": raw,
            "candidate_proposal": proposal,
            "candidate_analyzed_raw_input": raw,
        },
    })
    intake = state["business_intake"]
    assert intake["pending_question_ids"] == [
        "core_condition:recurring_orders"
    ]
    assert intake["core_selection"]["candidate_core_ids"] == [
        "continuous_review_q_r"
    ]

    state = node({
        **state,
        "business_intake": {
            **intake,
            "pending_answers": [{
                "question_id": "core_condition:recurring_orders",
                "option_id": "yes", "source_id": "business_clarification_form",
            }],
        },
    })
    intake = state["business_intake"]
    assert intake["core_selection"]["candidate_core_ids"] == [
        "continuous_review_q_r"
    ]
    assert intake["pending_question_ids"] == [
        "core_condition:continuous_inventory_review"
    ]
    assert intake["applicability_user_answers"] == {
        "single_decision_period": "contradicted",
        "recurring_orders": "supported",
    }


def test_single_candidate_waits_when_required_condition_is_unknown() -> None:
    raw = "Wir planen eine Bestellmenge für einen Verkaufszeitraum."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("maximize_profit")
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": [{
                "core_id": "single_period_newsvendor",
                "source_id": "raw_input",
                "excerpt": "Bestellmenge für einen Verkaufszeitraum",
            }],
            "reason": "Einperiodiges Modell ist zu prüfen.",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]
    node = create_business_intake_node(llm)
    state = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
        },
    })
    assert state["business_intake"]["pending_question_ids"] == [
        "core_condition:single_decision_period"
    ]

    for condition_id, answer in (
        ("single_decision_period", "user_unknown"),
        ("no_replenishment_within_period", "yes"),
    ):
        intake = state["business_intake"]
        state = node({
            **state,
            "business_intake": {
                **intake,
                "pending_answers": [{
                    "question_id": f"core_condition:{condition_id}",
                    "option_id": answer,
                    "source_id": "business_clarification_form",
                }],
            },
        })

    assert get_workflow_status(state) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert state["business_intake"]["pending_question_ids"] == []
    assert state["business_intake"]["core_selection"][
        "selected_core_id"
    ] is None


def test_text_candidate_does_not_require_fixed_profile_answers() -> None:
    raw = "Wir bestellen regelmäßig nach. Bitte berechnen Sie eine Bestellmenge."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": [{
                "core_id": "eoq",
                "source_id": "raw_input",
                "excerpt": "Wir bestellen regelmäßig nach",
            }],
            "reason": "EOQ ist als Kandidat zu prüfen.",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]

    node = create_business_intake_node(llm)
    result = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert result["business_intake"]["pending_question_ids"] == [
        "core_condition:recurring_orders"
    ]
    result = node({
        **result,
        "business_intake": {
            **result["business_intake"],
            "pending_answers": [{
                "question_id": "core_condition:recurring_orders",
                "option_id": "yes",
                "source_id": "business_clarification_form",
            }],
        },
    })
    assert get_workflow_status(result) == WORKFLOW_BUSINESS_CORE_SELECTED
    assert result["business_intake"]["core_selection"][
        "selected_core_id"
    ] == "eoq"


def test_confirmed_newsvendor_conditions_survive_later_questions() -> None:
    raw = "Wir planen die Bestellmenge für einen Verkaufszeitraum."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("maximize_profit")
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": [
                {
                    "core_id": core_id,
                    "source_id": "raw_input",
                    "excerpt": "Bestellmenge",
                }
                for core_id in ("eoq", "single_period_newsvendor")
            ],
            "reason": "Der Ablauf ist noch unklar.",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]
    node = create_business_intake_node(llm)
    state = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
        },
    })

    for condition_id in (
        "single_decision_period", "no_replenishment_within_period",
    ):
        intake = state["business_intake"]
        assert intake["pending_question_ids"] == [
            f"core_condition:{condition_id}"
        ]
        state = node({
            **state,
            "business_intake": {
                **intake,
                "pending_answers": [{
                    "question_id": f"core_condition:{condition_id}",
                    "option_id": "yes",
                    "source_id": "business_clarification_form",
                }],
            },
        })

    assert get_workflow_status(state) == WORKFLOW_BUSINESS_CORE_SELECTED
    assert state["business_intake"]["applicability_user_answers"] == {
        "single_decision_period": "supported",
        "no_replenishment_within_period": "supported",
    }


def test_node_requests_missing_business_answers() -> None:
    llm = Mock()
    node = create_business_intake_node(llm)

    result = node(
        {
            "raw_input": "",
            "business_intake": {
                "profile": (
                    create_empty_business_problem_profile()
                ),
                "asked_question_ids": [],
            },
        }
    )

    intake_result = result["business_intake"]

    assert get_workflow_status(result) == (
        WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    )
    assert intake_result["pending_question_ids"] == [
        "recommendation",
        "procurement_pattern",
        "product_scope",
        "primary_goal",
    ]
    assert intake_result["core_selection"]["status"] == (
        "needs_clarification"
    )
    llm.invoke.assert_not_called()


def test_node_reports_unsupported_business_case() -> None:
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
    profile["product_scope"] = confirmed(
        "single_product"
    )
    profile["primary_goal"] = confirmed(
        "minimize_cost"
    )

    node = create_business_intake_node(Mock())

    result = node(
        {
            "raw_input": "",
            "business_intake": {
                "profile": profile,
                "asked_question_ids": [],
            },
        }
    )

    assert get_workflow_status(result) == (
        WORKFLOW_BUSINESS_CASE_UNSUPPORTED
    )
    assert result["business_intake"][
        "core_selection"
    ]["status"] == "unsupported"


@pytest.mark.parametrize(
    "procurement_status", ["missing", "ambiguous", "inferred"]
)
def test_empty_candidate_proposal_asks_open_procurement_question(
    procurement_status,
) -> None:
    raw = (
        "Für einen Artikel planen wir eine Bestellung und möchten die "
        "jährlichen Bestell- und Lagerkosten minimieren. Wie oft bestellt "
        "wird, müssen wir noch klären."
    )
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed("order_quantity")
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")
    if procurement_status == "ambiguous":
        profile["procurement_pattern"] = create_profile_field(
            status="ambiguous",
            candidate_values=["one_time", "recurring"],
            evidence=[{
                "source_id": "raw_input", "source_type": "free_text",
                "excerpt": "Wie oft bestellt wird, müssen wir noch klären",
            }],
        )
    elif procurement_status == "inferred":
        profile["procurement_pattern"] = create_profile_field(
            status="inferred", value="one_time",
            evidence=[{
                "source_id": "raw_input", "source_type": "free_text",
                "excerpt": "eine Bestellung",
            }],
        )
        profile["replenishment_during_period"] = create_profile_field(
            status="ambiguous",
            candidate_values=["possible", "not_possible"],
            evidence=[{
                "source_id": "raw_input", "source_type": "free_text",
                "excerpt": "Wie oft bestellt wird, müssen wir noch klären",
            }],
        )
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({
            "inferences": [], "ambiguities": [],
        })),
        AIMessage(content=json.dumps({
            "candidates": [], "reason": "Der Bestellrhythmus ist offen.",
        })),
        AIMessage(content=json.dumps({
            "candidates": [{
                "core_id": "eoq", "source_id": "raw_input",
                "excerpt": "Bestell- und Lagerkosten minimieren",
            }],
            "reason": "Wiederkehrende Bestellungen wurden bestätigt.",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]
    node = create_business_intake_node(llm)

    first = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
        },
    })
    assert get_workflow_status(first) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert first["business_intake"]["pending_question_ids"] == [
        "procurement_pattern"
    ]
    assert first["business_intake"]["core_selection"]["status"] == (
        "needs_clarification"
    )

    second = node({
        **first,
        "business_intake": {
            **first["business_intake"],
            "pending_answers": [{
                "question_id": "procurement_pattern",
                "option_id": "recurring",
                "source_id": "business_clarification_form",
            }],
        },
    })
    assert get_workflow_status(second) == WORKFLOW_BUSINESS_CORE_SELECTED
    assert second["business_intake"]["core_selection"][
        "selected_core_id"
    ] == "eoq"


def test_procurement_question_replaces_duplicate_recurring_order_question() -> None:
    raw = (
        "Für einen Artikel planen wir eine Bestellung und möchten die "
        "jährlichen Bestell- und Lagerkosten minimieren. Wie oft bestellt "
        "wird, müssen wir noch klären."
    )
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed("order_quantity")
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")
    profile["procurement_pattern"] = create_profile_field(
        status="ambiguous", candidate_values=["one_time", "recurring"],
        evidence=[{
            "source_id": "raw_input", "source_type": "free_text",
            "excerpt": "Wie oft bestellt wird, müssen wir noch klären",
        }],
    )
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({"findings": []}))
    node = create_business_intake_node(llm)
    first = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "analyzed_raw_input": raw,
            "candidate_proposal": {
                "candidates": [{
                    "core_id": "eoq", "source_id": "raw_input",
                    "excerpt": "Bestell- und Lagerkosten minimieren",
                }],
                "reason": "Die Kostenentscheidung passt möglicherweise zu EOQ.",
            },
            "candidate_analyzed_raw_input": raw,
            "asked_question_ids": [],
            "pending_answers": [],
        },
    })

    assert get_workflow_status(first) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert first["business_intake"]["pending_question_ids"] == [
        "procurement_pattern"
    ]
    assert first["business_intake"]["core_question_plan"] is None

    second = node({
        **first,
        "business_intake": {
            **first["business_intake"],
            "pending_answers": [{
                "question_id": "procurement_pattern",
                "option_id": "recurring",
                "source_id": "business_clarification_form",
            }],
        },
    })

    assert second["business_intake"]["applicability_user_answers"][
        "recurring_orders"
    ] == "supported"
    assert get_workflow_status(second) == WORKFLOW_BUSINESS_CORE_SELECTED


def test_empty_candidate_after_quantity_answer_asks_inferred_procurement() -> None:
    raw = (
        "Für einen Artikel planen wir eine Bestellung und möchten die "
        "jährlichen Bestell- und Lagerkosten minimieren. Wie oft bestellt "
        "wird, müssen wir noch klären."
    )
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = create_profile_field(
        status="inferred", value="minimize_cost",
        evidence=[{
            "source_id": "raw_input", "source_type": "free_text",
            "excerpt": "Bestell- und Lagerkosten minimieren",
        }],
    )
    profile["procurement_pattern"] = create_profile_field(
        status="inferred", value="recurring",
        evidence=[{
            "source_id": "raw_input", "source_type": "free_text",
            "excerpt": "eine Bestellung",
        }],
    )
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({
        "candidates": [], "reason": "Der Bestellrhythmus ist nicht belegt.",
    }))
    node = create_business_intake_node(llm)

    result = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "analyzed_raw_input": raw,
            "candidate_proposal": {
                "candidates": [], "reason": "Noch keine Kandidaten.",
            },
            "candidate_analyzed_raw_input": raw,
            "asked_question_ids": [],
            "pending_question_ids": ["recommendation"],
            "pending_answers": [{
                "question_id": "recommendation",
                "option_id": "order_quantity",
                "source_id": "business_clarification_form",
            }],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    assert result["business_intake"]["pending_question_ids"] == [
        "procurement_pattern"
    ]


def test_empty_candidate_proposal_still_rejects_known_incompatibility() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity_and_reorder_point"
    )
    profile["procurement_pattern"] = confirmed("recurring")
    profile["inventory_review"] = confirmed("periodic")
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({
            "inferences": [], "ambiguities": [],
        })),
        AIMessage(content=json.dumps({
            "candidates": [], "reason": "Kein registriertes Modell passt.",
        })),
    ]

    result = create_business_intake_node(llm)({
        "raw_input": "Wir prüfen den Bestand nur wöchentlich.",
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_BUSINESS_CASE_UNSUPPORTED


def test_node_handles_invalid_agent_response() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content="keine gültige JSON-Antwort"
    )

    node = create_business_intake_node(llm)

    result = node(
        {
            "raw_input": "Eine unklare Beschreibung.",
            "business_intake": {
                "profile": (
                    create_empty_business_problem_profile()
                ),
                "asked_question_ids": [],
                "pending_question_ids": [
                    "recommendation",
                ],
            },
        }
    )

    intake_result = result["business_intake"]

    assert get_workflow_status(result) == (
        WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED
    )
    assert intake_result["analysis_error"] == (
        "Die Beschreibung konnte nicht zuverlässig "
        "ausgewertet werden. Bitte starten Sie die "
        "Analyse erneut."
    )
    assert intake_result["pending_question_ids"] == [
        "recommendation",
    ]
    assert "core_selection" not in intake_result


def test_profile_analysis_failure_replays_submitted_condition_answer() -> None:
    raw = "Wir bestellen einmal vor einer Veranstaltung."
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("maximize_profit")
    answer = {
        "question_id": "core_condition:single_decision_period",
        "option_id": "yes", "source_id": "business_clarification_form",
    }
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="{}")
    node = create_business_intake_node(llm)
    failed = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "pending_question_ids": [answer["question_id"]],
            "pending_answers": [answer],
            "analyzed_raw_input": "Ältere Beschreibung.",
        },
    })

    assert get_workflow_status(failed) == WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED
    assert failed["business_intake"]["pending_answers"] == [answer]

    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"inferences": [], "ambiguities": []})),
        AIMessage(content=json.dumps({
            "candidates": [{
                "core_id": "single_period_newsvendor",
                "source_id": "raw_input", "excerpt": "Veranstaltung",
            }],
            "reason": "Einmalige Bestellung.",
        })),
        AIMessage(content=json.dumps({"findings": []})),
    ]
    retried = node(failed)
    intake = retried["business_intake"]
    assert intake["applicability_user_answers"]["single_decision_period"] == "supported"
    assert intake["pending_question_ids"] == [
        "core_condition:no_replenishment_within_period"
    ]


def test_profile_analysis_failure_retains_submitted_core_choice() -> None:
    answer = {
        "question_id": "core_choice",
        "option_id": "eoq",
        "source_id": "business_clarification_form",
    }
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="{}")
    result = create_business_intake_node(llm)({
        "raw_input": "Wir planen regelmäßig Bestellungen.",
        "business_intake": {
            "profile": create_empty_business_problem_profile(),
            "asked_question_ids": [],
            "pending_question_ids": ["core_choice"],
            "pending_answers": [answer],
            "candidate_proposal": {
                "candidates": [{
                    "core_id": "eoq", "source_id": "raw_input",
                    "excerpt": "Bestellungen",
                }],
                "reason": "Bisheriger Kandidat.",
            },
            "analyzed_raw_input": "Vorherige Beschreibung.",
        },
    })

    assert get_workflow_status(result) == WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED
    assert result["business_intake"]["pending_answers"] == [answer]


def test_node_applies_pending_business_answers() -> None:
    profile = create_empty_business_problem_profile()
    llm = Mock()
    node = create_business_intake_node(llm)

    pending_question_ids = [
        "recommendation",
        "procurement_pattern",
        "product_scope",
        "primary_goal",
    ]

    result = node(
        {
            "raw_input": "",
            "business_intake": {
                "profile": profile,
                "asked_question_ids": [],
                "pending_question_ids": (
                    pending_question_ids
                ),
                "pending_answers": [
                    {
                        "question_id": "recommendation",
                        "option_id": "order_quantity",
                        "source_id": "form_1",
                    },
                    {
                        "question_id": (
                            "procurement_pattern"
                        ),
                        "option_id": "recurring",
                        "source_id": "form_1",
                    },
                    {
                        "question_id": "product_scope",
                        "option_id": "single_product",
                        "source_id": "form_1",
                    },
                    {
                        "question_id": "primary_goal",
                        "option_id": "minimize_cost",
                        "source_id": "form_1",
                    },
                ],
            },
        }
    )

    intake_result = result["business_intake"]

    assert get_workflow_status(result) == (
        WORKFLOW_BUSINESS_CORE_SELECTED
    )
    assert intake_result["core_selection"][
        "selected_core_id"
    ] == "eoq"
    assert intake_result["pending_question_ids"] == []
    assert intake_result["pending_answers"] == []
    assert intake_result["asked_question_ids"] == (
        pending_question_ids
    )
    llm.invoke.assert_not_called()


def test_node_rejects_answer_for_stale_question() -> None:
    node = create_business_intake_node(Mock())

    with pytest.raises(
        ValueError,
        match="nicht mehr offen",
    ):
        node(
            {
                "raw_input": "",
                "business_intake": {
                    "profile": (
                        create_empty_business_problem_profile()
                    ),
                    "asked_question_ids": [],
                    "pending_question_ids": [
                        "recommendation",
                    ],
                    "pending_answers": [
                        {
                            "question_id": "primary_goal",
                            "option_id": "minimize_cost",
                            "source_id": "form_1",
                        }
                    ],
                },
            }
        )


def test_initializes_modular_core_from_selection() -> None:
    state = {
        "business_intake": {
            "profile": (
                create_empty_business_problem_profile()
            ),
            "core_selection": {
                "status": "selected",
                "selected_core_id": "eoq",
                "candidate_core_ids": ["eoq"],
                "reason": "Eindeutige Auswahl.",
            },
        },
        "modular_model": {
            "core_id": "single_period_newsvendor",
            "products": [
                {
                    "product_id": "product_1",
                    "forecast": {},
                    "facts": {},
                }
            ],
            "global_facts": {},
            "evidence": [],
            "proposal": {},
            "specification": {},
            "validation": {},
            "pending_input_submissions": [],
            "review_confirmed": True,
            "optimization_result": {
                "status": "optimal",
            },
            "preparation_error": "Alter Fehler",
            "review_explanation": "Alte Erklärung",
        },
    }

    result = initialize_modular_core_node(state)
    modular_result = result["modular_model"]

    assert modular_result["core_id"] == "eoq"
    assert modular_result["products"] == (
        state["modular_model"]["products"]
    )
    assert modular_result["review_confirmed"] is False
    assert modular_result["optimization_result"] is None
    assert modular_result["preparation_error"] is None

    assert "proposal" not in modular_result
    assert "specification" not in modular_result
    assert "validation" not in modular_result
    assert "review_explanation" not in modular_result
    assert modular_result[
        "pending_input_submissions"
    ] == []

    assert state["modular_model"]["core_id"] == (
        "single_period_newsvendor"
    )
    assert "proposal" in state["modular_model"]


def test_rejects_modular_initialization_before_selection() -> None:
    with pytest.raises(
        ValueError,
        match="noch kein Grundmodell",
    ):
        initialize_modular_core_node(
            {
                "business_intake": {
                    "profile": (
                        create_empty_business_problem_profile()
                    ),
                    "core_selection": {
                        "status": "needs_clarification",
                        "selected_core_id": None,
                        "candidate_core_ids": [
                            "eoq",
                            "single_period_newsvendor",
                        ],
                        "reason": "Weitere Angaben fehlen.",
                    },
                },
            }
        )


def test_rejects_modular_initialization_without_intake() -> None:
    with pytest.raises(
        ValueError,
        match="Geschäftseinstiegs fehlt",
    ):
        initialize_modular_core_node({})


def test_node_does_not_reanalyze_unchanged_description() -> None:
    raw_input = (
        "Das Produkt wird regelmäßig nachbestellt."
    )
    profile = create_empty_business_problem_profile()
    profile["procurement_pattern"] = create_profile_field(
        status="inferred",
        value="recurring",
        evidence=[
            {
                "source_id": "raw_input",
                "source_type": "free_text",
                "excerpt": raw_input,
            }
        ],
    )

    llm = Mock()
    node = create_business_intake_node(llm)

    result = node(
        {
            "raw_input": raw_input,
            "business_intake": {
                "profile": profile,
                "asked_question_ids": [],
                "pending_question_ids": [
                    "recommendation",
                    "product_scope",
                    "primary_goal",
                ],
                "pending_answers": [],
                "analyzed_raw_input": raw_input,
            },
        }
    )

    llm.invoke.assert_not_called()

    assert result["business_intake"][
        "profile"
    ]["procurement_pattern"]["value"] == "recurring"
    assert result["business_intake"][
        "analyzed_raw_input"
    ] == raw_input


def test_initializes_single_product_and_raw_evidence() -> None:
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed(
        "single_product"
    )

    result = initialize_modular_core_node(
        {
            "raw_input": (
                "Das Produkt wird regelmäßig nachbestellt."
            ),
            "business_intake": {
                "profile": profile,
                "core_selection": {
                    "status": "selected",
                    "selected_core_id": "eoq",
                    "candidate_core_ids": ["eoq"],
                    "reason": "Eindeutige Auswahl.",
                },
            },
        }
    )

    modular_state = result["modular_model"]

    assert modular_state["products"] == [
        {
            "product_id": "product_1",
            "forecast": {},
            "facts": {},
        }
    ]
    assert modular_state["global_facts"] == {}
    assert modular_state["evidence"] == [
        {
            "source_id": "raw_input",
            "content": (
                "Das Produkt wird regelmäßig nachbestellt."
            ),
        },
        {
            "source_id": "business_intake_form",
            "content": '{"product_scope": "single_product"}',
        },
    ]
    assert result["requirements"] == {
        "product_scope": "single_product",
        "constraints": [],
        "raw_description": (
            "Das Produkt wird regelmäßig nachbestellt."
        ),
    }


def test_initializes_from_form_without_text_or_stale_proposal() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed("order_quantity")
    profile["procurement_pattern"] = confirmed("recurring")
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")

    result = initialize_modular_core_node(
        {
            "raw_input": "",
            "business_intake": {
                "profile": profile,
                "core_selection": {
                    "status": "selected",
                    "selected_core_id": "eoq",
                    "candidate_core_ids": ["eoq"],
                    "reason": "Eindeutige Auswahl.",
                },
            },
            "modular_model": {
                "evidence": [
                    {
                        "source_id": "raw_input",
                        "content": "Alter Beschreibungstext",
                    }
                ],
                "preliminary_proposal": {
                    "core_id": "single_period_newsvendor",
                },
            },
        }
    )

    modular_state = result["modular_model"]

    assert modular_state["core_id"] == "eoq"
    assert "preliminary_proposal" not in modular_state
    assert len(modular_state["evidence"]) == 1
    assert modular_state["evidence"][0]["source_id"] == (
        "business_intake_form"
    )
    assert json.loads(
        modular_state["evidence"][0]["content"]
    ) == {
        "recommendation": "order_quantity",
        "procurement_pattern": "recurring",
        "product_scope": "single_product",
        "primary_goal": "minimize_cost",
    }


def test_does_not_invent_multiple_products() -> None:
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed(
        "multiple_products"
    )

    result = initialize_modular_core_node(
        {
            "raw_input": "",
            "business_intake": {
                "profile": profile,
                "core_selection": {
                    "status": "selected",
                    "selected_core_id": (
                        "single_period_newsvendor"
                    ),
                    "candidate_core_ids": [
                        "single_period_newsvendor",
                    ],
                    "reason": "Eindeutige Auswahl.",
                },
            },
        }
    )

    assert result["modular_model"]["products"] == []
