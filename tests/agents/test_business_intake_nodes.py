import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage
import pytest

from agentic_optimization.agents.business_intake_nodes import (
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

    assert get_workflow_status(result) == (
        WORKFLOW_BUSINESS_CORE_SELECTED
    )
    assert intake_result["core_selection"]["status"] == (
        "selected"
    )
    assert intake_result["core_selection"][
        "selected_core_id"
    ] == "eoq"
    assert intake_result["pending_question_ids"] == []
    assert intake_result["analysis_error"] is None
    assert result["raw_input"] == raw_input
    assert llm.invoke.call_count == 3
    assert intake_result["applicability_plan"]["status"] == "provisional"
    assert intake_result["applicability_text_evidence"] == [{
        "condition_id": "no_quantity_discounts",
        "status": "supported",
        "source_id": "raw_input",
        "excerpt": "Es gibt keine Mengenrabatte.",
    }]

    initialized = initialize_modular_core_node(result)
    assert initialized["modular_model"]["core_assumption_evidence"] == (
        intake_result["applicability_text_evidence"]
    )


def test_event_description_does_not_trigger_redundant_core_questions():
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

    result = create_business_intake_node(llm)({
        "raw_input": raw_input,
        "business_intake": {
            "profile": create_empty_business_problem_profile(),
            "asked_question_ids": [],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_BUSINESS_CORE_SELECTED
    assert result["business_intake"]["pending_question_ids"] == []
    assert result["business_intake"]["core_selection"][
        "selected_core_id"
    ] == "single_period_newsvendor"
    assert llm.invoke.call_count == 3


def test_text_candidates_need_explicit_choice_when_both_plausible() -> None:
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
                for core_id in ("eoq", "single_period_newsvendor")
            ],
            "reason": "Der zeitliche Ablauf ist unklar.",
        })),
        AIMessage(content=json.dumps({
            "question": (
                "Legen Sie die Bestellmenge einmal vor einem "
                "Verkaufszeitraum fest?"
            ),
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
        "core_condition:single_decision_period"
    ]
    question = build_business_intake_view_data(intake)["questions"][0]
    assert {item["option_id"] for item in question["options"]} == {
        "yes", "no", "user_unknown"
    }
    assert question["prompt"].startswith("Legen Sie")
    assert "einmal vor dem betrachteten Zeitraum" in (
        question["options"][0]["label"]
    )

    second = node({
        **first,
        "business_intake": {
            **intake,
            "pending_answers": [{
                "question_id": "core_condition:single_decision_period",
                "option_id": "no",
                "source_id": "business_clarification_form",
            }],
        },
    })
    assert get_workflow_status(second) == WORKFLOW_BUSINESS_CORE_SELECTED
    assert second["business_intake"]["core_selection"][
        "selected_core_id"
    ] == "eoq"
    assert llm.invoke.call_count == 4


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
                for core_id in ("eoq", "single_period_newsvendor")
            ],
            "reason": "Die Entscheidungsstruktur ist offen.",
        })),
        AIMessage(content=json.dumps({
            "question": "Einmalige Entscheidung vor dem Zeitraum?"
        })),
        AIMessage(content=json.dumps({
            "question": "Keine Nachbestellung im Zeitraum?"
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

    for condition_id in (
        "single_decision_period", "no_replenishment_within_period"
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
        "single_decision_period": "unknown",
        "no_replenishment_within_period": "unknown",
    }
    view = build_business_intake_view_data(intake)
    assert view["message_type"] == "warning"
    assert "nicht bekannt" in view["message"]


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

    result = create_business_intake_node(llm)({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_BUSINESS_CORE_SELECTED
    assert result["business_intake"]["pending_question_ids"] == []
    assert result["business_intake"]["core_selection"][
        "selected_core_id"
    ] == "eoq"


def test_confirmed_candidate_conditions_survive_later_core_choice() -> None:
    raw = "Wir planen die Bestellmenge für einen Verkaufszeitraum."
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
                    "excerpt": "Bestellmenge",
                }
                for core_id in ("eoq", "single_period_newsvendor")
            ],
            "reason": "Der Ablauf ist noch unklar.",
        })),
        AIMessage(content=json.dumps({
            "question": "Wird die Menge vor dem Zeitraum festgelegt?"
        })),
        AIMessage(content=json.dumps({
            "question": "Sind Nachbestellungen im Zeitraum ausgeschlossen?"
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
        "single_decision_period", "no_replenishment_within_period"
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

    intake = state["business_intake"]
    assert intake["pending_question_ids"] == ["core_choice"]
    result = node({
        **state,
        "business_intake": {
            **intake,
            "pending_answers": [{
                "question_id": "core_choice",
                "option_id": "single_period_newsvendor",
                "source_id": "business_clarification_form",
            }],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_BUSINESS_CORE_SELECTED
    assert result["business_intake"]["applicability_user_answers"] == {
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
