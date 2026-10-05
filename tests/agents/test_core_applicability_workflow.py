"""Integrationsfälle der Kerneignung im Geschäftseinstieg."""

import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage
import pytest

from agentic_optimization.agents.business_intake_nodes import (
    create_business_intake_node,
    initialize_modular_core_node,
)
from agentic_optimization.agents.core_applicability_workflow import (
    plan_candidate_applicability,
)
from agentic_optimization.agents.modular_workflow_nodes import (
    execute_modular_model_node,
)
from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
    create_profile_field,
)
from agentic_optimization.state import (
    WORKFLOW_BUSINESS_CASE_UNSUPPORTED,
    WORKFLOW_BUSINESS_CORE_SELECTED,
    WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
    get_workflow_status,
)
from agentic_optimization.ui.business_intake_view_builder import (
    build_business_intake_view_data,
)


def _confirmed(value):
    return create_profile_field(
        status="confirmed",
        value=value,
        evidence=[{"source_id": "form", "source_type": "user_selection"}],
    )


def _newsvendor_profile():
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = _confirmed("order_quantity")
    profile["procurement_pattern"] = _confirmed("one_time")
    profile["replenishment_during_period"] = _confirmed("not_possible")
    profile["product_scope"] = _confirmed("single_product")
    profile["primary_goal"] = _confirmed("maximize_profit")
    return profile


def _eoq_profile():
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = _confirmed("order_quantity")
    profile["procurement_pattern"] = _confirmed("recurring")
    profile["product_scope"] = _confirmed("single_product")
    profile["primary_goal"] = _confirmed("minimize_cost")
    return profile


def _q_r_profile():
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = _confirmed("reorder_point")
    profile["procurement_pattern"] = _confirmed("recurring")
    profile["inventory_review"] = _confirmed("continuous")
    profile["product_scope"] = _confirmed("single_product")
    profile["primary_goal"] = _confirmed("minimize_cost")
    return profile


def test_user_unknown_overrides_positive_text_inference() -> None:
    plan = plan_candidate_applicability(
        "single_period_newsvendor",
        _newsvendor_profile(),
        {
            "single_decision_period": "supported",
            "no_replenishment_within_period": "supported",
        },
        {"single_decision_period": "unknown"},
    )

    assert plan["status"] == "needs_clarification"
    assert plan["question_condition_id"] is None


@pytest.mark.parametrize(
    ("profile_factory", "raw", "condition_id", "excerpt", "expected"),
    [
        (
            _eoq_profile,
            "Wir bestellen regelmäßig; der Bedarf schwankt stark.",
            "steady_demand",
            "der Bedarf schwankt stark",
            WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
        ),
        (
            _newsvendor_profile,
            "Die Saison dauert 30 Tage; wir bestellen einmal vorher.",
            "single_decision_period",
            "wir bestellen einmal vorher",
            WORKFLOW_BUSINESS_CORE_SELECTED,
        ),
        (
            _q_r_profile,
            "Wir prüfen den Bestand laufend, aber die Lieferzeit schwankt.",
            "constant_lead_time",
            "die Lieferzeit schwankt",
            WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
        ),
        (
            _q_r_profile,
            "Wir prüfen den Bestand nur jeden Freitag.",
            "continuous_inventory_review",
            "nur jeden Freitag",
            WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
        ),
    ],
)
def test_model_assumptions_change_core_routing(
    profile_factory, raw, condition_id, excerpt, expected
) -> None:
    status = (
        "supported"
        if expected == WORKFLOW_BUSINESS_CORE_SELECTED
        else "contradicted"
    )
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({
        "findings": [{
            "condition_id": condition_id,
            "status": status,
            "source_id": "raw_input",
            "excerpt": excerpt,
        }],
    }))

    result = create_business_intake_node(llm)({
        "raw_input": raw,
        "business_intake": {
            "profile": profile_factory(),
            "analyzed_raw_input": raw,
            "asked_question_ids": [],
            "pending_answers": [],
        },
    })

    assert get_workflow_status(result) == expected
    assert result["business_intake"]["applicability_plan"] is not None


def test_multi_product_single_period_is_not_treated_as_multi_period() -> None:
    profile = _newsvendor_profile()
    profile["product_scope"] = _confirmed("multiple_products")

    result = create_business_intake_node(Mock())({
        "raw_input": "",
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "pending_answers": [],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_BUSINESS_CORE_SELECTED
    assert result["business_intake"]["core_selection"][
        "selected_core_id"
    ] == "single_period_newsvendor"


def test_repeated_profit_planning_does_not_choose_eoq() -> None:
    profile = _eoq_profile()
    profile["primary_goal"] = _confirmed("maximize_profit")

    result = create_business_intake_node(Mock())({
        "raw_input": "",
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "pending_answers": [],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_BUSINESS_CASE_UNSUPPORTED


def test_conflicting_text_gets_targeted_question_and_answer() -> None:
    raw = "Wir bestellen einmal, aber entscheiden jede Woche erneut."
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({
        "findings": [{
            "condition_id": "single_decision_period",
            "status": "contradicted",
            "source_id": "raw_input",
            "excerpt": "entscheiden jede Woche erneut",
        }],
    }))
    node = create_business_intake_node(llm)
    initial = {
        "raw_input": raw,
        "business_intake": {
            "profile": _newsvendor_profile(),
            "analyzed_raw_input": raw,
            "asked_question_ids": [],
            "pending_answers": [],
        },
    }

    result = node(initial)

    assert get_workflow_status(result) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    intake = result["business_intake"]
    assert intake["pending_question_ids"] == [
        "core_condition:single_decision_period"
    ]
    question = build_business_intake_view_data(intake)["questions"][0]
    assert question["options"][0]["option_id"] == "yes"
    assert "widersprechen" in question["prompt"]

    answered = node({
        **result,
        "business_intake": {
            **intake,
            "pending_answers": [{
                "question_id": "core_condition:single_decision_period",
                "option_id": "yes",
                "source_id": "business_clarification_form",
            }],
        },
    })

    assert get_workflow_status(answered) == WORKFLOW_BUSINESS_CORE_SELECTED
    assert llm.invoke.call_count == 1
    assert answered["business_intake"]["applicability_plan"]["status"] == (
        "review_ready"
    )


def test_explicit_no_rejects_core() -> None:
    profile = _newsvendor_profile()
    state = {
        "raw_input": "",
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "pending_answers": [],
            "applicability_core_id": "single_period_newsvendor",
            "applicability_analyzed_raw_input": "",
            "applicability_text_findings": {
                "single_decision_period": "conflict",
            },
            "pending_question_ids": [
                "core_condition:single_decision_period"
            ],
        },
    }
    node = create_business_intake_node(Mock())
    result = node({
        **state,
        "business_intake": {
            **state["business_intake"],
            "pending_answers": [{
                "question_id": "core_condition:single_decision_period",
                "option_id": "no",
                "source_id": "chat_assistant",
            }],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_BUSINESS_CASE_UNSUPPORTED
    assert result["business_intake"]["core_selection"]["status"] == (
        "unsupported"
    )


def test_quantity_discount_requires_confirmation_before_rejection() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = _confirmed("order_quantity")
    profile["procurement_pattern"] = _confirmed("recurring")
    profile["product_scope"] = _confirmed("single_product")
    profile["primary_goal"] = _confirmed("minimize_cost")
    raw = "Unser Lieferant gewährt Mengenrabatte."
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({
        "findings": [{
            "condition_id": "no_quantity_discounts",
            "status": "contradicted",
            "source_id": "raw_input",
            "excerpt": "gewährt Mengenrabatte",
        }],
    }))

    node = create_business_intake_node(llm)
    result = node({
        "raw_input": raw,
        "business_intake": {
            "profile": profile,
            "analyzed_raw_input": raw,
            "asked_question_ids": [],
            "pending_answers": [],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    intake = result["business_intake"]
    assert intake["pending_question_ids"] == [
        "core_condition:no_quantity_discounts"
    ]

    confirmed = node({
        **result,
        "business_intake": {
            **intake,
            "pending_answers": [{
                "question_id": "core_condition:no_quantity_discounts",
                "option_id": "no",
                "source_id": "business_clarification_form",
            }],
        },
    })

    assert get_workflow_status(confirmed) == WORKFLOW_BUSINESS_CASE_UNSUPPORTED
    intake = confirmed["business_intake"]
    assert intake["core_selection"]["selected_core_id"] is None
    view = build_business_intake_view_data(intake)
    assert "Mengenrabatte" in view["message"]


def test_unproved_eoq_assumptions_reach_review_as_pending() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = _confirmed("order_quantity")
    profile["procurement_pattern"] = _confirmed("recurring")
    profile["product_scope"] = _confirmed("single_product")
    profile["primary_goal"] = _confirmed("minimize_cost")
    node = create_business_intake_node(Mock())
    result = node({
        "raw_input": "",
        "business_intake": {
            "profile": profile,
            "asked_question_ids": [],
            "pending_answers": [],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_BUSINESS_CORE_SELECTED
    initialized = initialize_modular_core_node(result)
    assert initialized["modular_model"]["pending_core_assumptions"] == [
        "steady_demand",
        "no_quantity_discounts",
        "no_shortages",
    ]
    assert initialized["modular_model"]["core_assumptions_confirmed"] is False


def test_full_eoq_description_keeps_unmentioned_assumptions_open() -> None:
    raw = (
        "Wir führen einen Artikel dauerhaft und bestellen regelmäßig "
        "nach. Der Jahresbedarf beträgt 3.500 Einheiten. Pro Bestellung "
        "entstehen Kosten von 120 Euro, die jährlichen Lagerhaltungskosten "
        "betragen 12 Euro je Einheit. Die Bestell- und Lagerhaltungskosten "
        "sollen minimiert werden. Bitte bestimme die Bestellmenge."
    )
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps({"findings": []})
    )
    result = create_business_intake_node(llm)({
        "raw_input": raw,
        "business_intake": {
            "profile": _eoq_profile(),
            "analyzed_raw_input": raw,
            "asked_question_ids": [],
            "pending_answers": [],
        },
    })

    assert get_workflow_status(result) == WORKFLOW_BUSINESS_CORE_SELECTED
    initialized = initialize_modular_core_node(result)
    assert initialized["modular_model"][
        "pending_core_assumptions"
    ] == [
        "steady_demand",
        "no_quantity_discounts",
        "no_shortages",
    ]


def test_wrong_llm_support_for_regular_ordering_stays_unconfirmed() -> None:
    raw = (
        "Wir führen einen Artikel dauerhaft und bestellen "
        "regelmäßig nach."
    )
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({
        "findings": [{
            "condition_id": "steady_demand",
            "status": "supported",
            "source_id": "raw_input",
            "excerpt": raw,
        }],
    }))

    result = create_business_intake_node(llm)({
        "raw_input": raw,
        "business_intake": {
            "profile": _eoq_profile(),
            "analyzed_raw_input": raw,
            "asked_question_ids": [],
            "pending_answers": [],
        },
    })
    initialized = initialize_modular_core_node(result)

    assert "steady_demand" in initialized["modular_model"][
        "pending_core_assumptions"
    ]


def test_supported_eoq_claim_still_needs_review_confirmation() -> None:
    raw = "Der Bedarf verteilt sich gleichmäßig über das Jahr."
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({
        "findings": [{
            "condition_id": "steady_demand",
            "status": "supported",
            "source_id": "raw_input",
            "excerpt": raw,
        }],
    }))
    result = create_business_intake_node(llm)({
        "raw_input": raw,
        "business_intake": {
            "profile": _eoq_profile(),
            "analyzed_raw_input": raw,
            "asked_question_ids": [],
            "pending_answers": [],
        },
    })

    initialized = initialize_modular_core_node(result)
    assert "core_applicability_evidence" not in initialized[
        "modular_model"
    ]
    assert initialized["modular_model"][
        "pending_core_assumptions"
    ] == [
        "steady_demand",
        "no_quantity_discounts",
        "no_shortages",
    ]


def test_optimization_refuses_unconfirmed_core_assumptions() -> None:
    with pytest.raises(ValueError, match="ausdrücklich bestätigt"):
        execute_modular_model_node({
            "modular_model": {
                "proposal": {},
                "specification": {},
                "pending_core_assumptions": ["steady_demand"],
                "review_confirmed": True,
                "core_assumptions_confirmed": False,
            },
        })
