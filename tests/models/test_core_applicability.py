import pytest

from agentic_optimization.models.core_applicability import (
    CONDITION_GUIDANCE,
    CORE_CONDITIONS,
    assess_core_applicability,
    ground_applicability_findings,
    plan_core_applicability,
)


def test_unknown_eoq_assumptions_are_not_accepted() -> None:
    result = assess_core_applicability("eoq", {})

    assert result["status"] == "needs_clarification"
    assert set(result["unresolved"]) == {
        "recurring_orders",
        "steady_demand",
        "no_quantity_discounts",
        "no_shortages",
    }


def test_quantity_discounts_reject_classic_eoq() -> None:
    result = assess_core_applicability(
        "eoq",
        {"no_quantity_discounts": "contradicted"},
    )

    assert result["status"] == "rejected"
    assert result["contradicted"] == [
        "no_quantity_discounts"
    ]


def test_supported_eoq_assumptions_make_it_eligible() -> None:
    result = assess_core_applicability(
        "eoq",
        {
            "recurring_orders": "supported",
            "steady_demand": "supported",
            "no_quantity_discounts": "supported",
            "no_shortages": "supported",
        },
    )

    assert result["status"] == "eligible"


def test_repeated_order_decisions_reject_single_period_newsvendor() -> None:
    result = assess_core_applicability(
        "single_period_newsvendor",
        {"single_decision_period": "contradicted"},
    )

    assert result["status"] == "rejected"
    assert result["contradicted"] == [
        "single_decision_period"
    ]


def test_periodic_review_rejects_continuous_q_r() -> None:
    result = assess_core_applicability(
        "continuous_review_q_r",
        {"continuous_inventory_review": "contradicted"},
    )

    assert result["status"] == "rejected"


def test_accepts_finding_with_exact_source_excerpt() -> None:
    findings = ground_applicability_findings(
        "eoq",
        [{
            "condition_id": "steady_demand",
            "status": "supported",
            "source_id": "raw_input",
            "excerpt": "Der Bedarf ist gleichmäßig",
        }],
        {"raw_input": "Der Bedarf ist gleichmäßig über das Jahr."},
    )

    assert findings == {"steady_demand": "supported"}
    assert assess_core_applicability(
        "eoq", findings
    )["status"] == "needs_clarification"


def test_rejects_invented_source_excerpt() -> None:
    with pytest.raises(ValueError, match="Nicht belegter Befund"):
        ground_applicability_findings(
            "eoq",
            [{
                "condition_id": "steady_demand",
                "status": "supported",
                "source_id": "raw_input",
                "excerpt": "Der Bedarf ist gleichmäßig",
            }],
            {"raw_input": "Wir bestellen regelmäßig nach."},
        )


def test_llm_can_support_condition_without_fixed_wording() -> None:
    description = "Der Absatz verläuft Woche für Woche nahezu identisch."
    findings = ground_applicability_findings(
        "eoq",
        [{
            "condition_id": "steady_demand",
            "status": "supported",
            "source_id": "raw_input",
            "excerpt": description,
        }],
        {"raw_input": description},
    )

    assert findings == {"steady_demand": "supported"}


def test_unmentioned_conditions_remain_open() -> None:
    description = "Der Jahresbedarf beträgt 3.500 Einheiten."
    findings = ground_applicability_findings(
        "eoq", [],
        {"raw_input": description},
    )

    assert findings == {}
    assert plan_core_applicability("eoq", findings)[
        "pending_confirmation"
    ] == list(CORE_CONDITIONS["eoq"])


def test_every_condition_has_guidance() -> None:
    assert {
        condition
        for conditions in CORE_CONDITIONS.values()
        for condition in conditions
    } == set(CONDITION_GUIDANCE)


def test_conflicting_evidence_requires_clarification() -> None:
    findings = ground_applicability_findings(
        "eoq",
        [
            {
                "condition_id": "no_quantity_discounts",
                "status": "supported",
                "source_id": "raw_input",
                "excerpt": "Es gibt keine Mengenrabatte",
            },
            {
                "condition_id": "no_quantity_discounts",
                "status": "contradicted",
                "source_id": "clarification_1",
                "excerpt": "Der Lieferant bietet Mengenrabatte",
            },
        ],
        {
            "raw_input": "Es gibt keine Mengenrabatte.",
            "clarification_1": "Der Lieferant bietet Mengenrabatte.",
        },
    )

    assert findings == {"no_quantity_discounts": "conflict"}
    assessment = assess_core_applicability("eoq", findings)
    assert assessment["status"] == "needs_clarification"
    assert assessment["conflicted"] == ["no_quantity_discounts"]


def test_unknown_eoq_ordering_pattern_requires_clarification() -> None:
    plan = plan_core_applicability("eoq", {})

    assert plan["status"] == "needs_clarification"
    assert plan["question_condition_id"] == "recurring_orders"
    assert plan["question_reason"] == "missing"


def test_newsvendor_asks_about_decision_period_first() -> None:
    plan = plan_core_applicability("single_period_newsvendor", {})

    assert plan["status"] == "needs_clarification"
    assert plan["question_condition_id"] == "single_decision_period"
    assert plan["question_reason"] == "missing"


def test_newsvendor_then_asks_about_replenishment() -> None:
    plan = plan_core_applicability(
        "single_period_newsvendor",
        {"single_decision_period": "supported"},
    )

    assert plan["status"] == "needs_clarification"
    assert plan["question_condition_id"] == "no_replenishment_within_period"
    assert plan["question_reason"] == "missing"


def test_unknown_structural_condition_is_not_asked_again_or_accepted() -> None:
    plan = plan_core_applicability(
        "single_period_newsvendor",
        {},
        deferred_conditions={"single_decision_period"},
    )
    assert plan["status"] == "needs_clarification"
    assert plan["question_condition_id"] == "no_replenishment_within_period"

    plan = plan_core_applicability(
        "single_period_newsvendor",
        {"no_replenishment_within_period": "supported"},
        deferred_conditions={"single_decision_period"},
    )
    assert plan["status"] == "needs_clarification"
    assert plan["question_condition_id"] is None


def test_contradiction_rejects_q_r_before_other_questions() -> None:
    plan = plan_core_applicability(
        "continuous_review_q_r",
        {"continuous_inventory_review": "contradicted"},
    )

    assert plan["status"] == "rejected"
    assert plan["question_condition_id"] is None
    assert plan["contradicted"] == ["continuous_inventory_review"]


def test_conflicting_eoq_claims_trigger_targeted_question() -> None:
    plan = plan_core_applicability(
        "eoq",
        {"no_quantity_discounts": "conflict"},
    )

    assert plan["status"] == "needs_clarification"
    assert plan["question_condition_id"] == "no_quantity_discounts"
    assert plan["question_reason"] == "conflict"
    assert "no_quantity_discounts" not in plan["pending_confirmation"]


def test_llm_supported_eoq_structure_still_needs_direct_confirmation() -> None:
    plan = plan_core_applicability(
        "eoq",
        {
            "recurring_orders": "supported",
            "steady_demand": "supported",
            "no_quantity_discounts": "supported",
            "no_shortages": "supported",
        },
    )

    assert plan["status"] == "provisional"
    assert plan["pending_confirmation"] == list(
        CORE_CONDITIONS["eoq"]
    )
    assert plan["question_condition_id"] is None


def test_explicitly_confirmed_conditions_can_proceed_to_review() -> None:
    findings = {
        condition_id: "supported"
        for condition_id in CORE_CONDITIONS["eoq"]
    }
    plan = plan_core_applicability(
        "eoq",
        findings,
        confirmed_conditions=CORE_CONDITIONS["eoq"],
    )

    assert plan["status"] == "review_ready"
    assert plan["pending_confirmation"] == []
