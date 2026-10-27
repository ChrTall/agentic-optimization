from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
    create_profile_field,
)
from agentic_optimization.ui.requirements_panel import build_requirement_items


def test_lists_requirements_and_marked_assumptions() -> None:
    profile = create_empty_business_problem_profile()
    profile["primary_goal"] = create_profile_field(
        status="inferred",
        value="minimize_cost",
        evidence=[{
            "source_id": "raw_input",
            "source_type": "free_text",
            "excerpt": "Kosten minimieren",
        }],
    )

    items = build_requirement_items({
        "profile": profile,
        "applicability_text_findings": {"no_quantity_discounts": "supported"},
        "applicability_text_evidence": [{
            "condition_id": "no_quantity_discounts",
            "excerpt": "keine Mengenrabatte",
        }],
        "applicability_user_answers": {"recurring_orders": "supported"},
    })

    by_id = {item["target_id"]: item for item in items}

    goal = by_id["profile:primary_goal"]
    assert goal["kind"] == "requirement"
    assert goal["status"] == "proposed"
    assert goal["value_label"] == "Gesamtkosten reduzieren"
    assert goal["quote"] == "Kosten minimieren"
    assert "user_unknown" not in {o["option_id"] for o in goal["options"]}

    discounts = by_id["condition:no_quantity_discounts"]
    assert discounts["kind"] == "assumption"
    assert discounts["status"] == "proposed"
    assert discounts["quote"] == "keine Mengenrabatte"

    assert by_id["condition:recurring_orders"]["status"] == "confirmed"


def test_missing_fields_are_not_listed() -> None:
    assert build_requirement_items({
        "profile": create_empty_business_problem_profile()
    }) == []
