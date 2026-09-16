from agentic_optimization.agents.business_profile_adapter import (
    build_requirements_from_business_profile,
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


def test_maps_recurring_cost_profile() -> None:
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

    requirements = (
        build_requirements_from_business_profile(
            profile,
            raw_input="Regelmäßig nachbestellen.",
        )
    )

    assert requirements == {
        "decision": "order_quantity",
        "objective": "minimize_cost",
        "product_scope": "single_product",
        "planning_horizon": "repeated_cycles",
        "constraints": [],
        "raw_description": (
            "Regelmäßig nachbestellen."
        ),
    }


def test_maps_service_goal_and_continuous_review() -> None:
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
    profile["primary_goal"] = confirmed(
        "avoid_stockouts"
    )

    requirements = (
        build_requirements_from_business_profile(
            profile
        )
    )

    assert requirements["decision"] is None
    assert requirements["objective"] == (
        "reach_service_level"
    )
    assert requirements["planning_horizon"] == (
        "repeated_cycles"
    )
    assert requirements["review_type"] == (
        "continuous_review"
    )


def test_does_not_invent_unresolved_values() -> None:
    profile = create_empty_business_problem_profile()

    requirements = (
        build_requirements_from_business_profile(
            profile
        )
    )

    assert "decision" not in requirements
    assert "objective" not in requirements
    assert "product_scope" not in requirements
    assert "planning_horizon" not in requirements
    assert "review_type" not in requirements


def test_maps_multiple_product_scope() -> None:
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed(
        "multiple_products"
    )

    requirements = (
        build_requirements_from_business_profile(
            profile
        )
    )

    assert requirements["product_scope"] == (
        "multiple_products"
    )
