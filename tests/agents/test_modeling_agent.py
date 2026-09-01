import pytest

from agentic_optimization.agents.modeling_agent import (
    modeling_agent,
)
from agentic_optimization.models.registry import (
    MODEL_REGISTRY,
    find_matching_model_variant,
)
from agentic_optimization.state import (
    WORKFLOW_MODEL_SELECTED,
    WORKFLOW_MODEL_UNSUPPORTED,
    get_workflow_status,
)


def create_supported_requirements() -> dict:
    return {
        "decision": "order_quantity",
        "objective": "maximize_profit",
        "demand_type": "uncertain",
        "product_scope": "single_product",
        "planning_horizon": "single_period",
        "constraints": [],
        "raw_description": "",
    }


def create_supported_eoq_requirements() -> dict:
    return {
        "decision": "order_quantity",
        "objective": "minimize_cost",
        "demand_type": "known",
        "product_scope": "single_product",
        "planning_horizon": "repeated_cycles",
        "constraints": [],
        "raw_description": "",
    }

def test_selects_registered_newsvendor_variant() -> None:
    result = modeling_agent(
        {
            "requirements": create_supported_requirements(),
        }
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_MODEL_SELECTED
    )
    assert result["selected_model_family"] == "newsvendor"
    assert (
        result["selected_model_variant"]
        == "single_product_single_period"
    )
    assert result["model_selection_error"] is None
    assert "model_config" not in result


def test_selects_registered_basic_eoq_variant() -> None:
    result = modeling_agent(
        {
            "requirements": (
                create_supported_eoq_requirements()
            ),
        }
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_MODEL_SELECTED
    )

    assert (
        result["selected_model_family"]
        == "eoq"
    )

    assert (
        result["selected_model_variant"]
        == "basic_eoq"
    )

    assert result["model_selection_error"] is None


def test_registry_contains_basic_eoq_variant() -> None:
    family = MODEL_REGISTRY["eoq"]

    assert family["family_id"] == "eoq"
    assert family["family_name"] == "EOQ"

    variant = family["variants"]["basic_eoq"]

    assert variant["variant_id"] == "basic_eoq"

    assert variant["selection_criteria"] == {
        "decision": "order_quantity",
        "objective": "minimize_cost",
        "demand_type": "known",
        "product_scope": "single_product",
        "planning_horizon": "repeated_cycles",
    }


def test_constraints_do_not_affect_model_selection() -> None:
    requirements = create_supported_requirements()
    requirements["constraints"] = [
        "Budget maximal 10000 Euro",
    ]

    result = modeling_agent(
        {
            "requirements": requirements,
        }
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_MODEL_SELECTED
    )
    assert result["selected_model_family"] == "newsvendor"
    assert (
        result["selected_model_variant"]
        == "single_product_single_period"
    )


@pytest.mark.parametrize(
    ("field_name", "unsupported_value"),
    [
        ("decision", "production_quantity"),
        ("objective", "minimize_cost"),
        ("demand_type", "known"),
        ("product_scope", "multiple_products"),
        ("planning_horizon", "multiple_periods"),
    ],
)
def test_returns_unsupported_for_unregistered_combination(
    field_name: str,
    unsupported_value: str,
) -> None:
    requirements = create_supported_requirements()
    requirements[field_name] = unsupported_value

    result = modeling_agent(
        {
            "requirements": requirements,
        }
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_MODEL_UNSUPPORTED
    )
    assert result["selected_model_family"] is None
    assert result["selected_model_variant"] is None
    assert result["model_selection_error"]
    assert result["ready_for_optimization"] is False
    assert "model_config" not in result


def test_selects_q_r_for_continuous_multi_period_review() -> None:
    requirements = {
        "decision": "order_quantity",
        "objective": "minimize_cost",
        "demand_type": "uncertain",
        "product_scope": "single_product",
        "planning_horizon": "multiple_periods",
        "review_type": "continuous_review",
        "constraints": [],
        "raw_description": "",
    }

    result = modeling_agent(
        {"requirements": requirements}
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_MODEL_SELECTED
    )
    assert result["selected_model_family"] == "reorder_point"
    assert result["selected_model_variant"] == "q_r_policy"


def test_periodic_review_is_reserved_but_not_implemented() -> None:
    requirements = {
        "decision": "order_quantity",
        "objective": "minimize_cost",
        "demand_type": "uncertain",
        "product_scope": "single_product",
        "planning_horizon": "multiple_periods",
        "review_type": "periodic_review",
        "constraints": [],
        "raw_description": "",
    }

    result = modeling_agent(
        {"requirements": requirements}
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_MODEL_UNSUPPORTED
    )
    assert result["selected_model_family"] is None


def test_registry_returns_no_match_for_incomplete_requirements() -> None:
    requirements = create_supported_requirements()
    requirements["planning_horizon"] = None

    match = find_matching_model_variant(requirements)

    assert match is None


def test_registry_rejects_ambiguous_selection_criteria() -> None:
    family = MODEL_REGISTRY["newsvendor"]
    original_variants = family["variants"]

    existing_variant = original_variants[
        "single_product_single_period"
    ]

    duplicate_variant = {
        **existing_variant,
        "variant_id": "duplicate_variant",
        "variant_name": "Duplicate Variant",
    }

    family["variants"] = {
        **original_variants,
        "duplicate_variant": duplicate_variant,
    }

    try:
        with pytest.raises(
            ValueError,
            match="mehrere Varianten",
        ):
            find_matching_model_variant(
                create_supported_requirements()
            )
    finally:
        family["variants"] = original_variants
