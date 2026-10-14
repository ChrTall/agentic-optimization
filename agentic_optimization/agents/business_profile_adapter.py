"""Kompatibilitätsadapter vom Geschäftsprofil zu Requirements."""

from copy import deepcopy

from agentic_optimization.models.core_applicability import PROFILE_CONDITIONS
from agentic_optimization.models.problem_profile import (
    BusinessProblemProfile,
    ProfileField,
    create_profile_field,
)
from agentic_optimization.state import (
    Requirements,
)


def _resolved_value(
    field: ProfileField[object],
) -> object:
    """Liefert ausschließlich bestätigte oder abgeleitete Werte."""

    if field["status"] not in {
        "inferred",
        "confirmed",
    }:
        return None

    return field["value"]


def reconcile_profile_with_condition_answers(
    profile: BusinessProblemProfile,
    *,
    core_id: str,
    condition_answers: dict[str, str],
) -> BusinessProblemProfile:
    """Direkte Antworten überstimmen ältere Profilableitungen desselben Kerns."""

    reconciled = deepcopy(profile)
    for field_id, expected, condition_id in PROFILE_CONDITIONS.get(core_id, ()):
        if condition_answers.get(condition_id) == "supported":
            reconciled[field_id] = create_profile_field(
                status="confirmed",
                value=expected,
                evidence=[{
                    "source_id": "business_answers",
                    "source_type": "user_selection",
                }],
            )
    return reconciled


def build_requirements_from_business_profile(
    profile: BusinessProblemProfile,
    *,
    raw_input: str = "",
    condition_answers: dict[str, str] | None = None,
) -> Requirements:
    """Überführt bekannte Profilwerte in das Übergangsformat."""

    requirements: Requirements = {
        "constraints": [],
        "raw_description": raw_input,
    }
    explicit_conditions = {
        condition_id: answer
        for condition_id, answer in (condition_answers or {}).items()
        if answer in {"supported", "contradicted"}
    }
    if explicit_conditions:
        requirements["explicit_model_conditions"] = explicit_conditions

    recommendation = _resolved_value(
        profile["recommendation"]
    )

    if recommendation == "order_quantity":
        requirements["decision"] = "order_quantity"
    elif recommendation in {
        "reorder_point",
        "order_quantity_and_reorder_point",
    }:
        # Das alte Requirements-Format kann den Meldebestand
        # noch nicht korrekt ausdrücken. Der ausgewählte Kern
        # bleibt dafür die verbindliche Information.
        requirements["decision"] = None

    primary_goal = _resolved_value(
        profile["primary_goal"]
    )

    if primary_goal == "minimize_cost":
        requirements["objective"] = "minimize_cost"
    elif primary_goal == "maximize_profit":
        requirements["objective"] = "maximize_profit"
    elif primary_goal in {
        "avoid_stockouts",
        "reach_service_level",
    }:
        requirements["objective"] = "reach_service_level"

    product_scope = _resolved_value(
        profile["product_scope"]
    )

    if product_scope in {
        "single_product",
        "multiple_products",
    }:
        requirements["product_scope"] = product_scope

    procurement_pattern = _resolved_value(
        profile["procurement_pattern"]
    )

    if procurement_pattern == "one_time":
        requirements["planning_horizon"] = (
            "single_period"
        )
    elif procurement_pattern == "recurring":
        requirements["planning_horizon"] = (
            "repeated_cycles"
        )

    inventory_review = _resolved_value(
        profile["inventory_review"]
    )

    if inventory_review == "continuous":
        requirements["review_type"] = (
            "continuous_review"
        )
    elif inventory_review == "periodic":
        requirements["review_type"] = (
            "periodic_review"
        )

    return requirements
