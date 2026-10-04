"""Darstellungsdaten für den modellneutralen Geschäftseinstieg."""

from copy import deepcopy
from typing import Literal, TypedDict

from agentic_optimization.state import (
    BusinessIntakeState,
)
from agentic_optimization.ui.business_problem_form import (
    BUSINESS_QUESTIONS,
    BusinessQuestion,
    build_applicability_question,
)


BusinessIntakeMessageType = Literal[
    "none",
    "success",
    "error",
]


class BusinessIntakeViewData(TypedDict):
    """UI-unabhängige Daten des Geschäftseinstiegs."""

    questions: list[BusinessQuestion]
    resolved_field_count: int
    total_field_count: int
    message_type: BusinessIntakeMessageType
    message: str | None


def build_business_intake_view_data(
    intake_state: BusinessIntakeState,
) -> BusinessIntakeViewData:
    """Erzeugt verständliche Darstellungsdaten aus dem State."""

    profile = intake_state.get("profile")

    if profile is None:
        raise ValueError(
            "Das Geschäftsprofil für die Darstellung fehlt."
        )

    questions: list[BusinessQuestion] = []

    for question_id in intake_state.get(
        "pending_question_ids",
        [],
    ):
        question = BUSINESS_QUESTIONS.get(question_id)

        if question is None and question_id.startswith("core_condition:"):
            core_id = intake_state.get("applicability_core_id")
            if core_id is not None:
                question = build_applicability_question(
                    core_id, question_id.split(":", 1)[1]
                )
                plan = intake_state.get("applicability_plan")
                if plan and plan["question_reason"] == "conflict":
                    question["prompt"] = (
                        "Ihre Angaben dazu widersprechen sich. "
                        + question["prompt"]
                    )

        if question is None:
            raise ValueError(
                f"Die Geschäftsfrage '{question_id}' "
                "ist nicht registriert."
            )

        questions.append(deepcopy(question))

    profile_field_ids = (
        "recommendation",
        "procurement_pattern",
        "replenishment_during_period",
        "inventory_review",
        "product_scope",
        "primary_goal",
    )

    resolved_field_count = sum(
        profile[field_id]["status"]
        in {
            "inferred",
            "confirmed",
        }
        for field_id in profile_field_ids
    )

    analysis_error = intake_state.get(
        "analysis_error"
    )
    core_selection = intake_state.get(
        "core_selection"
    )

    if analysis_error:
        message_type: BusinessIntakeMessageType = (
            "error"
        )
        message = analysis_error
    elif (
        core_selection is not None
        and core_selection["status"] == "unsupported"
    ):
        message_type = "error"
        message = (
            "Diese betriebliche Situation kann mit den "
            "derzeit verfügbaren Grundmodellen noch nicht "
            "zuverlässig abgebildet werden."
        )
        if (
            intake_state.get("applicability_plan")
            and intake_state["applicability_plan"]["status"] == "rejected"
        ):
            message += " " + core_selection["reason"]
    elif (
        core_selection is not None
        and core_selection["status"] == "selected"
        and not questions
    ):
        message_type = "success"
        message = (
            "Die betriebliche Ausgangssituation wurde "
            "vollständig erfasst."
        )
    else:
        message_type = "none"
        message = None

    return {
        "questions": questions,
        "resolved_field_count": resolved_field_count,
        "total_field_count": len(profile_field_ids),
        "message_type": message_type,
        "message": message,
    }
