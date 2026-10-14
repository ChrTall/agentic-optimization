"""Darstellungsdaten für den modellneutralen Geschäftseinstieg."""

from copy import deepcopy
from typing import Literal, TypedDict

from agentic_optimization.state import (
    BusinessIntakeState,
)
from agentic_optimization.models.core_applicability import CONDITION_LABELS
from agentic_optimization.models.core_registry import CORE_REGISTRY
from agentic_optimization.ui.business_problem_form import (
    BUSINESS_QUESTIONS,
    BusinessQuestion,
    build_applicability_question,
)


BusinessIntakeMessageType = Literal[
    "none",
    "success",
    "warning",
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

        if question is None and question_id == "core_choice":
            proposal = intake_state.get("candidate_proposal")
            candidate_ids = (
                [item["core_id"] for item in proposal["candidates"]]
                if proposal else []
            )
            question = {
                "question_id": "core_choice",
                "prompt": (
                    "Welche Beschreibung passt zu Ihrer "
                    "Bestellsituation?"
                ),
                "options": [
                    {
                        "option_id": core_id,
                        "label": CORE_REGISTRY[core_id]["definition"][
                            "decision_description"
                        ],
                        "help": CORE_REGISTRY[core_id]["definition"][
                            "description"
                        ],
                    }
                    for core_id in candidate_ids
                ],
            }

        if question is None and question_id.startswith("core_condition:"):
            plan = intake_state.get("core_question_plan")
            if (
                plan is not None
                and plan["question_id"] == question_id
                and plan["core_id"] is not None
                and plan["condition_id"] is not None
            ):
                condition_id = plan["condition_id"]
                question = build_applicability_question(
                    plan["core_id"], condition_id
                )
                if plan["kind"] == "candidate_condition":
                    proposed_prompt = intake_state.get(
                        "core_question_prompt"
                    )
                    if proposed_prompt:
                        question["prompt"] = proposed_prompt
                    question["options"] = [
                        {
                            "option_id": "yes",
                            "label": "Ja — " + CONDITION_LABELS[condition_id],
                        },
                        {
                            "option_id": "no",
                            "label": (
                                "Nein — diese Voraussetzung trifft "
                                "nicht zu."
                            ),
                        },
                        {
                            "option_id": "user_unknown",
                            "label": "Noch nicht bekannt",
                        },
                    ]
                if plan["reason"] == "conflict":
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
    elif (
        core_selection is not None
        and core_selection["status"] == "needs_clarification"
        and not questions
        and (
            "unknown" in intake_state.get(
                "applicability_user_answers", {}
            ).values()
            or profile["primary_goal"]["status"] == "user_unknown"
        )
    ):
        message_type = "warning"
        message = (
            "Für eine Berechnung fehlen noch entscheidende Angaben: "
            + (
                "Das Optimierungsziel ist nicht bekannt. "
                if profile["primary_goal"]["status"] == "user_unknown"
                else ""
            )
            + (
                "Mindestens eine Modellvoraussetzung ist nicht bekannt. "
                if "unknown" in intake_state.get(
                    "applicability_user_answers", {}
                ).values()
                else ""
            )
            + "Es wird keine Annahme stillschweigend gesetzt. "
            "Sie können die Beschreibung später ergänzen."
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
