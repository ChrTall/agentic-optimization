"""LangGraph-Knoten für den modellneutralen Geschäftseinstieg."""

import json
import logging
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from agentic_optimization.agents.business_intake import (
    prepare_business_intake,
)
from agentic_optimization.agents.business_profile_agent import (
    create_business_profile_agent,
)
from agentic_optimization.agents.core_applicability_agent import (
    create_core_applicability_agent,
)
from agentic_optimization.agents.core_applicability_workflow import (
    plan_candidate_applicability,
)
from agentic_optimization.models.core_applicability import CONDITION_GUIDANCE
from agentic_optimization.state import (
    WORKFLOW_BUSINESS_CASE_UNSUPPORTED,
    WORKFLOW_BUSINESS_CORE_SELECTED,
    WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED,
    WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
    GraphState,
    with_control,
)
from agentic_optimization.ui.business_problem_form import (
    apply_business_question_answer,
)
from agentic_optimization.agents.business_profile_adapter import (
    build_requirements_from_business_profile,
)

LOGGER = logging.getLogger(__name__)

def create_business_intake_node(
    llm: BaseChatModel,
) -> Callable[[GraphState], GraphState]:
    """Erzeugt den Knoten für den adaptiven Geschäftseinstieg."""

    profile_agent = create_business_profile_agent(llm)
    applicability_agent = create_core_applicability_agent(llm)

    def prepare_business_intake_node(
        state: GraphState,
    ) -> GraphState:
        intake_state = state.get("business_intake")

        if intake_state is None:
            raise ValueError(
                "Der Zustand des Geschäftseinstiegs fehlt."
            )

        profile = intake_state.get("profile")

        if profile is None:
            raise ValueError(
                "Das Geschäftsprofil fehlt."
            )

        asked_question_ids = list(
            intake_state.get(
                "asked_question_ids",
                [],
            )
        )
        pending_question_ids = set(
            intake_state.get(
                "pending_question_ids",
                [],
            )
        )
        applicability_answers = dict(
            intake_state.get("applicability_user_answers", {})
        )

        for answer in intake_state.get(
            "pending_answers",
            [],
        ):
            question_id = answer["question_id"]

            if question_id not in pending_question_ids:
                raise ValueError(
                    f"Die Geschäftsfrage '{question_id}' "
                    "ist nicht mehr offen oder gehört "
                    "nicht zum aktuellen Einstieg."
                )

            if question_id.startswith("core_condition:"):
                condition_id = question_id.split(":", 1)[1]
                if answer["option_id"] not in {"yes", "no"}:
                    raise ValueError("Ungültige Antwort zur Modellvoraussetzung.")
                applicability_answers[condition_id] = (
                    "supported" if answer["option_id"] == "yes"
                    else "contradicted"
                )
            else:
                profile = apply_business_question_answer(
                    profile,
                    question_id=question_id,
                    option_id=answer["option_id"],
                    source_id=answer["source_id"],
                )

            if question_id not in asked_question_ids:
                asked_question_ids.append(question_id)

        raw_input = state.get("raw_input", "")
        analyzed_raw_input = intake_state.get(
            "analyzed_raw_input"
        )

        raw_input_for_analysis = (
            ""
            if raw_input == analyzed_raw_input
            else raw_input
        )

        try:
            preparation = prepare_business_intake(
                profile=profile,
                profile_agent=profile_agent,
                raw_input=raw_input_for_analysis,
                source_id="raw_input",
                asked_question_ids=asked_question_ids,
            )
        except (TypeError, ValueError) as error:
            LOGGER.exception(
                "Die Analyse des Geschäftsprofils ist "
                "fehlgeschlagen: %s",
                error,
            )

            updated_intake_state = {
                **intake_state,
                "profile": profile,
                "asked_question_ids": asked_question_ids,
                "pending_answers": [],
                "analysis_error": (
                    "Die Beschreibung konnte nicht "
                    "zuverlässig ausgewertet werden. "
                    "Bitte starten Sie die Analyse erneut."
                ),
                "pending_question_ids": list(
                    intake_state.get(
                        "pending_question_ids",
                        [],
                    )
                ),
            }
            updated_intake_state.pop(
                "core_selection",
                None,
            )

            return with_control(
                {
                    **state,
                    "business_intake": updated_intake_state,
                },
                event=None,
                workflow_status=(
                    WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED
                ),
            )

        question_ids = [
            question["question_id"]
            for question in preparation["questions"]
        ]
        core_selection = preparation["core_selection"]
        applicability_plan = None
        applicability_core_id = core_selection.get("selected_core_id")
        text_findings = dict(
            intake_state.get("applicability_text_findings", {})
        )

        if applicability_core_id is not None:
            if (
                intake_state.get("applicability_core_id")
                != applicability_core_id
                or intake_state.get("applicability_analyzed_raw_input")
                != raw_input
            ):
                applicability_answers = {}
                text_findings = {}
                if raw_input.strip():
                    try:
                        proposal = applicability_agent(
                            core_id=applicability_core_id,
                            evidence=[{
                                "source_id": "raw_input",
                                "content": raw_input,
                            }],
                        )
                        text_findings = proposal["findings"]
                    except (TypeError, ValueError) as error:
                        LOGGER.exception(
                            "Die Prüfung der Modellvoraussetzungen "
                            "ist fehlgeschlagen: %s", error
                        )
                        return with_control(
                            {
                                **state,
                                "business_intake": {
                                    **intake_state,
                                    "profile": preparation["profile"],
                                    "analysis_error": (
                                        "Die Modellvoraussetzungen konnten "
                                        "nicht zuverlässig geprüft werden. "
                                        "Bitte versuchen Sie es erneut."
                                    ),
                                    "pending_answers": [],
                                    "pending_question_ids": question_ids,
                                },
                            },
                            event=None,
                            workflow_status=(
                                WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED
                            ),
                        )

            applicability_plan = plan_candidate_applicability(
                applicability_core_id,
                preparation["profile"],
                text_findings,
                applicability_answers,
            )
            if applicability_plan["status"] == "rejected":
                core_selection = {
                    **core_selection,
                    "status": "unsupported",
                    "selected_core_id": None,
                    "reason": (
                        "Eine Voraussetzung des Grundmodells ist "
                        "ausdrücklich nicht erfüllt: "
                        + " ".join(
                            CONDITION_GUIDANCE[condition_id]
                            for condition_id in applicability_plan[
                                "contradicted"
                            ]
                        )
                    ),
                }
            elif applicability_plan["status"] == "needs_clarification":
                core_selection = {
                    **core_selection,
                    "status": "needs_clarification",
                    "selected_core_id": None,
                    "reason": "Eine Voraussetzung des Grundmodells ist offen.",
                }
                question_ids.append(
                    "core_condition:"
                    + str(applicability_plan["question_condition_id"])
                )

        if core_selection["status"] == "unsupported":
            workflow_status = (
                WORKFLOW_BUSINESS_CASE_UNSUPPORTED
            )
        elif (
            question_ids
            or core_selection["status"]
            == "needs_clarification"
        ):
            workflow_status = (
                WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
            )
        else:
            workflow_status = (
                WORKFLOW_BUSINESS_CORE_SELECTED
            )

        updated_intake_state = {
            **intake_state,
            "profile": preparation["profile"],
            "asked_question_ids": asked_question_ids,
            "pending_question_ids": question_ids,
            "pending_answers": [],
            "analyzed_raw_input": raw_input,
            "core_selection": core_selection,
            "applicability_plan": applicability_plan,
            "applicability_core_id": applicability_core_id,
            "applicability_analyzed_raw_input": raw_input,
            "applicability_text_findings": text_findings,
            "applicability_user_answers": applicability_answers,
            "analysis_error": None,
        }

        return with_control(
            {
                **state,
                "business_intake": updated_intake_state,
            },
            event=None,
            workflow_status=workflow_status,
        )

    return prepare_business_intake_node


def initialize_modular_core_node(
    state: GraphState,
) -> GraphState:
    """Übernimmt den ausgewählten Kern in die Modellpipeline."""

    intake_state = state.get("business_intake")

    if intake_state is None:
        raise ValueError(
            "Der Zustand des Geschäftseinstiegs fehlt."
        )

    profile = intake_state.get("profile")

    if profile is None:
        raise ValueError(
            "Das Geschäftsprofil fehlt."
        )

    core_selection = intake_state.get(
        "core_selection"
    )

    if (
        core_selection is None
        or core_selection["status"] != "selected"
        or core_selection["selected_core_id"] is None
    ):
        raise ValueError(
            "Es wurde noch kein Grundmodell eindeutig "
            "ausgewählt."
        )

    existing_modular_state = state.get(
        "modular_model",
        {},
    )

    products = list(
        existing_modular_state.get(
            "products",
            [],
        )
    )

    product_scope = profile["product_scope"]

    if (
        not products
        and product_scope["status"]
        in {
            "inferred",
            "confirmed",
        }
        and product_scope["value"]
        == "single_product"
    ):
        products = [
            {
                "product_id": "product_1",
                "forecast": {},
                "facts": {},
            }
        ]

    evidence = [
        dict(item)
        for item in existing_modular_state.get("evidence", [])
        if item.get("source_id")
        not in {"raw_input", "business_intake_form"}
    ]

    raw_input = state.get("raw_input", "").strip()

    if raw_input:
        evidence.append(
            {
                "source_id": "raw_input",
                "content": raw_input,
            }
        )

    form_values = {
        field_name: field["value"]
        for field_name in (
            "recommendation",
            "procurement_pattern",
            "replenishment_during_period",
            "inventory_review",
            "product_scope",
            "primary_goal",
        )
        if (
            (field := profile[field_name])["status"]
            == "confirmed"
            and any(
                source["source_type"] == "user_selection"
                for source in field["evidence"]
            )
        )
    }

    if form_values:
        evidence.append(
            {
                "source_id": "business_intake_form",
                "content": json.dumps(
                    form_values,
                    ensure_ascii=False,
                ),
            }
        )

    modular_state = {
        **existing_modular_state,
        "core_id": core_selection["selected_core_id"],
        "pending_core_assumptions": list(
            (intake_state.get("applicability_plan") or {}).get(
                "pending_confirmation", []
            )
        ),
        "core_assumptions_confirmed": False,
        "products": products,
        "global_facts": dict(
            existing_modular_state.get(
                "global_facts",
                {},
            )
        ),
        "evidence": evidence,
        "pending_input_submissions": [],
        "review_confirmed": False,
        "optimization_result": None,
        "preparation_error": None,
    }

    for field_name in (
        "preliminary_proposal",
        "proposal",
        "specification",
        "validation",
    ):
        modular_state.pop(field_name, None)

    requirements = (
        build_requirements_from_business_profile(
            profile,
            raw_input=state.get(
                "raw_input",
                "",
            ),
        )
    )

    return {
        **state,
        "requirements": requirements,
        "modular_model": modular_state,
    }
