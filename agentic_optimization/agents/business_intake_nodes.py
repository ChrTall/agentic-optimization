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
    recover_missing_product_scope,
)
from agentic_optimization.agents.core_applicability_agent import (
    create_core_applicability_agent,
)
from agentic_optimization.agents.core_candidate_agent import (
    create_core_candidate_agent,
)
from agentic_optimization.agents.core_followup_agent import (
    create_core_followup_agent,
)
from agentic_optimization.agents.core_applicability_workflow import (
    plan_candidate_applicability,
)
from agentic_optimization.models.core_applicability import (
    CONDITION_GUIDANCE,
    CORE_CONDITIONS,
)
from agentic_optimization.models.core_candidate_selection import (
    select_from_core_candidates,
)
from agentic_optimization.models.core_question_planner import (
    plan_next_core_question,
)
from agentic_optimization.state import (
    WORKFLOW_BUSINESS_CASE_UNSUPPORTED,
    WORKFLOW_BUSINESS_CORE_SELECTED,
    WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED,
    WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
    GraphState,
    with_control,
)
from agentic_optimization.ui.business_problem_form import (
    BUSINESS_QUESTIONS,
    apply_business_question_answer,
    get_relevant_business_questions,
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
    candidate_agent = create_core_candidate_agent(llm)
    followup_agent = create_core_followup_agent(llm)

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
        chosen_core_id = intake_state.get("chosen_core_id")
        submitted_condition_answers: dict[str, str] = {}
        submitted_chosen_core_id = None

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

            if question_id == "core_choice":
                proposal = intake_state.get("candidate_proposal")
                allowed = {
                    item["core_id"]
                    for item in proposal["candidates"]
                } if proposal else set()
                if answer["option_id"] not in allowed:
                    raise ValueError("Ungültige Grundmodell-Auswahl.")
                chosen_core_id = answer["option_id"]
                submitted_chosen_core_id = chosen_core_id
            elif question_id.startswith("core_condition:"):
                condition_id = question_id.split(":", 1)[1]
                if answer["option_id"] not in {
                    "yes", "no", "user_unknown"
                }:
                    raise ValueError("Ungültige Antwort zur Modellvoraussetzung.")
                applicability_answers[condition_id] = (
                    "supported" if answer["option_id"] == "yes"
                    else "contradicted" if answer["option_id"] == "no"
                    else "unknown"
                )
                submitted_condition_answers[condition_id] = (
                    applicability_answers[condition_id]
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
        if raw_input_for_analysis.strip():
            # An earlier "not known" answer must not indefinitely block a
            # fresh clarification after the description gains information.
            reopened_conditions = {
                condition_id
                for condition_id, answer in intake_state.get(
                    "applicability_user_answers", {}
                ).items()
                if (
                    answer == "unknown"
                    and condition_id not in submitted_condition_answers
                )
            }
            for condition_id in reopened_conditions:
                applicability_answers.pop(condition_id, None)
            asked_question_ids = [
                question_id for question_id in asked_question_ids
                if question_id not in {
                    f"core_condition:{condition_id}"
                    for condition_id in reopened_conditions
                }
            ]
        # The chat app appends clarification messages to the existing text.
        # This is an extension of the same case, unlike replacing the text.
        additive_clarification = (
            isinstance(analyzed_raw_input, str)
            and bool(analyzed_raw_input.strip())
            and raw_input.startswith(f"{analyzed_raw_input.rstrip()}\n")
            and bool(raw_input[len(analyzed_raw_input.rstrip()) + 1:].strip())
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
        candidate_proposal = intake_state.get("candidate_proposal")
        candidate_active = False
        core_question_prompt = None
        if raw_input_for_analysis.strip():
            try:
                candidate_proposal = candidate_agent(evidence=[{
                    "source_id": "raw_input", "content": raw_input,
                }])
            except (TypeError, ValueError) as error:
                LOGGER.exception(
                    "Die Modellkandidaten konnten nicht geprüft werden: %s",
                    error,
                )
                return with_control(
                    {
                        **state,
                        "business_intake": {
                            **intake_state,
                            "profile": preparation["profile"],
                            "analysis_error": (
                                "Die Modellkandidaten konnten nicht "
                                "zuverlässig geprüft werden. Bitte "
                                "versuchen Sie es erneut."
                            ),
                            # The submitted answers have only been applied to
                            # local variables. Replay them on the next attempt.
                            "pending_answers": list(
                                intake_state.get("pending_answers", [])
                            ),
                        },
                    },
                    event=None,
                    workflow_status=WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED,
                )
            # A replacement invalidates earlier answers; an appended chat
            # clarification belongs to the same case and keeps them.
            if not additive_clarification:
                chosen_core_id = submitted_chosen_core_id
                applicability_answers = submitted_condition_answers
            candidate_active = True
        elif (
            candidate_proposal is not None
            and intake_state.get("candidate_analyzed_raw_input") == raw_input
        ):
            candidate_active = True

        if candidate_active:
            if (
                raw_input_for_analysis.strip()
                and "product_scope" in question_ids
            ):
                preparation["profile"] = recover_missing_product_scope(
                    llm,
                    profile=preparation["profile"],
                    raw_input=raw_input,
                    source_id="raw_input",
                )
                question_ids = [
                    question["question_id"]
                    for question in get_relevant_business_questions(
                        preparation["profile"],
                        asked_question_ids=asked_question_ids,
                    )
                ]
            # Diese Profilfragen beschaffen Daten, entscheiden aber nicht
            # mehr anhand einer festen Antwortkombination über den Kern.
            question_ids = [
                question_id for question_id in question_ids
                if (
                    question_id in {"product_scope", "primary_goal"}
                    or preparation["profile"][
                        BUSINESS_QUESTIONS[question_id]["target_field"]
                    ]["status"] in {"conflict", "ambiguous"}
                )
            ]
            core_selection = select_from_core_candidates(
                candidate_proposal,
                preparation["profile"],
                condition_answers=applicability_answers,
                chosen_core_id=chosen_core_id,
            )
        if (
            core_selection["status"] == "selected"
            and preparation["profile"]["primary_goal"]["status"]
            == "user_unknown"
        ):
            core_selection = {
                **core_selection,
                "status": "needs_clarification",
                "selected_core_id": None,
                "reason": (
                    "Das Optimierungsziel ist noch nicht bekannt. "
                    "Ohne Ziel kann keine Berechnung gestartet werden."
                ),
            }
        applicability_plan = None
        applicability_core_id = core_selection.get("selected_core_id")
        text_findings = dict(
            intake_state.get("applicability_text_findings", {})
        )
        text_evidence = list(
            intake_state.get("applicability_text_evidence", [])
        )

        if applicability_core_id is not None:
            if (
                intake_state.get("applicability_core_id")
                != applicability_core_id
                or intake_state.get("applicability_analyzed_raw_input")
                != raw_input
            ):
                if not candidate_active:
                    applicability_answers = {}
                text_findings = {}
                text_evidence = []
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
                        text_evidence = proposal["proposed_findings"]
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
                                    "asked_question_ids": asked_question_ids,
                                    "applicability_user_answers": applicability_answers,
                                    "chosen_core_id": chosen_core_id,
                                    "analyzed_raw_input": raw_input,
                                    "candidate_proposal": candidate_proposal,
                                    "candidate_analyzed_raw_input": (
                                        raw_input if candidate_active else None
                                    ),
                                    "core_selection": core_selection,
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

            # Keep answers about other candidates for subsequent selection
            # turns; only the selected core's answers enter its own check.
            core_applicability_answers = {
                condition_id: answer
                for condition_id, answer in applicability_answers.items()
                if condition_id in CORE_CONDITIONS[applicability_core_id]
            }
            applicability_plan = plan_candidate_applicability(
                applicability_core_id,
                preparation["profile"],
                text_findings,
                core_applicability_answers,
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

        core_question_plan = (
            plan_next_core_question(
                core_selection,
                applicability_plan=applicability_plan,
                applicability_core_id=applicability_core_id,
                condition_answers=applicability_answers,
                asked_question_ids=asked_question_ids,
            )
            if candidate_active or applicability_plan is not None
            else None
        )
        if core_question_plan is not None:
            question_ids.append(core_question_plan["question_id"])
            if core_question_plan["kind"] == "candidate_condition":
                if (
                    core_question_plan["question_id"] in pending_question_ids
                    and intake_state.get("core_question_plan")
                    == core_question_plan
                ):
                    core_question_prompt = intake_state.get(
                        "core_question_prompt"
                    )
                else:
                    try:
                        core_question_prompt = followup_agent(
                            core_id=core_question_plan["core_id"],
                            condition_id=core_question_plan["condition_id"],
                            raw_input=raw_input,
                        )
                    except (TypeError, ValueError) as error:
                        LOGGER.warning(
                            "Dynamische Rückfrage ungültig; "
                            "Standardfrage wird verwendet: %s",
                            error,
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
            "applicability_text_evidence": text_evidence,
            "applicability_user_answers": applicability_answers,
            "candidate_proposal": candidate_proposal,
            "candidate_analyzed_raw_input": (
                raw_input if candidate_active else None
            ),
            "chosen_core_id": chosen_core_id,
            "core_question_plan": core_question_plan,
            "core_question_prompt": core_question_prompt,
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
        "core_assumption_evidence": list(
            intake_state.get("applicability_text_evidence", [])
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
        "review_explanation",
        "review_explanation_source",
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
