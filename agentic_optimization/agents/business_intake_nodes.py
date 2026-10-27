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
from agentic_optimization.agents.core_applicability_workflow import (
    plan_candidate_applicability,
)
from agentic_optimization.models.core_applicability import (
    CONDITION_GUIDANCE,
    CONDITION_LABELS,
    CORE_CONDITIONS,
    PROFILE_CONDITIONS,
)
from agentic_optimization.models.problem_profile import (
    BusinessProblemProfile,
    PROFILE_FIELD_ALLOWED_VALUES,
    create_profile_field,
)
from agentic_optimization.models.core_candidate_selection import (
    compatible_registered_core_ids,
    select_from_core_candidates,
)
from agentic_optimization.models.core_selection_catalog import (
    CORE_SELECTION_RULES,
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
    reconcile_profile_with_condition_answers,
)
from agentic_optimization.i18n import t

LOGGER = logging.getLogger(__name__)

DEFAULT_SINGLE_PRODUCT_ID = "Produkt 1"
"""Produkt-ID, wenn aus der Beschreibung kein Produktname hervorgeht."""

REQUIREMENTS_PANEL_SOURCE = "requirements_panel"
"""Quelle für Bestätigungen und Korrekturen erkannter Angaben."""


def _answer_evidence(
    profile: BusinessProblemProfile,
    condition_answers: dict[str, str],
) -> dict[str, str] | None:
    """Macht direkte Antworten als belegbare, vorrangige Quelle sichtbar."""

    conditions = {
        condition_id: {
            "statement": CONDITION_LABELS[condition_id],
            "answer": answer,
        }
        for condition_id, answer in sorted(condition_answers.items())
        if condition_id in CONDITION_LABELS
        and answer in {"supported", "contradicted"}
    }
    overridden_fields = {
        field_id
        for mappings in PROFILE_CONDITIONS.values()
        for field_id, _expected, condition_id in mappings
        if condition_id in conditions
    }
    profile_answers = {
        field_id: field["value"]
        for field_id in sorted(PROFILE_FIELD_ALLOWED_VALUES)
        if field_id not in overridden_fields
        if (field := profile[field_id])["status"] == "confirmed"
        and any(
            item["source_type"] == "user_selection"
            and item["source_id"] != "chat_assistant"
            for item in field["evidence"]
        )
    }
    if not conditions and not profile_answers:
        return None
    return {
        "source_id": "business_answers",
        "content": json.dumps({
            "precedence": "Direct user answers override older contradictory text.",
            "condition_answers": conditions,
            "profile_answers": profile_answers,
        }, ensure_ascii=False, sort_keys=True),
    }


def _profile_question_covers_core_condition(
    question_ids: list[str], core_id: str | None, condition_id: str | None,
) -> bool:
    """Vermeidet gleichzeitige Fragen zu derselben registrierten Voraussetzung."""

    if core_id is None or condition_id is None:
        return False
    mapped_fields = {
        field_id
        for field_id, _expected, mapped_condition in PROFILE_CONDITIONS.get(
            core_id, ()
        )
        if mapped_condition == condition_id
    }
    return any(
        BUSINESS_QUESTIONS[question_id]["target_field"] in mapped_fields
        for question_id in question_ids
        if question_id in BUSINESS_QUESTIONS
    )


def create_business_intake_node(
    llm: BaseChatModel,
    *,
    product_name_agent: Callable[[str], str | None] | None = None,
) -> Callable[[GraphState], GraphState]:
    """Erzeugt den Knoten für den adaptiven Geschäftseinstieg.

    ``product_name_agent`` ermittelt optional den Produktnamen aus der
    Beschreibung, sobald ein Grundmodell feststeht.
    """

    profile_agent = create_business_profile_agent(llm)
    applicability_agent = create_core_applicability_agent(llm)
    candidate_agent = create_core_candidate_agent(llm)

    def prepare_business_intake_node(
        state: GraphState,
    ) -> GraphState:
        intake_state = state.get("business_intake")

        if intake_state is None:
            raise ValueError(
                t("Der Zustand des Geschäftseinstiegs fehlt.")
            )

        profile = intake_state.get("profile")

        if profile is None:
            raise ValueError(
                t("Das Geschäftsprofil fehlt.")
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
                    t("Die Geschäftsfrage '{question_id}' ist nicht mehr offen oder gehört nicht zum aktuellen Einstieg.", question_id=question_id)
                )

            if question_id == "core_choice":
                proposal = intake_state.get("candidate_proposal")
                allowed = {
                    item["core_id"]
                    for item in proposal["candidates"]
                } if proposal else set()
                if answer["option_id"] not in allowed:
                    raise ValueError(t("Ungültige Grundmodell-Auswahl."))
                chosen_core_id = answer["option_id"]
                submitted_chosen_core_id = chosen_core_id
            elif question_id.startswith("core_condition:"):
                condition_id = question_id.split(":", 1)[1]
                if answer["option_id"] not in {
                    "yes", "no", "user_unknown"
                }:
                    raise ValueError(t("Ungültige Antwort zur Modellvoraussetzung."))
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
                if answer["source_id"] != "chat_assistant":
                    field_id = BUSINESS_QUESTIONS[question_id][
                        "target_field"
                    ]
                    field = profile[field_id]
                    for mappings in PROFILE_CONDITIONS.values():
                        for mapped_field, expected, condition_id in mappings:
                            if mapped_field != field_id:
                                continue
                            condition_answer = (
                                "unknown" if field["status"] == "user_unknown"
                                else "supported" if field["value"] == expected
                                else "contradicted"
                            )
                            applicability_answers[condition_id] = (
                                condition_answer
                            )
                            submitted_condition_answers[condition_id] = (
                                condition_answer
                            )

            if question_id not in asked_question_ids:
                asked_question_ids.append(question_id)

        # Korrekturen an bereits erkannten Angaben (bestätigen, ändern,
        # verwerfen). Anders als Antworten betreffen sie keine offene Frage.
        for override in intake_state.get("profile_overrides", []):
            target_id = override["target_id"]
            action = override["action"]

            if target_id.startswith("condition:"):
                condition_id = target_id.split(":", 1)[1]
                condition_answer = (
                    "supported" if action == "confirm"
                    else "contradicted" if action == "discard"
                    else override.get("value")
                )
                if condition_answer not in {
                    "supported", "contradicted", "unknown"
                }:
                    raise ValueError(t("Ungültige Korrektur einer Annahme."))
                applicability_answers[condition_id] = condition_answer
                submitted_condition_answers[condition_id] = condition_answer
                continue

            field_id = target_id.split(":", 1)[1]

            if field_id not in profile:
                raise ValueError(t("Unbekannte Angabe '{field_id}'.", field_id=field_id))

            if action == "discard":
                profile = {
                    **profile,
                    field_id: create_profile_field(status="missing"),
                }
                asked_question_ids = [
                    question_id
                    for question_id in asked_question_ids
                    if question_id != field_id
                ]
                continue

            if action == "confirm":
                option_id = next(
                    (
                        option["option_id"]
                        for option in BUSINESS_QUESTIONS[field_id]["options"]
                        if option.get("profile_value", option["option_id"])
                        == profile[field_id]["value"]
                    ),
                    None,
                )
                if option_id is None:
                    raise ValueError(
                        t("Die Angabe '{field_id}' kann nicht bestätigt werden.", field_id=field_id)
                    )
            else:
                option_id = override.get("value")
            profile = apply_business_question_answer(
                profile,
                question_id=field_id,
                option_id=option_id,
                source_id=REQUIREMENTS_PANEL_SOURCE,
            )
            field = profile[field_id]
            for mappings in PROFILE_CONDITIONS.values():
                for mapped_field, expected, condition_id in mappings:
                    if mapped_field != field_id:
                        continue
                    condition_answer = (
                        "supported" if field["value"] == expected
                        else "contradicted"
                    )
                    applicability_answers[condition_id] = condition_answer
                    submitted_condition_answers[condition_id] = (
                        condition_answer
                    )

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
                # These submissions are not persisted yet. Replay them when
                # the profile analysis is retried.
                "pending_answers": list(
                    intake_state.get("pending_answers", [])
                ),
                "analysis_error": (
                    t("Die Beschreibung konnte nicht zuverlässig ausgewertet werden. Bitte starten Sie die Analyse erneut.")
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
        profile_question_ids = list(question_ids)
        core_selection = preparation["core_selection"]
        candidate_proposal = intake_state.get("candidate_proposal")
        if raw_input_for_analysis.strip() and not additive_clarification:
            chosen_core_id = submitted_chosen_core_id
            applicability_answers = submitted_condition_answers
        answer_evidence = _answer_evidence(
            preparation["profile"], applicability_answers
        )
        answer_context = answer_evidence["content"] if answer_evidence else ""
        candidate_answer_context = intake_state.get(
            "candidate_analyzed_answer_context", ""
        )
        candidate_active = False
        if raw_input_for_analysis.strip():
            try:
                candidate_proposal = candidate_agent(evidence=[
                    {"source_id": "raw_input", "content": raw_input},
                    *([answer_evidence] if answer_evidence else []),
                ])
                candidate_answer_context = answer_context
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
                                t("Die Modellkandidaten konnten nicht zuverlässig geprüft werden. Bitte versuchen Sie es erneut.")
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
                    or (
                        question_id == "recommendation"
                        and (
                            preparation["profile"]["procurement_pattern"][
                                "status"
                            ] in {"ambiguous", "conflict"}
                            or (
                                preparation["profile"]["recommendation"][
                                    "status"
                                ] == "inferred"
                                and preparation["profile"][
                                    "procurement_pattern"
                                ]["status"] == "missing"
                            )
                        )
                    )
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
            # A cached shortlist may be obsolete after direct answers rule
            # every candidate out. Reconsider all registered cores once for
            # this answer context before declaring the case unsupported.
            if (
                core_selection["status"] == "unsupported"
                and raw_input.strip()
                and answer_evidence is not None
                and candidate_answer_context != answer_context
            ):
                try:
                    candidate_proposal = candidate_agent(evidence=[
                        {"source_id": "raw_input", "content": raw_input},
                        answer_evidence,
                    ])
                except (TypeError, ValueError) as error:
                    LOGGER.exception(
                        "Die Modellkandidaten konnten nach den Antworten "
                        "nicht geprüft werden: %s", error
                    )
                    return with_control({
                        **state,
                        "business_intake": {
                            **intake_state,
                            "profile": preparation["profile"],
                            "pending_answers": list(
                                intake_state.get("pending_answers", [])
                            ),
                            "analysis_error": (
                                t("Die Modellkandidaten konnten nicht zuverlässig geprüft werden. Bitte versuchen Sie es erneut.")
                            ),
                        },
                    }, event=None,
                       workflow_status=WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED)
                candidate_answer_context = answer_context
                core_selection = select_from_core_candidates(
                    candidate_proposal,
                    preparation["profile"],
                    condition_answers=applicability_answers,
                    chosen_core_id=chosen_core_id,
                )
            if core_selection["status"] == "unsupported":
                compatible_core_ids = set(compatible_registered_core_ids(
                    preparation["profile"], applicability_answers,
                ))
                possible_core_ids = compatible_core_ids or set(
                    compatible_registered_core_ids(
                        preparation["profile"], applicability_answers,
                        ignore_inferred=True,
                    )
                )
                relevant_fields = {
                    field_id
                    for rule in CORE_SELECTION_RULES
                    if rule["core_id"] in possible_core_ids
                    for field_id in rule["compatible_profile_values"]
                } | {"primary_goal"}
                open_question_ids = [
                    question_id
                    for question_id in profile_question_ids
                    if (
                        (field_id := BUSINESS_QUESTIONS[question_id][
                            "target_field"
                        ]) in relevant_fields
                        and preparation["profile"][field_id]["status"]
                        in {"missing", "ambiguous", "conflict"}
                    )
                ] if possible_core_ids else []
                if possible_core_ids:
                    # An empty LLM shortlist is not evidence that a text
                    # inference was confirmed. Ask about a relevant inferred
                    # field even when some registered cores remain feasible.
                    inferred_questions = [
                        question_id
                        for question_id, question in BUSINESS_QUESTIONS.items()
                        if (
                            (field_id := question["target_field"])
                            == question_id
                            and field_id in relevant_fields
                            and preparation["profile"][field_id]["status"]
                            == "inferred"
                            and question_id not in asked_question_ids
                        )
                    ]
                    if inferred_questions:
                        open_question_ids.append(inferred_questions[0])
                if open_question_ids:
                    question_ids = list(dict.fromkeys([
                        *question_ids, *open_question_ids,
                    ]))
                    core_selection = {
                        **core_selection,
                        "status": "needs_clarification",
                        "reason": (
                            t("Für eine zuverlässige Modellauswahl fehlen noch Angaben zur Bestellsituation.")
                        ),
                    }
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
                    t("Das Optimierungsziel ist noch nicht bekannt. Ohne Ziel kann keine Berechnung gestartet werden.")
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
                    applicability_answers = dict(submitted_condition_answers)
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
                                    "candidate_analyzed_answer_context": (
                                        candidate_answer_context
                                    ),
                                    "core_selection": core_selection,
                                    "analysis_error": (
                                        t("Die Modellvoraussetzungen konnten nicht zuverlässig geprüft werden. Bitte versuchen Sie es erneut.")
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
                all_condition_answers=applicability_answers,
            )
            if applicability_plan["status"] == "rejected":
                core_selection = {
                    **core_selection,
                    "status": "unsupported",
                    "selected_core_id": None,
                    "reason": (
                        t("Eine Voraussetzung des Grundmodells ist ausdrücklich nicht erfüllt: ")
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
                    "reason": t("Eine Voraussetzung des Grundmodells ist offen."),
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
        if (
            core_question_plan is not None
            and _profile_question_covers_core_condition(
                question_ids,
                core_question_plan["core_id"],
                core_question_plan["condition_id"],
            )
        ):
            # The direct form answer is mapped to condition answers when
            # submitted, so the core question would ask the same thing twice.
            core_question_plan = None
        if question_ids:
            # First resolve open business-profile questions. Otherwise an
            # LLM-dependent candidate shortlist can add a different
            # model-specific question to the same initial form on each run.
            core_question_plan = None
        if core_question_plan is not None:
            question_ids.append(core_question_plan["question_id"])

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

        product_name = intake_state.get("product_name")
        product_name_input = intake_state.get(
            "product_name_analyzed_raw_input"
        )

        if (
            product_name_agent is not None
            and workflow_status == WORKFLOW_BUSINESS_CORE_SELECTED
            and product_name_input != raw_input
            and raw_input.strip()
        ):
            # Der Produktname dient bei einem einzelnen Produkt als
            # Produkt-ID, damit die Oberfläche nicht "product_1" zeigt.
            product_name = product_name_agent(raw_input)
            product_name_input = raw_input

        updated_intake_state = {
            **intake_state,
            "profile": preparation["profile"],
            "asked_question_ids": asked_question_ids,
            "pending_question_ids": question_ids,
            "pending_answers": [],
            "profile_overrides": [],
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
            "candidate_analyzed_answer_context": candidate_answer_context,
            "chosen_core_id": chosen_core_id,
            "core_question_plan": core_question_plan,
            "analysis_error": None,
            "product_name": product_name,
            "product_name_analyzed_raw_input": product_name_input,
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
            t("Der Zustand des Geschäftseinstiegs fehlt.")
        )

    profile = intake_state.get("profile")

    if profile is None:
        raise ValueError(
            t("Das Geschäftsprofil fehlt.")
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
            t("Es wurde noch kein Grundmodell eindeutig ausgewählt.")
        )

    condition_answers = dict(
        intake_state.get("applicability_user_answers", {})
    )
    model_profile = reconcile_profile_with_condition_answers(
        profile,
        core_id=core_selection["selected_core_id"],
        condition_answers=condition_answers,
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

    product_scope = model_profile["product_scope"]

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
                "product_id": (
                    intake_state.get("product_name")
                    or t(DEFAULT_SINGLE_PRODUCT_ID)
                ),
                "forecast": {},
                "facts": {},
            }
        ]

    evidence = [
        dict(item)
        for item in existing_modular_state.get("evidence", [])
        if item.get("source_id")
        not in {"raw_input", "business_intake_form", "business_answers"}
    ]

    raw_input = state.get("raw_input", "").strip()

    if raw_input:
        evidence.append(
            {
                "source_id": "raw_input",
                "content": raw_input,
            }
        )
    condition_evidence = _answer_evidence(model_profile, condition_answers)
    if condition_evidence is not None and any(
        answer in {"supported", "contradicted"}
        for answer in condition_answers.values()
    ):
        evidence.append(condition_evidence)

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
            (field := model_profile[field_name])["status"]
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
            model_profile,
            raw_input=state.get(
                "raw_input",
                "",
            ),
            condition_answers=condition_answers,
        )
    )

    return {
        **state,
        "requirements": requirements,
        "modular_model": modular_state,
    }
