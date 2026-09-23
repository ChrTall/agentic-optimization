"""Koordination des modellneutralen Geschäftseinstiegs."""

from collections.abc import Collection
from copy import deepcopy
from typing import TypedDict

from agentic_optimization.agents.business_profile_agent import (
    BusinessProfileAgent,
)
from agentic_optimization.models.problem_profile import (
    BusinessProblemProfile,
    create_empty_business_problem_profile,
)
from agentic_optimization.ui.business_problem_form import (
    BusinessQuestion,
    get_relevant_business_questions,
)
from agentic_optimization.models.core_selector import (
    CoreSelectionDecision,
    select_model_core,
)
from agentic_optimization.state import (
    BusinessIntakeState,
)


class BusinessIntakePreparation(TypedDict):
    """Ergebnis der Vorbereitung des Geschäftseinstiegs."""

    profile: BusinessProblemProfile
    questions: list[BusinessQuestion]


def create_initial_business_intake_state(
) -> BusinessIntakeState:
    """Erzeugt den initialen Zustand des Geschäftseinstiegs."""

    profile = create_empty_business_problem_profile()
    questions = get_relevant_business_questions(
        profile
    )

    return {
        "profile": profile,
        "asked_question_ids": [],
        "pending_question_ids": [
            question["question_id"]
            for question in questions
        ],
        "pending_answers": [],
        "analyzed_raw_input": "",
        "analysis_error": None,
    }


def prepare_business_intake(
    *,
    profile: BusinessProblemProfile,
    profile_agent: BusinessProfileAgent,
    raw_input: str = "",
    source_id: str = "raw_input",
    asked_question_ids: Collection[str] = (),
) -> BusinessIntakePreparation:
    """Analysiert vorhandenen Text und bestimmt offene Fragen."""

    updated_profile = deepcopy(profile)

    if raw_input.strip():
        updated_profile = profile_agent(
            profile=updated_profile,
            raw_input=raw_input,
            source_id=source_id,
        )

    questions = get_relevant_business_questions(
        updated_profile,
        asked_question_ids=asked_question_ids,
    )

    core_selection = select_model_core(
        updated_profile
    )

    return {
        "profile": updated_profile,
        "questions": deepcopy(questions),
        "core_selection": core_selection,
    }