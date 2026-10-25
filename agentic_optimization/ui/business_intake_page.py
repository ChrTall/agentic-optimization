"""Streamlit-Seite für den modellneutralen Geschäftseinstieg."""

from typing import TypedDict

import streamlit as st

from agentic_optimization.state import (
    BusinessIntakeState,
    BusinessQuestionAnswer,
)
from agentic_optimization.ui.business_intake_view_builder import (
    build_business_intake_view_data,
)
from agentic_optimization.ui.business_problem_form import (
    BusinessQuestionOption,
)


class BusinessIntakePageSubmission(TypedDict):
    """Vom neuen Einstieg übermittelte Nutzereingaben."""

    raw_input: str
    answers: list[BusinessQuestionAnswer]


def format_business_question_option(
    option: BusinessQuestionOption,
) -> str:
    """Formatiert eine Antwortoption verständlich."""

    label = option["label"]
    help_text = option.get("help")

    if help_text:
        return f"{label} — {help_text}"

    return label


def render_business_intake_page(
    intake_state: BusinessIntakeState,
    *,
    raw_input_value: str = "",
    show_questions: bool = True,
) -> BusinessIntakePageSubmission | None:
    """
    Rendert den adaptiven Geschäftseinstieg.

    None bedeutet, dass die Seite noch nicht abgesendet wurde.
    """

    view = build_business_intake_view_data(
        intake_state
    )
    questions = (
        view["questions"]
        if show_questions
        else []
    )

    st.subheader("Ihre Bestellsituation")
    st.write(
        "Beschreiben Sie Ihre Situation im Chat. Alternativ können Sie "
        "die strukturierten Fragen im Formular beantworten. Weitere Angaben "
        "und Korrekturen können Sie jederzeit im Chat ergänzen."
    )

    if view["message_type"] == "error":
        st.error(view["message"])
    elif view["message_type"] == "warning":
        st.warning(view["message"])
    elif view["message_type"] == "success":
        st.success(view["message"])

    rendered_answers: list[
        tuple[str, str | None]
    ] = []

    with st.form("business_intake_form"):
        if questions:
            st.markdown("### Kurze Rückfragen")

        for question in questions:
            options_by_id = {
                option["option_id"]: option
                for option in question["options"]
            }

            selected_option_id = st.radio(
                question["prompt"],
                options=list(options_by_id),
                index=None,
                format_func=(
                    lambda option_id, options=options_by_id: (
                        format_business_question_option(options[option_id])
                    )
                ),
                key=(
                    "business_question_"
                    f"{question['question_id']}"
                ),
            )

            rendered_answers.append(
                (
                    question["question_id"],
                    selected_option_id,
                )
            )

        submitted = st.form_submit_button(
            "Angaben auswerten",
            type="primary",
            width="stretch",
        )

    if not submitted:
        return None

    answers: list[BusinessQuestionAnswer] = [
        {
            "question_id": question_id,
            "option_id": option_id,
            "source_id": "business_intake_form",
        }
        for question_id, option_id in rendered_answers
        if option_id is not None
    ]

    return {
        "raw_input": raw_input_value,
        "answers": answers,
    }
