"""Streamlit-Seite für offene Fragen zum Geschäftsproblem."""

import streamlit as st

from agentic_optimization.state import (
    BusinessIntakeState,
    BusinessQuestionAnswer,
)
from agentic_optimization.ui.business_intake_page import (
    format_business_question_option,
)
from agentic_optimization.ui.business_intake_view_builder import (
    build_business_intake_view_data,
)


def render_business_clarification_page(
    intake_state: BusinessIntakeState,
) -> list[BusinessQuestionAnswer] | None:
    """
    Rendert ausschließlich offene Rückfragen.

    None bedeutet, dass das Formular nicht abgesendet wurde.
    """

    view = build_business_intake_view_data(
        intake_state
    )

    st.subheader("Offene Angaben klären")

    if view["message_type"] == "error":
        st.error(view["message"])
    elif view["message_type"] == "warning":
        st.warning(view["message"])

    if not view["questions"]:
        if view["message_type"] not in {"error", "warning"}:
            st.info(
                "Derzeit sind keine konkreten Rückfragen offen."
            )
        return None

    st.write(
        "Für die passende Berechnung benötigen wir noch "
        "folgende Angaben:"
    )

    rendered_answers: list[
        tuple[str, str | None]
    ] = []

    with st.form("business_clarification_form"):
        for question in view["questions"]:
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
                        format_business_question_option(
                            options[option_id]
                        )
                    )
                ),
                key=(
                    "business_clarification_"
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
            "Weiter",
            type="primary",
            width="stretch",
        )

    if not submitted:
        return None

    return [
        {
            "question_id": question_id,
            "option_id": option_id,
            "source_id": "business_clarification_form",
        }
        for question_id, option_id in rendered_answers
        if option_id is not None
    ]
