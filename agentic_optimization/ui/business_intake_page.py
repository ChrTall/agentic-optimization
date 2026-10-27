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
from agentic_optimization.i18n import t


class BusinessIntakePageSubmission(TypedDict):
    """Vom neuen Einstieg übermittelte Nutzereingaben."""

    raw_input: str
    answers: list[BusinessQuestionAnswer]


def format_business_question_option(
    option: BusinessQuestionOption,
) -> str:
    """Formatiert eine Antwortoption verständlich."""

    label = t(option["label"])
    help_text = t(option.get("help"))

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

    st.subheader(t("Ihre Bestellsituation"))
    st.write(
        t("Beschreiben Sie Ihre Situation einfach **im Chat** – in eigenen Worten. Was der Assistent daraus erkennt, erscheint oben zur Bestätigung. Offene Punkte können Sie auch hier direkt beantworten.")
    )

    if view["message_type"] == "error":
        st.error(view["message"])
    elif view["message_type"] == "warning":
        st.warning(view["message"])
    elif view["message_type"] == "success":
        st.success(view["message"])

    if not questions:
        # Nach einer fehlgeschlagenen Analyse kann sie hier wiederholt
        # werden; die Beschreibung stammt aus dem Chat.
        if intake_state.get("analysis_error") and raw_input_value.strip():
            if st.button(
                t("Erneut auswerten"),
                type="primary",
                width="stretch",
                key="business_intake_retry",
            ):
                return {"raw_input": raw_input_value, "answers": []}

        return None

    raw_input = raw_input_value

    st.markdown(t("### Kurze Rückfragen"))
    answers = render_business_questions(
        questions,
        key_prefix="business_question_",
        source_id="business_intake_form",
    )

    if answers is None:
        return None

    return {
        "raw_input": raw_input,
        "answers": answers,
    }


def _mark_answer_changed(changed_key: str) -> None:
    st.session_state[changed_key] = True


def render_business_questions(
    questions: list,
    *,
    key_prefix: str,
    source_id: str,
) -> list[BusinessQuestionAnswer] | None:
    """Rendert Rückfragen; eine Auswahl wird sofort übernommen.

    Liefert die Antworten nur in dem Durchlauf, in dem der Nutzer eine
    Auswahl geändert hat. So stellt der Chat direkt die nächste Frage,
    ohne dass ein Button geklickt werden muss.
    """

    changed_key = f"{key_prefix}_changed"
    rendered_answers: list[tuple[str, str | None]] = []

    for question in questions:
        options_by_id = {
            option["option_id"]: option
            for option in question["options"]
        }

        selected_option_id = st.radio(
            t(question["prompt"]),
            options=list(options_by_id),
            index=None,
            format_func=(
                lambda option_id, options=options_by_id: (
                    format_business_question_option(options[option_id])
                )
            ),
            key=f"{key_prefix}{question['question_id']}",
            on_change=_mark_answer_changed,
            args=(changed_key,),
        )

        rendered_answers.append((question["question_id"], selected_option_id))

    if not st.session_state.pop(changed_key, False):
        return None

    answers: list[BusinessQuestionAnswer] = [
        {
            "question_id": question_id,
            "option_id": option_id,
            "source_id": source_id,
        }
        for question_id, option_id in rendered_answers
        if option_id is not None
    ]

    return answers or None
