"""Streamlit-Seite für offene Fragen zum Geschäftsproblem."""

import streamlit as st

from agentic_optimization.state import (
    BusinessIntakeState,
    BusinessQuestionAnswer,
)
from agentic_optimization.ui.business_intake_page import (
    render_business_questions,
)
from agentic_optimization.ui.business_intake_view_builder import (
    build_business_intake_view_data,
)
from agentic_optimization.i18n import t


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

    st.subheader(t("Offene Angaben klären"))

    if view["message_type"] == "error":
        st.error(view["message"])
    elif view["message_type"] == "warning":
        st.warning(view["message"])

    if not view["questions"]:
        if view["message_type"] not in {"error", "warning"}:
            st.info(
                t("Derzeit sind keine konkreten Rückfragen offen.")
            )
        return None

    st.write(
        t("Für die passende Berechnung benötigen wir noch folgende Angaben:")
    )

    return render_business_questions(
        view["questions"],
        key_prefix="business_clarification_",
        source_id="business_clarification_form",
    )
