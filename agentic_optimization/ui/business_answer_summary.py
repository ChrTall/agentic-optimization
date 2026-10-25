"""Nachvollziehbare und korrigierbare Übersicht des Geschäftsprofils."""

import streamlit as st

from agentic_optimization.ui.business_problem_form import BUSINESS_QUESTIONS


FIELD_LABELS = {
    "primary_goal": "Hauptziel", "recommendation": "Gesuchte Empfehlung",
    "procurement_pattern": "Beschaffung", "replenishment_during_period": "Nachbestellungen im Zeitraum",
    "inventory_review": "Bestandsprüfung", "product_scope": "Produkte",
}


def option_label(field_id, value):
    return next((o["label"] for o in BUSINESS_QUESTIONS[field_id]["options"]
                 if o.get("profile_value") == value), str(value))


def build_answer_summary(intake):
    rows = []
    for field_id, title in FIELD_LABELS.items():
        field = intake.get("profile", {}).get(field_id, {})
        status = field.get("status")
        if status == "confirmed":
            rows.append({"field_id": field_id, "title": title, "status": "Geklärt",
                         "label": option_label(field_id, field["value"])})
        elif status in {"conflict", "ambiguous"}:
            rows.append({"field_id": field_id, "title": title, "status": "Änderung noch zu klären",
                         "label": " / ".join(option_label(field_id, v)
                                             for v in field.get("candidate_values", []))})
    return rows


def render_answer_summary(intake):
    """Gibt (Einreichung, Bearbeitung aktiv) zurück; ändert keine Profilwerte still."""
    rows = build_answer_summary(intake)
    if rows:
        with st.expander("Bereits geklärt – prüfen und ändern", expanded=False):
            for row in rows:
                st.text(f"{row['title']}: {row['label']} · {row['status']}")
                if st.button(f"{row['title']} ändern", key=f"edit_business_{row['field_id']}"):
                    st.session_state["business_edit_question"] = row["field_id"]
                    st.session_state.pop(f"business_edit_value_{row['field_id']}", None)
                    st.rerun()
    qid = st.session_state.get("business_edit_question")
    if qid not in FIELD_LABELS:
        return None, False
    question = BUSINESS_QUESTIONS[qid]
    field = intake["profile"][qid]
    options = {o["option_id"]: o for o in question["options"]}
    current = next((index for index, o in enumerate(question["options"])
                    if field.get("value") is not None and o.get("profile_value") == field["value"]), None)
    with st.form("business_answer_edit"):
        selected = st.radio(question["prompt"], list(options), index=current,
                            format_func=lambda key: options[key]["label"], key=f"business_edit_value_{qid}")
        submitted = st.form_submit_button("Angabe aktualisieren", type="primary")
    if st.button("Bearbeitung abbrechen", key="cancel_business_edit"):
        st.session_state.pop("business_edit_question", None)
        st.rerun()
    if submitted and selected is not None:
        intake["pending_question_ids"] = list(dict.fromkeys([*intake.get("pending_question_ids", []), qid]))
        st.session_state.pop("business_edit_question", None)
        return [{"question_id": qid, "option_id": selected, "source_id": "business_clarification_form"}], True
    return None, True
