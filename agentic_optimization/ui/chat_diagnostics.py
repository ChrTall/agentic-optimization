"""Begrenzte, ausdrücklich aktivierte Diagnose im aktuellen Browsergespräch."""

from copy import deepcopy
import json

import streamlit as st

ENABLED_KEY = "chat_diagnostics_enabled"
RECORDS_KEY = "chat_diagnostics_records"


def intake_snapshot(graph):
    graph = graph or {}
    intake = graph.get("business_intake", {})
    return deepcopy({
        "workflow": graph.get("control"),
        "intake": {key: intake.get(key) for key in (
            "profile", "chat_verified_mode", "pending_question_ids", "core_selection",
            "applicability_user_answers", "applicability_text_findings", "core_question_plan",
        )},
    })


def start_diagnostic(message, step, graph, context):
    if not st.session_state.get(ENABLED_KEY):
        return None
    record = {
        "diagnostic_version": "intake-trace-1", "message": message, "step_before": step,
        "before": intake_snapshot(graph), "context": deepcopy(context), "events": [],
    }
    records = st.session_state.setdefault(RECORDS_KEY, [])
    records.append(record)
    del records[:-10]
    return record


def diagnostic_sink(record):
    if record is None:
        return None
    return lambda event: record["events"].append(deepcopy(event))


def render_diagnostics():
    with st.expander("Technische Chat-Diagnose"):
        st.checkbox("Diagnose für neue Nachrichten aktivieren", key=ENABLED_KEY)
        st.caption(
            "Erfasst die letzten zehn Nachrichten mit KI-Antworten und übernommenen Angaben "
            "nur in dieser Sitzung. Der Download enthält Ihre Geschäftsangaben. "
            "Es werden keine zusätzlichen KI-Aufrufe ausgelöst."
        )
        records = st.session_state.get(RECORDS_KEY, [])
        if records:
            st.download_button(
                "Chat-Diagnose herunterladen",
                data=json.dumps(records, ensure_ascii=False, indent=2, default=str),
                file_name="chat-diagnose.json", mime="application/json",
                key="download_chat_diagnostics",
            )
