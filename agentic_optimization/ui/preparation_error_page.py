"""Bedienbare Wiederherstellung bei technischen Vorbereitungsfehlern."""

import streamlit as st


def render_preparation_error_page(details):
    st.subheader("Die Vorbereitung konnte nicht abgeschlossen werden")
    st.error(details.get("message") or "Die automatische Verarbeitung wurde unterbrochen.")
    st.write("Ihre bisherigen Angaben bleiben erhalten. Sie können die Verarbeitung "
             "erneut versuchen oder Ihre Angaben im Chat ergänzen.")
    st.caption("Es liegt ein technischer Verarbeitungsfehler vor. Daraus lässt sich "
               "nicht ableiten, ob ein passendes Modell verfügbar ist.")
    if st.button("Vorbereitung erneut versuchen", type="primary", width="stretch",
                 key="retry_preparation"):
        return "retry"
    if st.button("Angaben im Chat ergänzen", width="stretch", key="edit_failed_description"):
        return "edit"
    st.caption("Beim kompletten Neustart werden alle bisherigen Angaben, Antworten "
               "und der Chat dieser Optimierung zurückgesetzt.")
    if st.button("Komplett neu starten", width="stretch", key="restart_failed_preparation"):
        return "restart"
    with st.expander("Technische Details", expanded=False):
        st.text("Fehlerkennung: " + details.get("reference", "nicht verfügbar"))
        st.text("Verarbeitungsschritt: " + details.get("stage_label", "Vorbereitung"))
        st.text("Fehlerart: " + details.get("kind", "unknown"))
    return None
