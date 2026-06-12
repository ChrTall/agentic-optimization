import streamlit as st
from agentic_optimization.graph import build_graph

# ── Seitenkonfiguration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic Optimization",
    page_icon="⚙️",
    layout="centered",
)

st.title("⚙️ Agentic Optimization")
st.caption("Beschreibe dein Optimierungsproblem – das System wählt ein passendes Modell.")

# ── Session State initialisieren ──────────────────────────────────────────────

if "step" not in st.session_state:
    st.session_state.step = "form"          # form → freitext → result / clarify
if "form_data" not in st.session_state:
    st.session_state.form_data = {}
if "result_state" not in st.session_state:
    st.session_state.result_state = None

# ── Schritt 1: Strukturiertes Formular ───────────────────────────────────────

if st.session_state.step == "form":
    st.subheader("Schritt 1 · Strukturierte Eingabe")

    with st.form("input_form"):
        decision = st.selectbox(
            "Welche Entscheidung soll optimiert werden?",
            ["", "Bestellmenge", "Produktionsmenge", "Lagerbestand", "Andere"],
        )
        objective = st.selectbox(
            "Was ist das Ziel der Optimierung?",
            ["", "Gewinn maximieren", "Kosten minimieren", "Servicelevel erreichen", "Andere"],
        )
        demand_type = st.selectbox(
            "Wie verhält sich die Nachfrage?",
            ["", "Sicher / bekannt", "Unsicher / saisonal", "Historisch geschätzt", "Andere"],
        )
        constraints = st.text_input(
            "Welche Nebenbedingungen gibt es?",
            placeholder="z.B. Budget 10.000 €, Lagerkapazität 500 Einheiten",
        )

        submitted = st.form_submit_button("Weiter →", use_container_width=True)

    if submitted:
        st.session_state.form_data = {
            "decision":    decision    or None,
            "objective":   objective   or None,
            "demand_type": demand_type or None,
            "constraints": constraints or None,
        }
        st.session_state.step = "freitext"
        st.rerun()

# ── Schritt 2: Freitextbeschreibung ──────────────────────────────────────────

elif st.session_state.step == "freitext":
    st.subheader("Schritt 2 · Problembeschreibung")

    # Formularauswahl anzeigen
    with st.expander("Deine Formularauswahl", expanded=False):
        for key, value in st.session_state.form_data.items():
            if value:
                st.markdown(f"**{key}:** {value}")

    raw_input = st.text_area(
        "Beschreibe dein Problem in eigenen Worten",
        placeholder=(
            "z.B. Wir verkaufen ein saisonales Produkt mit unsicherer Nachfrage. "
            "Die Lagerkapazität beträgt maximal 1.000 Einheiten."
        ),
        height=160,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück"):
            st.session_state.step = "form"
            st.rerun()
    with col2:
        analyze = st.button("Problem analysieren →", use_container_width=True, type="primary")

    if analyze:
        with st.spinner("Anforderungen werden analysiert …"):
            app = build_graph()
            initial_state = {
                "raw_input":  raw_input,
                "form_data":  st.session_state.form_data,
                "messages":   [],
            }
            st.session_state.result_state = app.invoke(initial_state)

        if st.session_state.result_state.get("clarification_question"):
            st.session_state.step = "clarify"
        else:
            st.session_state.step = "result"
        st.rerun()

# ── Schritt 3a: Rückfrage ─────────────────────────────────────────────────────

elif st.session_state.step == "clarify":
    st.subheader("Schritt 3 · Rückfragen")
    st.info(st.session_state.result_state["clarification_question"])

    answer = st.text_area(
        "Deine Antwort",
        placeholder="Beantworte die fehlenden Punkte …",
        height=120,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück"):
            st.session_state.step = "freitext"
            st.rerun()
    with col2:
        retry = st.button("Erneut analysieren →", use_container_width=True, type="primary")

    if retry and answer:
        with st.spinner("Analyse wird wiederholt …"):
            # Antwort des Users als zusätzlicher Freitext übergeben
            prev = st.session_state.result_state
            combined_raw = (prev.get("raw_input") or "") + "\n\nNachtrag: " + answer
            app = build_graph()
            new_state = {
                "raw_input": combined_raw,
                "form_data": st.session_state.form_data,
                "messages":  [],
            }
            st.session_state.result_state = app.invoke(new_state)

        if st.session_state.result_state.get("clarification_question"):
            st.rerun()
        else:
            st.session_state.step = "result"
            st.rerun()

# ── Schritt 3b: Ergebnis ──────────────────────────────────────────────────────

elif st.session_state.step == "result":
    rs = st.session_state.result_state
    result = rs.get("result", {})

    st.subheader("Ergebnis")
    st.success(result.get("summary", ""))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Erkannte Anforderungen**")
        reqs = result.get("requirements", {})
        for label, key in [
            ("Entscheidung", "decision"),
            ("Ziel",         "objective"),
            ("Nachfrage",    "demand_type"),
            ("Constraints",  "constraints"),
        ]:
            value = reqs.get(key) or "–"
            st.markdown(f"- **{label}:** {value}")

    with col2:
        st.markdown("**Ausgewähltes Modell**")
        model = result.get("selected_model", {})
        st.markdown(f"**{model.get('name', '–')}**")
        st.caption(model.get("explanation", ""))

    st.divider()
    st.markdown("**Nächste Schritte**")
    for step in result.get("next_steps", []):
        st.markdown(f"- {step}")

    with st.expander("Generierter PuLP-Code anzeigen"):
        st.code(result.get("generated_code", ""), language="python")

    if st.button("↺ Neu starten"):
        for key in ["step", "form_data", "result_state"]:
            del st.session_state[key]
        st.rerun()
