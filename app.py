import streamlit as st

from agentic_optimization.configuration.llm_config import load_llm_configuration
from agentic_optimization.graph import build_graph
from agentic_optimization.llm.factory import create_chat_model

# ── Seitenkonfiguration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic Optimization",
    page_icon="⚙️",
    layout="centered",
)

# ── Session State initialisieren ──────────────────────────────────────────────

if "step" not in st.session_state:
    st.session_state.step = "form"
if "form_data" not in st.session_state:
    st.session_state.form_data = {}
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None

# ── Fortschrittsanzeige ───────────────────────────────────────────────────────

STEPS = ["form", "freitext", "result"]
STEP_LABELS = ["① Formular", "② Beschreibung", "③ Ergebnis"]

step_index = STEPS.index(st.session_state.step) if st.session_state.step in STEPS else 1
cols = st.columns(3)
for i, (col, label) in enumerate(zip(cols, STEP_LABELS)):
    with col:
        if i < step_index:
            st.markdown(f"<p style='color:gray;text-align:center'>{label} ✓</p>", unsafe_allow_html=True)
        elif i == step_index:
            st.markdown(f"<p style='color:#1f6feb;font-weight:bold;text-align:center'>{label}</p>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:lightgray;text-align:center'>{label}</p>", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 1: Formular
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.step == "form":
    st.subheader("Strukturierte Eingabe")
    st.caption("Beantworte die Fragen so gut du kannst — fehlende Angaben kannst du im nächsten Schritt ergänzen.")

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
            "Welche Nebenbedingungen gibt es? (optional)",
            placeholder="z.B. Budget 10.000 €, Lagerkapazität 500 Einheiten",
        )
        submitted = st.form_submit_button("Weiter →", use_container_width=True, type="primary")

    if submitted:
        st.session_state.form_data = {
            "decision": decision or None,
            "objective": objective or None,
            "demand_type": demand_type or None,
            "constraints": constraints or None,
        }
        st.session_state.step = "freitext"
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 2: Freitext + Graph ausführen
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "freitext":
    st.subheader("Problembeschreibung")

    # Formularauswahl kurz anzeigen
    with st.expander("Deine Formularauswahl", expanded=False):
        for key, value in st.session_state.form_data.items():
            if value:
                st.markdown(f"**{key}:** {value}")

    raw_input = st.text_area(
        "Beschreibe dein Problem in eigenen Worten",
        placeholder="z.B. Wir verkaufen ein saisonales Produkt mit unsicherer Nachfrage und einer Lagerkapazität von max. 1.000 Einheiten.",
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
        with st.spinner("🤖 LLM analysiert dein Problem …"):
            llm = st.session_state.get("llm")
            if llm is None:
                try:
                    llm_config = load_llm_configuration()
                    llm = create_chat_model(llm_config)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                st.session_state.llm = llm
            app = build_graph(llm)
            initial_state = {
                "raw_input": raw_input,
                "form_data": st.session_state.form_data,
                "messages": [],
            }
            st.session_state.graph_state = app.invoke(initial_state)

        if st.session_state.graph_state.get("clarification_question"):
            st.session_state.step = "clarify"
        else:
            st.session_state.step = "result"
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 2b: Rückfrage (Clarification Loop)
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "clarify":
    st.subheader("Rückfrage")
    st.info(st.session_state.graph_state["clarification_question"])

    answer = st.text_area(
        "Deine Antwort",
        placeholder="Ergänze die fehlenden Angaben …",
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
        with st.spinner("🤖 LLM analysiert erneut …"):
            prev = st.session_state.graph_state
            combined = (prev.get("raw_input") or "") + "\n\nNachtrag: " + answer
            llm = st.session_state.get("llm")
            if llm is None:
                try:
                    llm_config = load_llm_configuration()
                    llm = create_chat_model(llm_config)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                st.session_state.llm = llm
            app = build_graph(llm)
            st.session_state.graph_state = app.invoke({
                "raw_input": combined,
                "form_data": st.session_state.form_data,
                "messages": [],
            })

        if st.session_state.graph_state.get("clarification_question"):
            st.rerun()
        else:
            st.session_state.step = "result"
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3: Ergebnis
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "result":
    gs = st.session_state.graph_state
    result = gs.get("result", {})
    reqs = result.get("requirements", {})
    model = result.get("selected_model", {})

    st.subheader("Ergebnis")
    st.success(result.get("summary", ""))

    # ── Erkannte Anforderungen ────────────────────────────────────────────────
    st.markdown("### 📋 Erkannte Anforderungen")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Entscheidung", reqs.get("decision", "–"))
        st.metric("Ziel", reqs.get("objective", "–"))
    with col2:
        st.metric("Nachfrage", reqs.get("demand_type", "–"))
        st.metric("Constraints", reqs.get("constraints", "–") or "–")

    st.divider()

    # ── Ausgewähltes Modell ───────────────────────────────────────────────────
    st.markdown("### 📐 Optimierungsmodell")
    st.markdown(f"**{model.get('name', '–')}**")
    st.caption(model.get("explanation", ""))

    # Mathematische Darstellung (Markdown, #36)
    model_type = model.get("type", "")
    decision_var = reqs.get("decision", "x")
    objective_text = reqs.get("objective", "")
    constraints_text = reqs.get("constraints", "keine angegeben")

    if model_type == "newsvendor":
        st.markdown("""
**Mathematische Struktur:**
```
maximize  E[ p · min(q, D) + s · max(q − D, 0) − c · q ]

subject to
    q ≥ 0
```
- `q` = Bestellmenge (Entscheidungsvariable)
- `D` = Nachfrage (unsicher)
- `p` = Verkaufspreis, `c` = Einkaufspreis, `s` = Restwert
""")

    elif model_type == "linear_programming":
        st.markdown(f"""
**Mathematische Struktur:**
```
maximize  c^T · x

subject to
    A · x ≤ b
    x ≥ 0
```
- `x` = {decision_var}
- Nebenbedingungen: {constraints_text}
""")

    else:
        st.markdown(f"""
**Mathematische Struktur:**
```
{"maximize" if "gewinn" in objective_text.lower() else "minimize"}  f(x)

subject to
    x ≥ 0
    {constraints_text}
```
""")

    st.divider()

    # ── Generierter PuLP-Code ─────────────────────────────────────────────────
    st.markdown("### 💻 Generierter PuLP-Code")
    st.code(result.get("generated_code", ""), language="python")

    st.divider()

    # ── Nächste Schritte ──────────────────────────────────────────────────────
    st.markdown("### 🔜 Nächste Schritte")
    for step in result.get("next_steps", []):
        st.markdown(f"- {step}")

    # ── Neu starten ───────────────────────────────────────────────────────────
    st.divider()
    if st.button("↺ Neu starten", use_container_width=True):
        for key in ["step", "form_data", "graph_state"]:
            del st.session_state[key]
        st.rerun()
