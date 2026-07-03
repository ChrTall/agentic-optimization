import streamlit as st

from agentic_optimization.configuration.llm_config import load_llm_configuration
from agentic_optimization.graph import build_graph
from agentic_optimization.llm.factory import create_chat_model
from agentic_optimization.visualization.markdown_renderer import render_model_markdown

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

STEPS = ["form", "freitext", "newsvendor_input", "result"]
STEP_LABELS = ["① Formular", "② Beschreibung", "③ Modellinput", "④ Ergebnis"]

step_index = STEPS.index(st.session_state.step) if st.session_state.step in STEPS else 1
cols = st.columns(len(STEP_LABELS))
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
        elif st.session_state.graph_state.get("model_config", {}).get("model_type") == "newsvendor":
            st.session_state.step = "newsvendor_input"
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
        elif st.session_state.graph_state.get("model_config", {}).get("model_type") == "newsvendor":
            st.session_state.step = "newsvendor_input"
            st.rerun()
        else:
            st.session_state.step = "result"
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3: Newsvendor-spezifische Eingaben
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "newsvendor_input":
    st.subheader("Newsvendor-Parameter")
    st.info("Es wurde ein Newsvendor-Modell erkannt. Bitte ergänze die modellspezifischen Eingaben.")

    with st.form("newsvendor_form"):
        selling_price = st.number_input(
            "Verkaufspreis pro Einheit",
            min_value=0.0,
            step=0.5,
            value=0.0,
        )

        purchase_cost = st.number_input(
            "Einkaufskosten pro Einheit",
            min_value=0.0,
            step=0.5,
            value=0.0,
        )

        salvage_value = st.number_input(
            "Restwert pro unverkaufter Einheit",
            min_value=0.0,
            step=0.5,
            value=0.0,
        )

        demand_scenarios = st.text_input(
            "Nachfrageszenarien",
            placeholder="z.B. 80, 100, 120, 140",
        )

        scenario_probabilities = st.text_input(
            "Wahrscheinlichkeiten der Szenarien",
            placeholder="z.B. 0.25, 0.25, 0.25, 0.25",
        )

        submitted = st.form_submit_button(
            "Ergebnis anzeigen →",
            use_container_width=True,
            type="primary",
        )

    def parse_number_list(value: str) -> list[float]:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    
    if submitted:
        try:
            parsed_demand_scenarios = parse_number_list(demand_scenarios)
            parsed_scenario_probabilities = parse_number_list(scenario_probabilities)

            if not parsed_demand_scenarios:
                st.error("Bitte gib mindestens ein Nachfrageszenario ein.")
                st.stop()

            if not parsed_scenario_probabilities:
                st.error("Bitte gib mindestens eine Wahrscheinlichkeit ein.")
                st.stop()

            if len(parsed_demand_scenarios) != len(parsed_scenario_probabilities):
                st.error("Die Anzahl der Nachfrageszenarien muss zur Anzahl der Wahrscheinlichkeiten passen.")
                st.stop()

            if abs(sum(parsed_scenario_probabilities) - 1.0) > 0.001:
                st.error("Die Wahrscheinlichkeiten müssen zusammen 1 ergeben.")
                st.stop()

            newsvendor_parameters = {
                "selling_price": selling_price,
                "purchase_cost": purchase_cost,
                "salvage_value": salvage_value,
                "demand_scenarios": parsed_demand_scenarios,
                "scenario_probabilities": parsed_scenario_probabilities,
            }

            st.session_state.form_data["model_parameters"] = {
                "newsvendor": newsvendor_parameters
            }

            if st.session_state.graph_state is not None:
                st.session_state.graph_state.setdefault("requirements", {})
                st.session_state.graph_state["requirements"]["model_parameters"] = {
                    "newsvendor": newsvendor_parameters
                }

                st.session_state.graph_state.setdefault("result", {})
                st.session_state.graph_state["result"].setdefault("requirements", {})
                st.session_state.graph_state["result"]["requirements"]["model_parameters"] = {
                    "newsvendor": newsvendor_parameters
                }
                st.session_state.graph_state["result"]["model_markdown"] = render_model_markdown(
                    st.session_state.graph_state.get("model_config", {}),
                    st.session_state.graph_state.get("requirements", {}),
                )

            st.session_state.step = "result"
            st.rerun()

        except ValueError:
            st.error(
                "Bitte gib Nachfrageszenarien und Wahrscheinlichkeiten als kommagetrennte Zahlen ein, "
                "z.B. 80, 100, 120 und 0.2, 0.5, 0.3."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4: Ergebnis
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "result":
    gs = st.session_state.graph_state
    result = gs.get("result", {})
    reqs = result.get("requirements", {})

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

    newsvendor_params = reqs.get("model_parameters", {}).get("newsvendor", {})

    if newsvendor_params:
        st.markdown("### 🧾 Erfasste Modellparameter")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Verkaufspreis",
                newsvendor_params.get("selling_price", "–"),
            )

        with col2:
            st.metric(
                "Einkaufskosten",
                newsvendor_params.get("purchase_cost", "–"),
            )

        with col3:
            st.metric(
                "Restwert",
                newsvendor_params.get("salvage_value", "–"),
            )

        demand_scenarios = newsvendor_params.get("demand_scenarios", [])
        scenario_probabilities = newsvendor_params.get("scenario_probabilities", [])

        if demand_scenarios and scenario_probabilities:
            st.markdown("#### Nachfrageszenarien")

            scenario_rows = [
                {
                    "Szenario": index + 1,
                    "Nachfrage": f"{demand:g}",
                    "Wahrscheinlichkeit": f"{probability:.0%}",
                }
                for index, (demand, probability) in enumerate(
                    zip(demand_scenarios, scenario_probabilities)
                )
            ]

            st.table(scenario_rows)

    st.divider()

    # ── Ausgewähltes Modell ───────────────────────────────────────────────────
    st.markdown("### 📐 Optimierungsmodell")
    st.markdown(result.get("model_markdown", "Keine Modellvisualisierung verfügbar."))

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
