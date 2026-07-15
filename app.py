import streamlit as st

from agentic_optimization.configuration.llm_config import load_llm_configuration
from agentic_optimization.graph import build_graph
from agentic_optimization.llm.factory import create_chat_model
from agentic_optimization.state import (
    WORKFLOW_MODEL_INPUTS_INVALID,
    EVENT_NEW_PROBLEM,
    EVENT_GENERAL_CLARIFICATION_SUBMITTED,
    EVENT_MODEL_INPUTS_SUBMITTED,
    WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_READY_FOR_RESULT,
    get_workflow_status,
)

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

STEPS = ["form", "freitext", "model_input_form", "result"]
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


def parse_constraints(value: str) -> list[str]:
    return [
        item.strip()
        for item in value.splitlines()
        if item.strip()
    ]


def route_ui_from_graph_state(graph_state: dict) -> str:
    """
    Übersetzt den fachlichen Workflow-Status aus dem Graph
    in den nächsten Streamlit-Schritt.

    Der Graph entscheidet, was fachlich als Nächstes nötig ist.
    Diese Funktion entscheidet, welche UI-Seite dafür angezeigt wird.
    """

    status = get_workflow_status(graph_state)

    if status == WORKFLOW_NEEDS_GENERAL_CLARIFICATION:
        return "clarify"

    if status in {
        WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
        WORKFLOW_MODEL_INPUTS_INVALID,
    }:
        return "model_input_form"

    if status == WORKFLOW_READY_FOR_RESULT:
        return "result"

    return "freitext"

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
        constraints = st.text_area(
            "Welche Nebenbedingungen gibt es? (optional)",
            placeholder=(
                "Eine Nebenbedingung pro Zeile, z.B.\n"
                "Budget 10.000 €\n"
                "Lagerkapazität 500 Einheiten"
            ),
            height=120,
        )
        submitted = st.form_submit_button("Weiter →", use_container_width=True, type="primary")

    if submitted:
        st.session_state.form_data = {
            "decision": decision or None,
            "objective": objective or None,
            "demand_type": demand_type or None,
            "constraints": parse_constraints(constraints) if constraints else [],
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
                if isinstance(value, list):
                    st.markdown(f"**{key}:**")
                    for item in value:
                        st.markdown(f"- {item}")
                else:
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
                "control": {
                    "event": EVENT_NEW_PROBLEM,
                },
                "raw_input": raw_input,
                "form_data": st.session_state.form_data,
                "messages": [],
            }
            st.session_state.graph_state = app.invoke(initial_state)

        st.session_state.step = route_ui_from_graph_state(
            st.session_state.graph_state
        )
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
            prev = st.session_state.graph_state or {}
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
                **prev,
                "control": {
                    **prev.get("control", {}),
                    "event": EVENT_GENERAL_CLARIFICATION_SUBMITTED,
                },
                "raw_input": combined,
                "form_data": st.session_state.form_data,
                "messages": [],
            })

        st.session_state.step = route_ui_from_graph_state(
            st.session_state.graph_state
        )
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3: Newsvendor-spezifische Eingaben
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "model_input_form":
    st.subheader("Modellspezifische Eingaben")

    graph_state = st.session_state.graph_state or {}
    selected_model = graph_state.get("selected_model")

    validation_errors = graph_state.get("model_input_validation_errors", [])
    for error in validation_errors:
        st.error(error)

    if selected_model != "newsvendor":
        st.warning("Für dieses Modell gibt es noch kein spezifisches Eingabeformular.")
        st.stop()

    st.info(
        "Es wurde ein Newsvendor-Modell erkannt. "
        "Bitte ergänze die modellspezifischen Eingaben."
    )

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
            "Eingaben prüfen →",
            use_container_width=True,
            type="primary",
        )

    def parse_number_list(value: str) -> list[float]:
        return [
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        ]

    if submitted:
        try:
            parsed_demand_scenarios = parse_number_list(demand_scenarios)
            parsed_scenario_probabilities = parse_number_list(
                scenario_probabilities
            )

            newsvendor_inputs = {
                "selling_price": selling_price,
                "purchase_cost": purchase_cost,
                "salvage_value": salvage_value,
                "demand_scenarios": parsed_demand_scenarios,
                "scenario_probabilities": parsed_scenario_probabilities,
            }

            prev_state = st.session_state.graph_state or {}

            next_state = {
                **prev_state,
                "control": {
                    **prev_state.get("control", {}),
                    "event": EVENT_MODEL_INPUTS_SUBMITTED,
                },
                "model_specific_inputs": {
                    **prev_state.get("model_specific_inputs", {}),
                    "newsvendor": newsvendor_inputs,
                },
            }

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
            st.session_state.graph_state = app.invoke(next_state)

            st.session_state.step = route_ui_from_graph_state(
                st.session_state.graph_state
            )

            st.rerun()

        except ValueError:
            st.error(
                "Bitte gib Nachfrageszenarien und Wahrscheinlichkeiten als "
                "kommagetrennte Zahlen ein, z.B. 80, 100, 120 und 0.2, 0.5, 0.3."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4: Ergebnis
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "result":
    gs = st.session_state.graph_state or {}
    result = gs.get("result", {})
    reqs = result.get("requirements", {})
    selected_model = result.get("selected_model", {})
    selected_model_type = selected_model.get("type")
    selected_model_inputs = result.get("selected_model_inputs", {})

    st.subheader("Ergebnis")
    st.success(result.get("summary", ""))

    # ── Erkannte Anforderungen ────────────────────────────────────────────────
    st.markdown("### 📋 Erkannte Anforderungen")

    constraints_list = reqs.get("constraints", [])

    if isinstance(constraints_list, str):
        constraints_list = [constraints_list]

    st.markdown("#### Problemstruktur")
    st.markdown(f"- **Entscheidung:** {reqs.get('decision', '–')}")
    st.markdown(f"- **Ziel:** {reqs.get('objective', '–')}")
    st.markdown(f"- **Nachfrage:** {reqs.get('demand_type', '–')}")

    st.markdown("#### Restriktionen")
    if constraints_list:
        for constraint in constraints_list:
            st.markdown(f"- {constraint}")
    else:
        st.markdown("- –")

    # ── Ausgewähltes Modell ───────────────────────────────────────────────────
    st.markdown("### 📐 Ausgewähltes Modell")

    st.markdown(f"- **Typ:** {selected_model_type or '–'}")
    st.markdown(f"- **Name:** {selected_model.get('name', '–')}")

    explanation = selected_model.get("explanation")
    if explanation:
        st.markdown(f"- **Begründung:** {explanation}")

    # ── Modellspezifische Eingaben ────────────────────────────────────────────
    if selected_model_inputs:
        st.markdown("### 🧾 Erfasste Modellparameter")

        if selected_model_type == "newsvendor":
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Verkaufspreis",
                    selected_model_inputs.get("selling_price", "–"),
                )

            with col2:
                st.metric(
                    "Einkaufskosten",
                    selected_model_inputs.get("purchase_cost", "–"),
                )

            with col3:
                st.metric(
                    "Restwert",
                    selected_model_inputs.get("salvage_value", "–"),
                )

            demand_scenarios = selected_model_inputs.get("demand_scenarios", [])
            scenario_probabilities = selected_model_inputs.get(
                "scenario_probabilities",
                [],
            )

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

        else:
            st.json(selected_model_inputs)

    st.divider()

    # ── Modellvisualisierung ──────────────────────────────────────────────────
    st.markdown("### 📐 Optimierungsmodell")
    st.markdown(
        result.get(
            "model_markdown",
            "Keine Modellvisualisierung verfügbar.",
        )
    )

    st.divider()

    # ── Generierter PuLP-Code ─────────────────────────────────────────────────
    # Code-Generierung ist aktuell bewusst noch nicht Hauptfokus.
    generated_code = result.get("generated_code", "")
    if generated_code:
        st.markdown("### 💻 Generierter PuLP-Code")
        st.code(generated_code, language="python")
        st.divider()

    # ── Nächste Schritte ──────────────────────────────────────────────────────
    st.markdown("### 🔜 Nächste Schritte")
    for step in result.get("next_steps", []):
        st.markdown(f"- {step}")

    # ── Neu starten ───────────────────────────────────────────────────────────
    st.divider()
    if st.button("↺ Neu starten", use_container_width=True):
        for key in ["step", "form_data", "graph_state"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()