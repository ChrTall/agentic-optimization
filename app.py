import streamlit as st

from agentic_optimization.configuration.llm_config import load_llm_configuration
from agentic_optimization.graph import build_graph
from agentic_optimization.llm.factory import create_chat_model
from agentic_optimization.state import (
    EVENT_GENERAL_CLARIFICATION_SUBMITTED,
    EVENT_MODEL_INPUTS_SUBMITTED,
    EVENT_NEW_PROBLEM,
    WORKFLOW_MODEL_INPUTS_INVALID,
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

step_for_progress = "freitext" if st.session_state.step == "clarify" else st.session_state.step
step_index = STEPS.index(step_for_progress) if step_for_progress in STEPS else 1
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
# SCHRITT 3: Modellspezifische Eingaben
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "model_input_form":
    st.subheader("Modellspezifische Eingaben")

    graph_state = st.session_state.graph_state or {}
    selected_family = graph_state.get("selected_model_family")
    selected_variant = graph_state.get("selected_model_variant")

    required_constraint_inputs = graph_state.get(
        "required_constraint_inputs",
        [],
    )

    constraint_validation_errors = graph_state.get(
        "constraint_validation_errors",
        [],
    )

    for error in constraint_validation_errors:
        st.error(error)

    # Technische Fehler beim Freitext-Mapping
    mapping_errors = graph_state.get(
        "constraint_mapping_errors",
        [],
    )

    if mapping_errors:
        for error in mapping_errors:
            st.error(error)

        st.warning(
            "Die genannten Nebenbedingungen konnten technisch nicht "
            "verarbeitet werden. Bitte gehe zurück und versuche es erneut."
        )

        if st.button(
            "← Problembeschreibung bearbeiten",
            use_container_width=True,
        ):
            st.session_state.step = "freitext"
            st.rerun()

        st.stop()

    # Fehler der modellspezifischen Eingaben
    validation_errors = graph_state.get(
        "model_input_validation_errors",
        [],
    )

    for error in validation_errors:
        st.error(error)

    if (
        selected_family != "newsvendor"
        or selected_variant != "single_product_single_period"
    ):
        st.warning(
            "Für die ausgewählte Modellfamilie und Variante "
            "gibt es noch kein spezifisches Eingabeformular."
        )
        st.stop()

    if required_constraint_inputs:
        st.info(
            "Die Modellparameter sind vollständig. "
            "Bitte ergänze noch die fehlenden Werte der Nebenbedingungen."
        )
    else:
        st.info(
            "Es wurde ein Newsvendor-Modell erkannt. "
            "Bitte ergänze die modellspezifischen Eingaben."
        )

    existing_newsvendor_inputs = (
        graph_state.get("model_specific_inputs", {})
        .get("newsvendor", {})
        .get("single_product_single_period", {})
    )

    existing_demand_scenarios = existing_newsvendor_inputs.get(
        "demand_scenarios",
        [],
    )

    existing_scenario_probabilities = existing_newsvendor_inputs.get(
        "scenario_probabilities",
        [],
    )

    constraint_input_values: dict[
        tuple[int, str],
        float | None,
    ] = {}

    with st.form("newsvendor_form"):
        selling_price = st.number_input(
            "Verkaufspreis pro Einheit",
            min_value=0.0,
            step=0.5,
            value=float(
                existing_newsvendor_inputs.get("selling_price", 0.0)
            ),
        )

        purchase_cost = st.number_input(
            "Einkaufskosten pro Einheit",
            min_value=0.0,
            step=0.5,
            value=float(
                existing_newsvendor_inputs.get("purchase_cost", 0.0)
            ),
        )

        salvage_value = st.number_input(
            "Restwert pro unverkaufter Einheit",
            min_value=0.0,
            step=0.5,
            value=float(
                existing_newsvendor_inputs.get("salvage_value", 0.0)
            ),
        )

        demand_scenarios = st.text_input(
            "Nachfrageszenarien",
            value=", ".join(
                str(value)
                for value in existing_demand_scenarios
            ),
            placeholder="z.B. 80, 100, 120, 140",
        )

        scenario_probabilities = st.text_input(
            "Wahrscheinlichkeiten der Szenarien",
            value=", ".join(
                str(value)
                for value in existing_scenario_probabilities
            ),
            placeholder="z.B. 0.25, 0.25, 0.25, 0.25",
        )

        if required_constraint_inputs:
            st.divider()
            st.markdown("#### Fehlende Werte für Nebenbedingungen")

            for required_input in required_constraint_inputs:
                constraint_index = required_input["constraint_index"]
                constraint_type = required_input["constraint_type"]
                source_text = required_input.get("source_text")

                if source_text:
                    st.caption(source_text)

                for field in required_input.get("fields", []):
                    field_name = field["name"]
                    field_type = field["type"]

                    widget_key = (
                        f"constraint_{constraint_index}_{field_name}"
                    )

                    if field_type == "number":
                        constraint_input_values[
                            (constraint_index, field_name)
                        ] = st.number_input(
                            field["label"],
                            value=None,
                            step=1.0,
                            help=field.get("help"),
                            key=widget_key,
                        )
                    else:
                        st.error(
                            f"Der Constraint-Feldtyp '{field_type}' "
                            f"für '{constraint_type}' wird noch nicht unterstützt."
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
            parsed_demand_scenarios = parse_number_list(
                demand_scenarios
            )
            parsed_scenario_probabilities = parse_number_list(
                scenario_probabilities
            )
        except ValueError:
            st.error(
                "Bitte gib Nachfrageszenarien und Wahrscheinlichkeiten als "
                "kommagetrennte Zahlen ein, z.B. 80, 100, 120 und "
                "0.2, 0.5, 0.3."
            )
            st.stop()

        newsvendor_inputs = {
            "selling_price": selling_price,
            "purchase_cost": purchase_cost,
            "salvage_value": salvage_value,
            "demand_scenarios": parsed_demand_scenarios,
            "scenario_probabilities": parsed_scenario_probabilities,
        }

        prev_state = st.session_state.graph_state or {}

        existing_constraints = prev_state.get(
            "structured_constraints",
            [],
        )

        updated_constraints = [
            {
                **constraint,
                "parameters": dict(
                    constraint.get("parameters", {})
                ),
            }
            for constraint in existing_constraints
        ]

        for (
            constraint_index,
            parameter_name,
        ), parameter_value in constraint_input_values.items():
            if parameter_value is None:
                continue

            if not 0 <= constraint_index < len(updated_constraints):
                st.error(
                    "Ein Constraint konnte nicht eindeutig aktualisiert werden."
                )
                st.stop()

            updated_constraints[
                constraint_index
            ]["parameters"][parameter_name] = parameter_value

        existing_model_inputs = prev_state.get(
            "model_specific_inputs",
            {},
        )

        next_state = {
            **prev_state,
            "control": {
                **prev_state.get("control", {}),
                "event": EVENT_MODEL_INPUTS_SUBMITTED,
            },
            "model_specific_inputs": {
                **existing_model_inputs,
                "newsvendor": {
                    **existing_model_inputs.get("newsvendor", {}),
                    "single_product_single_period": newsvendor_inputs,
                },
            },
            "structured_constraints": updated_constraints,
            "constraint_validation_errors": [],
            "required_constraint_inputs": [],
        }

        llm = st.session_state.get("llm")

        if llm is None:
            try:
                llm_config = load_llm_configuration()
                llm = create_chat_model(llm_config)
            except ValueError as error:
                st.error(str(error))
                st.stop()

            st.session_state.llm = llm

        app = build_graph(llm)
        st.session_state.graph_state = app.invoke(next_state)

        st.session_state.step = route_ui_from_graph_state(
            st.session_state.graph_state
        )

        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4: Ergebnis
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "result":
    gs = st.session_state.graph_state or {}
    result = gs.get("result", {})
    reqs = result.get("requirements", {})
    selected_model = result.get("selected_model", {})

    selected_family = selected_model.get("family_id")
    selected_family_name = selected_model.get("family_name")

    selected_variant = selected_model.get("variant_id")
    selected_variant_name = selected_model.get("variant_name")

    selected_model_inputs = result.get(
        "selected_model_inputs",
        {},
    )

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

    st.markdown(
        f"- **Modellfamilie:** "
        f"{selected_family_name or selected_family or '–'}"
    )

    st.markdown(
        f"- **Variante:** "
        f"{selected_variant_name or selected_variant or '–'}"
    )
    explanation = selected_model.get("explanation")
    if explanation:
        st.markdown(f"- **Begründung:** {explanation}")

    # ── Modellspezifische Eingaben ────────────────────────────────────────────
    if selected_model_inputs:
        st.markdown("### 🧾 Erfasste Modellparameter")

        if (
            selected_family == "newsvendor"
            and selected_variant == "single_product_single_period"
        ):
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

    structured_constraints = result.get(
        "structured_constraints",
        [],
    )

    unsupported_constraints = result.get(
        "unsupported_constraints",
        [],
    )

    if unsupported_constraints:
        st.markdown("#### Nicht unterstützte Constraints")

        for constraint in unsupported_constraints:
            st.error(str(constraint))

        st.markdown("### 🔒 Strukturierte optionale Constraints")

        if structured_constraints:
            for constraint in structured_constraints:
                constraint_type = constraint.get("type", "unbekannt")
                parameters = constraint.get("parameters", {})

                st.markdown(f"- **{constraint_type}**: `{parameters}`")
        else:
            st.markdown("- Keine optionalen Constraints angegeben.")

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
    generated_code = result.get("generated_code")
    generation_errors = result.get("generation_errors", [])

    with st.expander("Technische Details: generierter PuLP-Code"):
        if generated_code:
            st.code(generated_code, language="python")
        elif generation_errors:
            for error in generation_errors:
                st.error(error)
        else:
            st.info("Es wurde noch kein PuLP-Code erzeugt.")
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