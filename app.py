import streamlit as st

from agentic_optimization.configuration.llm_config import load_llm_configuration
from agentic_optimization.graph import build_graph
from agentic_optimization.llm.factory import create_chat_model
from agentic_optimization.models.registry import (
    get_model_variant_definition,
)
from agentic_optimization.state import (
    EVENT_GENERAL_CLARIFICATION_SUBMITTED,
    EVENT_MODEL_INPUTS_SUBMITTED,
    EVENT_NEW_PROBLEM,
    EVENT_OPTIMIZATION_REQUESTED,
    WORKFLOW_MODEL_INPUTS_INVALID,
    WORKFLOW_MODEL_UNSUPPORTED,
    WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_READY_FOR_RESULT,
    WORKFLOW_READY_FOR_REVIEW,
    get_workflow_status,
    invalidate_optimization_state,
)
from agentic_optimization.ui.field_renderer import (
    parse_input_value,
    render_input_field,
)
from agentic_optimization.ui.result_builder import (
    build_result_data,
)
from agentic_optimization.ui.review_builder import (
    build_review_data,
)

DECISION_LABELS = {
    "order_quantity": "Bestellmenge",
    "production_quantity": "Produktionsmenge",
    "inventory_level": "Lagerbestand",
}

OBJECTIVE_LABELS = {
    "maximize_profit": "Gewinn maximieren",
    "minimize_cost": "Kosten minimieren",
    "reach_service_level": "Servicelevel erreichen",
}

DEMAND_TYPE_LABELS = {
    "known": "Sicher / bekannt",
    "uncertain": "Unsicher / saisonal",
    "historical_estimate": "Historisch geschätzt",
}

PRODUCT_SCOPE_LABELS = {
    "single_product": "Für ein Produkt",
    "multiple_products": "Für mehrere Produkte",
}

PLANNING_HORIZON_LABELS = {
    "single_period": "Für eine einmalige Entscheidung",
    "multiple_periods": (
        "Für mehrere aufeinanderfolgende Zeiträume"
    ),
}


def format_selectbox_value(
    value: str | None,
    labels: dict[str, str],
) -> str:
    if value is None:
        return ""

    return labels.get(value, value)


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

STEPS = ["form", "freitext", "model_input_form", "review", "result"]
STEP_LABELS = ["① Formular", "② Beschreibung", "③ Modellinput", "④ Überprüfung", "⑤ Ergebnis"]

step_for_progress = (
    "form"
    if st.session_state.step == "model_unsupported"
    else (
        "freitext"
        if st.session_state.step == "clarify"
        else st.session_state.step
    )
)
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

    if status == WORKFLOW_READY_FOR_REVIEW:
        return "review"

    if status == WORKFLOW_MODEL_UNSUPPORTED:
        return "model_unsupported"

    return "freitext"

# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 1: Formular
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.step == "form":
    st.subheader("Strukturierte Eingabe")
    st.caption(
        "Beantworte die Fragen so gut du kannst. "
        "Fehlende Angaben können anschließend ergänzt werden."
    )

    decision = st.selectbox(
        "Welche Entscheidung soll optimiert werden?",
        options=[None, *DECISION_LABELS],
        format_func=lambda value: format_selectbox_value(
            value,
            DECISION_LABELS,
        ),
        key="form_decision",
    )

    objective = st.selectbox(
        "Was ist das Ziel der Optimierung?",
        options=[None, *OBJECTIVE_LABELS],
        format_func=lambda value: format_selectbox_value(
            value,
            OBJECTIVE_LABELS,
        ),
        key="form_objective",
    )

    demand_type = st.selectbox(
        "Wie verhält sich die Nachfrage?",
        options=[None, *DEMAND_TYPE_LABELS],
        format_func=lambda value: format_selectbox_value(
            value,
            DEMAND_TYPE_LABELS,
        ),
        key="form_demand_type",
    )

    product_scope = st.selectbox(
        (
            "Für wie viele Produkte soll eine Entscheidung "
            "getroffen werden?"
        ),
        options=[None, *PRODUCT_SCOPE_LABELS],
        format_func=lambda value: format_selectbox_value(
            value,
            PRODUCT_SCOPE_LABELS,
        ),
        key="form_product_scope",
    )

    planning_horizon = st.selectbox(
        "Für welchen Zeitraum soll geplant werden?",
        options=[None, *PLANNING_HORIZON_LABELS],
        format_func=lambda value: format_selectbox_value(
            value,
            PLANNING_HORIZON_LABELS,
        ),
        key="form_planning_horizon",
    )

    submitted = st.button(
        "Weiter →",
        use_container_width=True,
        type="primary",
        key="submit_general_form",
    )

    if submitted:
        st.session_state.form_data = {
            "decision": decision,
            "objective": objective,
            "demand_type": demand_type,
            "product_scope": product_scope,
            "planning_horizon": planning_horizon,
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


elif st.session_state.step == "model_unsupported":
    st.subheader("Berechnungsmethode nicht verfügbar")

    graph_state = st.session_state.graph_state or {}

    model_selection_error = graph_state.get(
        "model_selection_error"
    )

    st.warning(
        model_selection_error
        or (
            "Für diese Problemstruktur ist derzeit noch "
            "keine Berechnungsmethode verfügbar."
        )
    )

    model_config = graph_state.get("model_config", {})
    explanation = model_config.get("explanation")

    if explanation:
        st.info(explanation)

    if st.button(
        "← Angaben ändern",
        use_container_width=True,
    ):
        st.session_state.step = "form"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3: Modellspezifische Eingaben
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "model_input_form":
    st.subheader("Angaben für die Berechnung")

    graph_state = st.session_state.graph_state or {}

    selected_family = graph_state.get(
        "selected_model_family"
    )
    selected_variant = graph_state.get(
        "selected_model_variant"
    )

    if not selected_family or not selected_variant:
        st.error(
            "Es wurde keine vollständige Modellvariante "
            "ausgewählt."
        )
        st.stop()

    variant_definition = get_model_variant_definition(
        selected_family,
        selected_variant,
    )

    if variant_definition is None:
        st.error(
            "Die ausgewählte Modellvariante ist nicht "
            "registriert."
        )
        st.stop()

    # ---------------------------------------------------------
    # Vorherige Fehler anzeigen
    # ---------------------------------------------------------

    mapping_errors = graph_state.get(
        "constraint_mapping_errors",
        [],
    )

    if mapping_errors:
        for error in mapping_errors:
            st.error(error)

        st.warning(
            "Die beschriebenen Vorgaben konnten technisch "
            "nicht verarbeitet werden. Bitte bearbeite die "
            "Problembeschreibung."
        )

        if st.button(
            "← Problembeschreibung bearbeiten",
            use_container_width=True,
        ):
            st.session_state.step = "freitext"
            st.rerun()

        st.stop()

    model_input_errors = graph_state.get(
        "model_input_validation_errors",
        [],
    )

    for error in model_input_errors:
        st.error(error)

    constraint_validation_errors = graph_state.get(
        "constraint_validation_errors",
        [],
    )

    for error in constraint_validation_errors:
        st.error(error)

    # ---------------------------------------------------------
    # Variantendaten laden
    # ---------------------------------------------------------

    model_config = graph_state.get("model_config", {})

    variant_name = model_config.get(
        "variant_name",
        selected_variant,
    )

    st.info(
        f"Es wurde die Modellvariante '{variant_name}' "
        "ausgewählt."
    )

    input_fields = variant_definition["input_fields"]

    supported_constraints = variant_definition[
        "supported_constraints"
    ]

    existing_model_inputs = (
        graph_state.get("model_specific_inputs", {})
        .get(selected_family, {})
        .get(selected_variant, {})
    )

    constraint_candidates = graph_state.get(
        "constraint_candidates",
        [],
    )

    existing_active_constraints = graph_state.get(
        "structured_constraints",
        [],
    )

    unsupported_constraints = graph_state.get(
        "unsupported_constraints",
        [],
    )

    # ---------------------------------------------------------
    # Kandidaten und bestehende aktive Constraints indizieren
    # ---------------------------------------------------------

    candidate_by_type: dict[str, dict] = {}

    for candidate in constraint_candidates:
        candidate_type = candidate.get("type")

        if not isinstance(candidate_type, str):
            st.error(
                "Ein erkannter Constraint besitzt keinen "
                "gültigen Typ."
            )
            st.stop()

        if candidate_type in candidate_by_type:
            st.error(
                "Eine mögliche Vorgabe wurde mehrfach erkannt: "
                f"{candidate_type}."
            )
            st.stop()

        candidate_by_type[candidate_type] = candidate

    active_constraint_by_type: dict[str, dict] = {}

    for constraint in existing_active_constraints:
        constraint_type = constraint.get("type")

        if not isinstance(constraint_type, str):
            continue

        active_constraint_by_type[
            constraint_type
        ] = constraint

    # Bereits bestätigte Constraints haben Vorrang vor Kandidaten.
    initial_constraint_by_type = {
        **candidate_by_type,
        **active_constraint_by_type,
    }

    # ---------------------------------------------------------
    # Abschnitt 1: Modellparameter
    # ---------------------------------------------------------

    st.markdown("### 1. Angaben für die Berechnung")

    raw_model_input_values: dict[str, object] = {}

    for field_definition in input_fields:
        field_name = field_definition["name"]

        raw_model_input_values[field_name] = (
            render_input_field(
                field_definition,
                existing_model_inputs.get(field_name),
                key_prefix=(
                    f"model_input_"
                    f"{selected_family}_"
                    f"{selected_variant}"
                ),
            )
        )

    # ---------------------------------------------------------
    # Abschnitt 2: Constraints
    # ---------------------------------------------------------

    st.divider()
    st.markdown("### 2. Vorgaben und Grenzen")

    if constraint_candidates:
        st.info(
            "In deiner Beschreibung wurden mögliche Vorgaben "
            "erkannt. Bitte prüfe die vorausgewählten Angaben "
            "und passe sie bei Bedarf an."
        )
    else:
        st.caption(
            "Es wurden in der Beschreibung keine konkreten "
            "Vorgaben erkannt. Du kannst trotzdem passende "
            "Vorgaben auswählen."
        )

    if unsupported_constraints:
        st.warning(
            "Einige beschriebene Vorgaben werden von der "
            "ausgewählten Berechnungsmethode nicht unterstützt:"
        )

        for unsupported in unsupported_constraints:
            st.markdown(f"- {unsupported}")

    constraint_mode_key = (
        f"constraint_mode_"
        f"{selected_family}_"
        f"{selected_variant}"
    )

    if constraint_mode_key not in st.session_state:
        st.session_state[constraint_mode_key] = (
            "yes"
            if (
                constraint_candidates
                or existing_active_constraints
            )
            else "no"
        )

    constraint_mode = st.radio(
        (
            "Sollen zusätzliche Vorgaben oder Grenzen "
            "berücksichtigt werden?"
        ),
        options=["no", "yes"],
        format_func=lambda value: {
            "no": "Nein",
            "yes": "Ja",
        }[value],
        horizontal=True,
        key=constraint_mode_key,
    )

    constraints_enabled = constraint_mode == "yes"

    if not constraints_enabled:
        st.info(
            "Es werden keine zusätzlichen Vorgaben "
            "berücksichtigt. Die verfügbaren Möglichkeiten "
            "bleiben zur Information sichtbar."
        )

    selected_constraint_types: list[str] = []

    raw_constraint_parameter_values: dict[
        tuple[str, str],
        object,
    ] = {}

    for (
        constraint_type,
        constraint_definition,
    ) in supported_constraints.items():
        existing_constraint = (
            initial_constraint_by_type.get(
                constraint_type
            )
        )

        candidate_detected = (
            constraint_type in candidate_by_type
        )

        checkbox_key = (
            f"constraint_include_"
            f"{selected_family}_"
            f"{selected_variant}_"
            f"{constraint_type}"
        )

        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = (
                existing_constraint is not None
            )

        constraint_name = constraint_definition["name"]

        if candidate_detected:
            checkbox_label = (
                f"{constraint_name} – in der "
                "Beschreibung erkannt"
            )
        else:
            checkbox_label = constraint_name

        include_constraint = st.checkbox(
            checkbox_label,
            key=checkbox_key,
            disabled=not constraints_enabled,
        )

        if include_constraint and constraints_enabled:
            selected_constraint_types.append(
                constraint_type
            )

        existing_parameters = (
            existing_constraint.get(
                "parameters",
                {},
            )
            if existing_constraint
            else {}
        )

        for parameter_definition in (
            constraint_definition["parameters"]
        ):
            parameter_name = parameter_definition["name"]

            raw_constraint_parameter_values[
                (
                    constraint_type,
                    parameter_name,
                )
            ] = render_input_field(
                parameter_definition,
                existing_parameters.get(
                    parameter_name
                ),
                key_prefix=(
                    f"constraint_parameter_"
                    f"{selected_family}_"
                    f"{selected_variant}_"
                    f"{constraint_type}"
                ),
                disabled=(
                    not constraints_enabled
                    or not include_constraint
                ),
            )

        source_text = (
            existing_constraint.get("source_text")
            if existing_constraint
            else None
        )

        if source_text:
            st.caption(
                f"Erkannt aus: {source_text}"
            )

        st.markdown("---")

    submitted = st.button(
        "Eingaben prüfen →",
        use_container_width=True,
        type="primary",
        key=(
            f"submit_model_inputs_"
            f"{selected_family}_"
            f"{selected_variant}"
        ),
    )

    # ---------------------------------------------------------
    # Absenden und normalisieren
    # ---------------------------------------------------------

    if submitted:
        parsed_model_inputs: dict[str, object] = {}

        try:
            for field_definition in input_fields:
                field_name = field_definition["name"]

                parsed_model_inputs[field_name] = (
                    parse_input_value(
                        field_definition,
                        raw_model_input_values.get(
                            field_name
                        ),
                    )
                )
        except (TypeError, ValueError) as error:
            st.error(str(error))
            st.stop()

        final_structured_constraints: list[dict] = []

        if constraints_enabled:
            if unsupported_constraints:
                st.error(
                    "Mindestens eine beschriebene Vorgabe wird "
                    "nicht unterstützt. Bitte wähle „Nein“, "
                    "wenn keine Vorgaben berücksichtigt werden "
                    "sollen, oder bearbeite die "
                    "Problembeschreibung."
                )
                st.stop()

            try:
                for constraint_type in (
                    selected_constraint_types
                ):
                    constraint_definition = (
                        supported_constraints[
                            constraint_type
                        ]
                    )

                    parsed_parameters: dict[
                        str,
                        int | float,
                    ] = {}

                    for parameter_definition in (
                        constraint_definition[
                            "parameters"
                        ]
                    ):
                        parameter_name = (
                            parameter_definition["name"]
                        )

                        parsed_value = parse_input_value(
                            parameter_definition,
                            raw_constraint_parameter_values[
                                (
                                    constraint_type,
                                    parameter_name,
                                )
                            ],
                        )

                        if parsed_value is not None:
                            if isinstance(parsed_value, list):
                                raise TypeError(
                                    "Constraintparameter dürfen "
                                    "keine Zahlenlisten sein."
                                )

                            parsed_parameters[
                                parameter_name
                            ] = parsed_value

                    initial_constraint = (
                        initial_constraint_by_type.get(
                            constraint_type
                        )
                    )

                    final_structured_constraints.append(
                        {
                            "type": constraint_type,
                            "parameters": parsed_parameters,
                            "source_text": (
                                initial_constraint.get(
                                    "source_text"
                                )
                                if initial_constraint
                                else None
                            ),
                        }
                    )

            except (TypeError, ValueError) as error:
                st.error(str(error))
                st.stop()

        # Bei "Nein" werden Kandidaten und unsupported Angaben
        # bewusst nicht als aktive Constraints übernommen.
        final_unsupported_constraints = (
            unsupported_constraints
            if constraints_enabled
            else []
        )

        previous_state = (
            st.session_state.graph_state or {}
        )

        all_model_inputs = previous_state.get(
            "model_specific_inputs",
            {},
        )

        next_state = {
            **previous_state,
            "control": {
                **previous_state.get("control", {}),
                "event": EVENT_MODEL_INPUTS_SUBMITTED,
            },
            "model_specific_inputs": {
                **all_model_inputs,
                selected_family: {
                    **all_model_inputs.get(
                        selected_family,
                        {},
                    ),
                    selected_variant: (
                        parsed_model_inputs
                    ),
                },
            },
            "structured_constraints": (
                final_structured_constraints
            ),
            "unsupported_constraints": (
                final_unsupported_constraints
            ),
            "constraint_validation_errors": [],
            "required_constraint_inputs": [],
        }

        next_state = invalidate_optimization_state(next_state)

        st.session_state.review_sections = {
            "problem": False,
            "parameters": False,
            "constraints": False,
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

        st.session_state.graph_state = app.invoke(
            next_state
        )

        st.session_state.step = (
            route_ui_from_graph_state(
                st.session_state.graph_state
            )
        )

        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4: Überprüfung der Angaben
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "review":
    st.subheader("Angaben prüfen")

    st.write(
        "Bitte prüfe die drei Bereiche. Die Berechnung kann "
        "gestartet werden, sobald alle Angaben bestätigt sind."
    )

    graph_state = st.session_state.graph_state or {}

    selected_family = graph_state.get(
        "selected_model_family"
    )
    selected_variant = graph_state.get(
        "selected_model_variant"
    )

    if not selected_family or not selected_variant:
        st.error(
            "Für die Überprüfung fehlen Modellfamilie oder "
            "Modellvariante."
        )
        st.stop()

    variant_definition = get_model_variant_definition(
        selected_family,
        selected_variant,
    )

    if variant_definition is None:
        st.error(
            "Die ausgewählte Modellvariante ist nicht "
            "registriert."
        )
        st.stop()

    review_data = build_review_data(
        graph_state,
        variant_definition,
    )

    if "review_sections" not in st.session_state:
        st.session_state.review_sections = {
            "problem": False,
            "parameters": False,
            "constraints": False,
        }

    review_sections = (
        st.session_state.review_sections
    )

    # ---------------------------------------------------------
    # Karte 1: Aufgabe
    # ---------------------------------------------------------

    with st.container(border=True):
        st.markdown("### 1. Ihre Aufgabe")

        for item in review_data["problem"]:
            st.markdown(
                f"**{item['label']}**  \n"
                f"{item['value']}"
            )

        st.caption(
            "Ausgewählte Berechnungsmethode: "
            f"{review_data['model_name']}"
        )

        if review_sections["problem"]:
            st.success("Dieser Bereich wurde bestätigt.")

        col_change, col_confirm = st.columns(2)

        with col_change:
            st.button(
                "Ändern",
                key="change_review_problem",
                use_container_width=True,
                disabled=True,
                help=(
                    "Das gezielte Zurückspringen zu Formular 1 "
                    "wird in einem späteren Schritt ergänzt."
                ),
            )

        with col_confirm:
            if st.button(
                (
                    "Bestätigt"
                    if review_sections["problem"]
                    else "Aufgabe bestätigen"
                ),
                key="confirm_review_problem",
                use_container_width=True,
                disabled=review_sections["problem"],
            ):
                st.session_state.review_sections = {
                    **review_sections,
                    "problem": True,
                }
                st.rerun()

    # Nach einem Rerun den aktuellen Zustand erneut lesen.
    review_sections = (
        st.session_state.review_sections
    )

    # ---------------------------------------------------------
    # Karte 2: Berechnungsangaben
    # ---------------------------------------------------------

    with st.container(border=True):
        st.markdown("### 2. Angaben für die Berechnung")

        for item in review_data["parameters"]:
            st.markdown(
                f"**{item['label']}**  \n"
                f"{item['value']}"
            )

        if review_sections["parameters"]:
            st.success("Dieser Bereich wurde bestätigt.")

        col_change, col_confirm = st.columns(2)

        with col_change:
            st.button(
                "Ändern",
                key="change_review_parameters",
                use_container_width=True,
                disabled=True,
                help=(
                    "Das gezielte Zurückspringen zu Formular 3 "
                    "wird in einem späteren Schritt ergänzt."
                ),
            )

        with col_confirm:
            if st.button(
                (
                    "Bestätigt"
                    if review_sections["parameters"]
                    else "Berechnungsangaben bestätigen"
                ),
                key="confirm_review_parameters",
                use_container_width=True,
                disabled=review_sections["parameters"],
            ):
                st.session_state.review_sections = {
                    **review_sections,
                    "parameters": True,
                }
                st.rerun()

    review_sections = (
        st.session_state.review_sections
    )

    # ---------------------------------------------------------
    # Karte 3: Vorgaben und Grenzen
    # ---------------------------------------------------------

    with st.container(border=True):
        st.markdown("### 3. Vorgaben und Grenzen")

        if review_data["constraints"]:
            for item in review_data["constraints"]:
                st.markdown(
                    f"**{item['label']}**  \n"
                    f"{item['value']}"
                )
        else:
            st.info(
                "Es werden keine zusätzlichen Vorgaben oder "
                "Grenzen berücksichtigt."
            )

        if review_sections["constraints"]:
            st.success("Dieser Bereich wurde bestätigt.")

        col_change, col_confirm = st.columns(2)

        with col_change:
            st.button(
                "Ändern",
                key="change_review_constraints",
                use_container_width=True,
                disabled=True,
                help=(
                    "Das gezielte Zurückspringen zu Formular 3 "
                    "wird in einem späteren Schritt ergänzt."
                ),
            )

        with col_confirm:
            if st.button(
                (
                    "Bestätigt"
                    if review_sections["constraints"]
                    else "Vorgaben bestätigen"
                ),
                key="confirm_review_constraints",
                use_container_width=True,
                disabled=review_sections["constraints"],
            ):
                st.session_state.review_sections = {
                    **review_sections,
                    "constraints": True,
                }
                st.rerun()

    review_sections = (
        st.session_state.review_sections
    )

    # ---------------------------------------------------------
    # Abschließende Bestätigung
    # ---------------------------------------------------------

    confirmed_count = sum(
        bool(review_sections[section])
        for section in (
            "problem",
            "parameters",
            "constraints",
        )
    )

    all_sections_confirmed = (
        confirmed_count == 3
    )

    st.progress(
        confirmed_count / 3,
        text=(
            f"{confirmed_count} von 3 Bereichen bestätigt"
        ),
    )

    if all_sections_confirmed:
        st.success(
            "Alle Bereiche wurden geprüft. Die Berechnung "
            "kann jetzt gestartet werden."
        )
    else:
        st.info(
            "Bestätige bitte noch alle Bereiche, bevor du "
            "die Berechnung startest."
        )

    start_optimization = st.button(
        "Optimierung starten",
        type="primary",
        use_container_width=True,
        disabled=not all_sections_confirmed,
    )

    if start_optimization:
        llm = st.session_state.get("llm")

        if llm is None:
            st.error(
                "Das Sprachmodell ist nicht initialisiert."
            )
            st.stop()

        next_state = {
            **graph_state,
            "control": {
                **graph_state.get("control", {}),
                "event": EVENT_OPTIMIZATION_REQUESTED,
            },
            "review_confirmed": True,
        }

        app = build_graph(llm)

        st.session_state.graph_state = app.invoke(
            next_state
        )

        st.session_state.step = route_ui_from_graph_state(
            st.session_state.graph_state
        )

        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 5: Optimierungsergebnis
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "result":
    graph_state = st.session_state.graph_state or {}

    result_data = build_result_data(
        graph_state
    )

    st.subheader("Optimierungsergebnis")

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    if result_data["successful"]:
        st.success(
            result_data["status_label"]
        )
        st.write(
            result_data["status_message"]
        )

    elif result_data["errors"]:
        st.error(
            result_data["status_label"]
        )
        st.write(
            result_data["status_message"]
        )

        for error in result_data["errors"]:
            st.error(error)

    else:
        st.warning(
            result_data["status_label"]
        )
        st.write(
            result_data["status_message"]
        )

    # ---------------------------------------------------------
    # Zentrale Ergebniskennzahlen
    # ---------------------------------------------------------

    if result_data["metrics"]:
        metric_columns = st.columns(
            len(result_data["metrics"])
        )

        for column, metric in zip(
            metric_columns,
            result_data["metrics"],
        ):
            with column:
                st.metric(
                    metric["label"],
                    metric["value"],
                )

    # ---------------------------------------------------------
    # Variantenspezifische Ergebnisdetails
    # ---------------------------------------------------------

    if result_data["detail_rows"]:
        with st.expander(
            result_data["detail_title"]
            or "Ergebnis im Detail",
            expanded=True,
        ):
            st.caption(
                "Die Tabelle zeigt, wie sich die Empfehlung "
                "in den betrachteten Nachfragesituationen "
                "auswirkt."
            )

            st.table(
                result_data["detail_rows"]
            )

    # ---------------------------------------------------------
    # Berücksichtigte Vorgaben
    # ---------------------------------------------------------

    with st.expander(
        "Berücksichtigte Vorgaben und Grenzen"
    ):
        if result_data["constraints"]:
            for constraint in result_data[
                "constraints"
            ]:
                st.markdown(
                    f"**{constraint['label']}**  \n"
                    f"{constraint['value']}"
                )
        else:
            st.info(
                "Es wurden keine zusätzlichen Vorgaben oder "
                "Grenzen berücksichtigt."
            )

    # ---------------------------------------------------------
    # Technische Details
    # ---------------------------------------------------------

    with st.expander("Technische Details"):
        st.markdown(
            f"**Modellfamilie**  \n"
            f"{result_data['family_name']}"
        )

        st.markdown(
            f"**Modellvariante**  \n"
            f"{result_data['variant_name']}"
        )

        st.markdown(
            f"**Solverstatus**  \n"
            f"{result_data['solver_status']}"
        )

        model_representation = result_data[
            "model_representation"
        ]

        if model_representation:
            st.markdown(
                "**Tatsächlich berechnetes "
                "Optimierungsmodell**"
            )

            st.caption(
                "Diese Darstellung entspricht dem "
                "PuLP-Modell, das an den Solver "
                "übergeben wurde."
            )

            st.code(
                model_representation,
                language="text",
            )
        else:
            st.info(
                "Für diese Berechnung ist keine technische "
                "Modellrepräsentation verfügbar."
            )

    # ---------------------------------------------------------
    # Neue Optimierung
    # ---------------------------------------------------------

    st.divider()

    if st.button(
        "Neue Optimierung starten",
        use_container_width=True,
    ):
        keys_to_clear = [
            "step",
            "form_data",
            "graph_state",
            "review_sections",
            "form_decision",
            "form_objective",
            "form_demand_type",
            "form_product_scope",
            "form_planning_horizon",
            "submit_general_form",
            "confirm_review_problem",
            "confirm_review_parameters",
            "confirm_review_constraints",
            "change_review_problem",
            "change_review_parameters",
            "change_review_constraints",
        ]

        dynamic_key_prefixes = (
            "model_input_",
            "constraint_mode_",
            "constraint_include_",
            "constraint_parameter_",
            "submit_model_inputs_",
        )

        keys_to_clear.extend(
            key
            for key in list(st.session_state)
            if key.startswith(dynamic_key_prefixes)
        )

        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()