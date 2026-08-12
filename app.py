import streamlit as st

from agentic_optimization.configuration.llm_config import load_llm_configuration
from agentic_optimization.graph import build_graph
from agentic_optimization.llm.factory import create_chat_model
from agentic_optimization.models.registry import (
    get_model_family_definition,
    get_model_variant_definition,
)
from agentic_optimization.state import (
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
from agentic_optimization.ui.pdf_input import (
    extract_text_from_pdf,
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

FORM_SUMMARY_FIELDS = (
    (
        "Entscheidung",
        "decision",
        DECISION_LABELS,
    ),
    (
        "Ziel",
        "objective",
        OBJECTIVE_LABELS,
    ),
    (
        "Nachfrage",
        "demand_type",
        DEMAND_TYPE_LABELS,
    ),
    (
        "Produktumfang",
        "product_scope",
        PRODUCT_SCOPE_LABELS,
    ),
    (
        "Planungszeitraum",
        "planning_horizon",
        PLANNING_HORIZON_LABELS,
    ),
)


def format_selectbox_value(
    value: str | None,
    labels: dict[str, str],
) -> str:
    """Übersetzt einen internen Auswahlwert in das zugehörige UI-Label."""

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
if "raw_problem_description" not in st.session_state:
    st.session_state.raw_problem_description = ""
if "description_input_mode" not in st.session_state:
    st.session_state.description_input_mode = "freitext"
if "pdf_extraction_signature" not in st.session_state:
    st.session_state.pdf_extraction_signature = None

DESCRIPTION_INPUT_MODE_LABELS = {
    "freitext": "✏️ Freitext eingeben",
    "pdf": "📄 Dokument hochladen (PDF)",
}

# ── Fortschrittsanzeige ───────────────────────────────────────────────────────

STEPS = ["form", "freitext", "model_input_form", "review", "result"]
STEP_LABELS = ["① Formular", "② Beschreibung", "③ Modellinput", "④ Überprüfung", "⑤ Ergebnis"]

step_for_progress = (
    "form"
    if st.session_state.step == "model_unsupported"
    else st.session_state.step
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


def route_ui_from_graph_state(graph_state: dict) -> str:
    """
    Übersetzt den fachlichen Workflow-Status aus dem Graph
    in den nächsten Streamlit-Schritt.

    Der Graph entscheidet, was fachlich als Nächstes nötig ist.
    Diese Funktion entscheidet, welche UI-Seite dafür angezeigt wird.
    """

    status = get_workflow_status(graph_state)

    if status == WORKFLOW_NEEDS_GENERAL_CLARIFICATION:
        return "freitext"

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
# SCHRITT 1: Grundlegende Angaben
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.step == "form":
    st.subheader("Grundlegende Angaben")
    st.caption(
        "Beantworten Sie die Fragen so gut wie möglich. "
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

        # Eine Änderung der grundlegenden Angaben macht
        # sämtliche vorherigen Ableitungen ungültig.
        st.session_state.graph_state = None

        st.session_state.step = "freitext"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 2: Freitext + Graph ausführen
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "freitext":
    st.subheader("Problembeschreibung")

    graph_state = st.session_state.graph_state or {}

    has_clarification = (
        get_workflow_status(graph_state)
        == WORKFLOW_NEEDS_GENERAL_CLARIFICATION
    )

    # ---------------------------------------------------------
    # Bisherige Formularangaben
    # ---------------------------------------------------------

    with st.expander(
        "Ihre bisherigen Angaben",
        expanded=False,
    ):
        for (
            label,
            field_name,
            value_labels,
        ) in FORM_SUMMARY_FIELDS:
            value = st.session_state.form_data.get(
                field_name
            )

            if value is None:
                continue

            display_value = value_labels.get(
                value,
                str(value),
            )

            st.markdown(
                f"**{label}:** {display_value}"
            )

    # ---------------------------------------------------------
    # Clarification direkt in Formular 2
    # ---------------------------------------------------------

    if has_clarification:
        clarification_question = graph_state.get(
            "clarification_question",
            "",
        )

        st.warning(
            clarification_question
            or (
                "Die Beschreibung enthält unvollständige "
                "oder widersprüchliche Angaben. Bitte "
                "korrigieren Sie den Text und analysieren "
                "Sie ihn erneut."
            )
        )

        st.caption(
            "Ihr bisheriger Text bleibt erhalten. Ändern "
            "Sie nur die betreffende Angabe."
        )

    # ---------------------------------------------------------
    # Auswahl der Eingabeart: Freitext oder PDF-Dokument
    # ---------------------------------------------------------

    st.session_state.description_input_mode = st.radio(
        "Wie möchten Sie Ihr Problem beschreiben?",
        options=list(DESCRIPTION_INPUT_MODE_LABELS),
        format_func=lambda value: (
            DESCRIPTION_INPUT_MODE_LABELS[value]
        ),
        horizontal=True,
        key="description_input_mode_choice",
        index=list(DESCRIPTION_INPUT_MODE_LABELS).index(
            st.session_state.description_input_mode
        ),
    )

    input_mode = st.session_state.description_input_mode

    # ---------------------------------------------------------
    # PDF-Upload: Text wird extrahiert und in das Beschreibungs-
    # feld unten übernommen, damit der bestehende Analyse-Pfad
    # (raw_input) unverändert bleibt.
    # ---------------------------------------------------------

    if input_mode == "pdf":
        uploaded_pdf = st.file_uploader(
            (
                "PDF-Dokument hochladen (z. B. Liefervertrag, "
                "Spezifikation)"
            ),
            type=["pdf"],
            key="pdf_uploader",
        )

        if uploaded_pdf is not None:
            file_signature = (
                uploaded_pdf.name,
                uploaded_pdf.size,
            )

            if (
                st.session_state.pdf_extraction_signature
                != file_signature
            ):
                with st.spinner(
                    "PDF wird gelesen und Text extrahiert …"
                ):
                    extraction_result = (
                        extract_text_from_pdf(uploaded_pdf)
                    )

                if extraction_result.error:
                    st.error(extraction_result.error)
                else:
                    st.session_state.raw_problem_description = (
                        extraction_result.text
                    )
                    st.success(
                        "Text wurde aus "
                        f"{extraction_result.page_count} "
                        "Seite(n) übernommen. Sie können ihn "
                        "unten noch anpassen."
                    )

                st.session_state.pdf_extraction_signature = (
                    file_signature
                )
        else:
            st.session_state.pdf_extraction_signature = None

        st.caption(
            "Der extrahierte Text erscheint editierbar im "
            "Textfeld unten und wird genauso analysiert wie "
            "eine manuell eingegebene Beschreibung."
        )

    # ---------------------------------------------------------
    # Editierbarer Beschreibungstext (gemeinsame Grundlage für
    # Freitext- und PDF-Eingabe)
    # ---------------------------------------------------------

    raw_input = st.text_area(
        "Beschreiben Sie Ihr Problem in eigenen Worten",
        placeholder=(
            "Zum Beispiel: Wir verkaufen ein saisonales "
            "Produkt mit unsicherer Nachfrage und einer "
            "Lagerkapazität von maximal 1.000 Einheiten."
        ),
        height=200,
        key="raw_problem_description",
    )

    col_back, col_analyze = st.columns([1, 3])

    with col_back:
        if st.button(
            "← Zurück",
            use_container_width=True,
        ):
            st.session_state.step = "form"
            st.rerun()

    with col_analyze:
        analyze = st.button(
            (
                "Erneut analysieren →"
                if has_clarification
                else "Problem analysieren →"
            ),
            use_container_width=True,
            type="primary",
        )

    # ---------------------------------------------------------
    # Erste und erneute Analyse: derselbe Codepfad
    # ---------------------------------------------------------

    if analyze:
        with st.spinner(
            "Ihre Beschreibung wird analysiert …"
        ):
            llm = st.session_state.get("llm")

            if llm is None:
                try:
                    llm_config = (
                        load_llm_configuration()
                    )
                    llm = create_chat_model(
                        llm_config
                    )
                except ValueError as error:
                    st.error(str(error))
                    st.stop()

                st.session_state.llm = llm

            app = build_graph(llm)

            analysis_state = {
                "control": {
                    "event": EVENT_NEW_PROBLEM,
                },
                "raw_input": raw_input,
                "form_data": (
                    st.session_state.form_data
                ),
            }

            st.session_state.graph_state = (
                app.invoke(analysis_state)
            )

        st.session_state.step = (
            route_ui_from_graph_state(
                st.session_state.graph_state
            )
        )

        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# FALL: Problemstruktur wird noch nicht unterstützt
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "model_unsupported":
    st.subheader("Problemart noch nicht unterstützt")

    graph_state = (
        st.session_state.graph_state
        or {}
    )

    st.warning(
        graph_state.get(
            "model_selection_error",
            (
                "Für diese Problemstruktur ist derzeit "
                "keine Berechnungsmethode verfügbar."
            ),
        )
    )

    st.info(
        "Passen Sie die grundlegenden Angaben oder die "
        "Problembeschreibung an."
    )

    if st.button(
        "← Grundlegende Angaben bearbeiten",
        use_container_width=True,
    ):
        st.session_state.step = "form"
        st.rerun()

    if st.button(
        "Problembeschreibung bearbeiten",
        use_container_width=True,
    ):
        st.session_state.step = "freitext"
        st.rerun()

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3: Modellspezifische Eingaben
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "model_input_form":
    st.subheader("Angaben zur Berechnung")

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
            "Die beschriebenen Rahmenbedingungen konnten "
            "technisch nicht verarbeitet werden. Bitte "
            "bearbeiten Sie die Problembeschreibung."
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

    constraint_validation_errors = graph_state.get(
        "constraint_validation_errors",
        [],
    )

    # ---------------------------------------------------------
    # UI-Daten der ausgewählten Modellvariante laden
    # ---------------------------------------------------------

    family_definition = get_model_family_definition(
        selected_family
    )

    family_name = (
        family_definition["family_name"]
        if family_definition is not None
        else selected_family
    )

    variant_name = variant_definition["variant_name"]

    message_column, info_column = st.columns(
        [12, 1]
    )

    with message_column:
        st.success(
            "**Das Problem wurde erkannt.**\n\n"
            "Für die Berechnung werden noch die "
            "folgenden Angaben benötigt."
        )

    with info_column, st.popover(
        "ⓘ",
        help=(
            "Technische Informationen zur ausgewählten "
            "Berechnungsmethode"
        ),
    ):
        st.markdown(
            "**Technische Modellinformationen:**"
        )

        st.markdown(
            f"**Modellfamilie:** {family_name}"
        )

        st.markdown(
            f"**Modellvariante:** {variant_name}"
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

    st.markdown("### 1. Benötigte Eingabedaten")

    model_input_error_area = st.container()

    with model_input_error_area:
        if model_input_errors:
            st.error(
                "Bitte korrigieren Sie die folgenden "
                "Eingabedaten:"
            )

            for error in model_input_errors:
                st.error(error)

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
    st.markdown("### 2. Betriebliche Rahmenbedingungen")

    constraint_error_area = st.container()

    with constraint_error_area:
        if constraint_validation_errors:
            st.error(
                "Bitte korrigieren Sie die folgenden "
                "betrieblichen Rahmenbedingungen:"
            )

            for error in constraint_validation_errors:
                st.error(error)

    if constraint_candidates:
        st.info(
            "In Ihrer Beschreibung wurden mögliche "
            "betriebliche Rahmenbedingungen erkannt. "
            "Bitte prüfen Sie die vorausgewählten Angaben "
            "und passen Sie diese bei Bedarf an."
        )
    else:
        st.caption(
            "In Ihrer Beschreibung wurden keine konkreten "
            "betrieblichen Rahmenbedingungen erkannt. "
            "Sie können trotzdem passende "
            "Rahmenbedingungen auswählen."
        )

    if unsupported_constraints:
        st.warning(
            "Einige beschriebene Rahmenbedingungen werden "
            "von der ausgewählten Berechnungsmethode nicht "
            "unterstützt:"
        )

        for unsupported in unsupported_constraints:
            st.markdown(f"- {unsupported}")

    constraint_choice_key = (
        f"constraint_choice_"
        f"{selected_family}_"
        f"{selected_variant}"
    )

    question_column, choice_column = st.columns(
        [4,1],
        vertical_alignment="center",
    )

    with question_column:
        st.markdown(
            "#### Rahmenbedingungen berücksichtigen?"
        )

    with choice_column:
        constraint_choice = st.radio(
            "Auswahl der Rahmenbedingungen",
            options=["yes", "no"],
            index=(
                0
                if (
                    constraint_candidates
                    or existing_active_constraints
                )
                else 1
            ),
            format_func=lambda value: {
                "yes": "Ja",
                "no": "Nein",
            }[value],
            horizontal=True,
            key=constraint_choice_key,
            label_visibility="collapsed",
        )

    constraints_enabled = (
        constraint_choice == "yes"
    )

    if constraints_enabled:
        st.caption(
            "Wählen Sie die zutreffenden Rahmenbedingungen "
            "aus und prüfen Sie deren Werte."
        )
    else:
        st.info(
            "Es werden keine zusätzlichen betrieblichen "
            "Rahmenbedingungen berücksichtigt. Die "
            "verfügbaren Möglichkeiten bleiben zur "
            "Information sichtbar."
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

        constraint_name = constraint_definition[
            "name"
        ]

        existing_parameters = (
            existing_constraint.get(
                "parameters",
                {},
            )
            if existing_constraint
            else {}
        )

        with st.container(border=True):
            title_column, status_column = st.columns(
                [4, 1],
                vertical_alignment="center",
            )

            with title_column:
                st.markdown(
                    f"#### {constraint_name}"
                )

                include_constraint = st.toggle(
                    "Bei der Berechnung berücksichtigen",
                    key=checkbox_key,
                    disabled=not constraints_enabled,
                )

            constraint_is_active = (
                include_constraint
                and constraints_enabled
            )

            with status_column:
                if candidate_detected:
                    st.badge(
                        "Erkannt",
                        color="blue",
                    )

                if constraint_is_active:
                    st.badge(
                        "Aktiviert",
                        color="green",
                    )
                else:
                    st.badge(
                        "Deaktiviert",
                        color="gray",
                    )

            if constraint_is_active:
                selected_constraint_types.append(
                    constraint_type
                )

            for parameter_definition in (
                constraint_definition["parameters"]
            ):
                parameter_name = (
                    parameter_definition["name"]
                )

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
                existing_constraint.get(
                    "source_text"
                )
                if existing_constraint
                else None
            )

            if source_text:
                st.caption(
                    f"Aus Ihrer Beschreibung erkannt: "
                    f"{source_text}"
                )

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

    submission_error_area = st.container()

    with submission_error_area:
        if (
            model_input_errors
            or constraint_validation_errors
        ):
            st.error(
                "Die Eingaben enthalten Fehler. Bitte prüfen "
                "Sie die markierten Bereiche."
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
            with model_input_error_area:
                st.error(
                    "Bitte korrigieren Sie die "
                    "Eingabedaten:"
                )
                st.error(str(error))

            with submission_error_area:
                st.error(
                    "Die Eingaben enthalten Fehler. Bitte prüfen "
                    "Sie die markierten Bereiche."
                )

            st.stop()

        final_structured_constraints: list[dict] = []

        if constraints_enabled:
            if not selected_constraint_types:
                with constraint_error_area:
                    st.error(
                        "Sie haben ausgewählt, dass betriebliche "
                        "Rahmenbedingungen berücksichtigt werden "
                        "sollen. Aktivieren Sie mindestens eine "
                        "Rahmenbedingung oder wählen Sie „Nein“."
                    )

                with submission_error_area:
                    st.error(
                        "Die Eingaben enthalten Fehler. Bitte prüfen "
                        "Sie die markierten Bereiche."
                    )

                st.stop()

            if unsupported_constraints:
                with constraint_error_area:
                    st.error(
                        "Mindestens eine beschriebene "
                        "Rahmenbedingung wird nicht unterstützt. "
                        "Wählen Sie „Nein“, wenn keine "
                        "zusätzlichen Rahmenbedingungen "
                        "berücksichtigt werden sollen, oder "
                        "bearbeiten Sie die "
                        "Problembeschreibung."
                    )

                with submission_error_area:
                    st.error(
                        "Die Eingaben enthalten Fehler. Bitte prüfen "
                        "Sie die markierten Bereiche."
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

                        parameter_required = (
                            parameter_definition[
                                "required"
                            ]
                        )

                        if (
                            parameter_required
                            and parsed_value is None
                        ):
                            raise ValueError(
                                "Bitte geben Sie einen Wert für "
                                f"„{parameter_definition['label']}“ "
                                "ein."
                            )

                        if parsed_value is None:
                            continue

                        if isinstance(parsed_value, list):
                            raise TypeError(
                                "Parameter betrieblicher "
                                "Rahmenbedingungen dürfen "
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
                with constraint_error_area:
                    st.error(
                        "Bitte korrigieren Sie die "
                        "betrieblichen Rahmenbedingungen:"
                    )
                    st.error(str(error))

                with submission_error_area:
                    st.error(
                        "Die Eingaben enthalten Fehler. Bitte prüfen "
                        "Sie die markierten Bereiche."
                    )

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
        }

        # Neue Formular-3-Daten verwerfen Review-Bestätigung und alte Ergebnisse.
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
        "Bitte prüfen Sie die drei Bereiche. Die Berechnung "
        "kann gestartet werden, sobald alle Angaben "
        "bestätigt sind."
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
        st.markdown("### 1. Problemübersicht")

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
        st.markdown("### 2. Eingabedaten")

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
    # Karte 3: Betriebliche Rahmenbedingungen
    # ---------------------------------------------------------

    with st.container(border=True):
        st.markdown("### 3. Betriebliche Rahmenbedingungen")

        if review_data["constraints"]:
            for item in review_data["constraints"]:
                st.markdown(
                    f"**{item['label']}**  \n"
                    f"{item['value']}"
                )
        else:
            st.info("Es werden keine zusätzlichen betrieblichen Rahmenbedingungen berücksichtigt.")

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
                    else "Rahmenbedingungen bestätigen"
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
            "Bitte bestätigen Sie alle Bereiche, bevor Sie "
            "die Berechnung starten."
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

        # Die Optimierung startet erst nach Bestätigung aller drei Review-Bereiche.
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
    # Fachliche Ergebniskennzahlen der Modellvariante
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
    # Berücksichtigte betriebliche Rahmenbedingungen
    # ---------------------------------------------------------

    with st.expander(
    "Berücksichtigte betriebliche Rahmenbedingungen"
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
                "Es wurden keine zusätzlichen betrieblichen "
                "Rahmenbedingungen berücksichtigt."
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
            "raw_problem_description",
            "review_sections",
            "form_decision",
            "form_objective",
            "form_demand_type",
            "form_product_scope",
            "form_planning_horizon",
            "submit_general_form",
            "form_planning_horizon",
            "submit_general_form",
            "description_input_mode",
            "description_input_mode_choice",
            "pdf_extraction_signature",
            "pdf_uploader",
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
            "constraints_enabled_",
            "constraint_choice_",
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
