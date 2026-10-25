import logging

import streamlit as st

from agentic_optimization.agents.business_intake import (
    create_initial_business_intake_state,
)
from agentic_optimization.configuration.llm_config import (
    load_llm_configuration,
)
from agentic_optimization.graph import build_graph
from agentic_optimization.llm.errors import (
    LLM_SERVICE_ERRORS,
    describe_llm_error,
    is_llm_service_error,
)
from agentic_optimization.llm.factory import create_chat_model
from agentic_optimization.models.core_applicability import (
    CORE_ASSUMPTION_CONFIRMATION_PREFIX,
)
from agentic_optimization.state import (
    EVENT_BUSINESS_INTAKE_REQUESTED,
    EVENT_MODULAR_INPUTS_SUBMITTED,
    EVENT_MODULAR_MODEL_PREPARATION_REQUESTED,
    EVENT_MODULAR_OPTIMIZATION_REQUESTED,
    EVENT_MODULAR_PRODUCT_SCHEMA_REQUESTED,
    EVENT_MODULAR_REQUIREMENT_EXCLUSIONS_CONFIRMED,
    EVENT_MODULAR_RESULT_EXPLANATION_REQUESTED,
    EVENT_MODULAR_REVIEW_EXPLANATION_REQUESTED,
    WORKFLOW_BUSINESS_CORE_SELECTED,
    WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED,
    WORKFLOW_MODEL_INPUTS_INVALID,
    WORKFLOW_MODEL_PREPARATION_FAILED,
    WORKFLOW_MODEL_UNSUPPORTED,
    WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
    WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_OPTIMIZATION_COMPLETED,
    WORKFLOW_OPTIMIZATION_FAILED,
    WORKFLOW_PRODUCT_TABLE_READY,
    WORKFLOW_READY_FOR_REVIEW,
    get_workflow_status,
    invalidate_optimization_state,
)
from agentic_optimization.ui.business_clarification_page import (
    render_business_clarification_page,
)
from agentic_optimization.ui.business_intake_page import (
    render_business_intake_page,
)
from agentic_optimization.ui.chat_assistant_panel import (
    ChatWorkflowHandlers,
    get_chat_product_rows,
    render_chat_panel,
    reset_chat_state,
)
from agentic_optimization.ui.modular_input_page import (
    render_modular_input_page,
)
from agentic_optimization.ui.modular_product_list_page import (
    render_dynamic_product_table_page,
)
from agentic_optimization.ui.modular_result_page import (
    render_modular_result_page,
)
from agentic_optimization.ui.modular_review_page import (
    render_modular_exclusion_confirmation,
    render_modular_review_page,
    render_modular_selection_issues,
)

from agentic_optimization.models.preparation_recovery import failed_preparation_state
from agentic_optimization.ui.preparation_error_page import render_preparation_error_page
from agentic_optimization.ui.business_answer_summary import render_answer_summary

LOGGER = logging.getLogger(__name__)

STEP_LABELS = [
    "① Situation",
    "② Rückfragen",
    "③ Modellinput",
    "④ Überprüfung",
    "⑤ Ergebnis",
]

STEP_INDEX = {
    "business_intake": 0,
    "business_clarification": 1,
    "model_unsupported": 1,
    "preparation_failed": 2,
    "product_table": 2,
    "model_input_form": 2,
    "review": 3,
    "result": 4,
}


# Fehler, die eine Workflow-Aktion abbrechen und dem Nutzer als Meldung
# statt als Traceback angezeigt werden.
WORKFLOW_ERRORS = (ValueError, *LLM_SERVICE_ERRORS)


def show_workflow_error(error: BaseException) -> None:
    """Zeigt einen Workflow- oder Anbieterfehler verständlich an."""

    if is_llm_service_error(error):
        LOGGER.exception("Aufruf des Sprachmodells fehlgeschlagen.")
        st.error(describe_llm_error(error))
    else:
        st.error(str(error))


def get_or_create_llm():
    """Lädt das konfigurierte Sprachmodell höchstens einmal."""

    llm = st.session_state.get("llm")

    if llm is None:
        llm_config = load_llm_configuration()
        llm = create_chat_model(llm_config)
        st.session_state.llm = llm

    return llm


def route_ui_from_graph_state(graph_state: dict) -> str:
    """Übersetzt den Workflowstatus in die nächste UI-Seite."""

    status = get_workflow_status(graph_state)

    if status == WORKFLOW_MODEL_PREPARATION_FAILED:
        return "preparation_failed"

    if status == WORKFLOW_PRODUCT_TABLE_READY:
        return "product_table"

    if status in {
        WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
        WORKFLOW_MODEL_INPUTS_INVALID,
    }:
        return "model_input_form"

    if status == WORKFLOW_READY_FOR_REVIEW:
        return ("review" if graph_state.get("modular_model", {}).get(
            "form_3_review_confirmed", False) else "model_input_form")

    if status in {
        WORKFLOW_OPTIMIZATION_COMPLETED,
        WORKFLOW_OPTIMIZATION_FAILED,
    }:
        return "result"

    if (
        status == WORKFLOW_NEEDS_GENERAL_CLARIFICATION
        and graph_state.get("modular_model", {}).get("proposal")
        is not None
    ):
        return "model_unsupported"

    if status == WORKFLOW_MODEL_UNSUPPORTED:
        return "model_unsupported"

    return "business_intake"


def advance_after_business_intake(
    workflow,
    graph_state: dict,
) -> None:
    """Bereitet nach der Geschäftsanalyse den modularen Kern vor."""

    st.session_state.graph_state = graph_state
    status = get_workflow_status(graph_state)

    if status == WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED:
        st.session_state.graph_state = failed_preparation_state(
            graph_state, RuntimeError("Business analysis failed"), "business_analysis")
        st.session_state.step = "preparation_failed"
        return

    if status == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT:
        st.session_state.step = "business_clarification"
        return

    if status != WORKFLOW_BUSINESS_CORE_SELECTED:
        st.session_state.step = "business_intake"
        return

    for key in list(st.session_state):
        if key.startswith(CORE_ASSUMPTION_CONFIRMATION_PREFIX):
            st.session_state.pop(key, None)

    modular_state = graph_state.get("modular_model", {})

    if not modular_state.get("products", []):
        next_event = EVENT_MODULAR_PRODUCT_SCHEMA_REQUESTED
    else:
        next_event = EVENT_MODULAR_MODEL_PREPARATION_REQUESTED

    next_state = {
        **graph_state,
        "control": {
            **graph_state.get("control", {}),
            "event": next_event,
        },
    }
    try:
        prepared_state = workflow.invoke(next_state)
    except Exception as error:
        prepared_state = failed_preparation_state(next_state, error,
            "product_schema" if next_event == EVENT_MODULAR_PRODUCT_SCHEMA_REQUESTED else "preparation")
    st.session_state.graph_state = prepared_state
    st.session_state.step = route_ui_from_graph_state(
        prepared_state
    )


def render_progress() -> None:
    """Zeigt den aktuellen Schritt des modularen Ablaufs."""

    current_index = STEP_INDEX.get(
        st.session_state.step,
        0,
    )
    columns = st.columns(len(STEP_LABELS))

    for index, (column, label) in enumerate(
        zip(columns, STEP_LABELS)
    ):
        with column:
            if index < current_index:
                color = "gray"
                suffix = " ✓"
                weight = "normal"
            elif index == current_index:
                color = "#1f6feb"
                suffix = ""
                weight = "bold"
            else:
                color = "lightgray"
                suffix = ""
                weight = "normal"

            st.markdown(
                "<p style='"
                f"color:{color};font-weight:{weight};"
                "text-align:center'>"
                f"{label}{suffix}</p>",
                unsafe_allow_html=True,
            )

    st.divider()


def reset_workflow() -> None:
    """Entfernt den Zustand der aktuellen Optimierung."""

    keys_to_clear = [
        "step",
        "graph_state",
        "raw_problem_description",
        "review_sections",
        "business_intake_documents",
        "business_edit_question",
    ]
    keys_to_clear.extend(
        key
        for key in list(st.session_state)
        if key.startswith("modular_")
        or key.startswith(CORE_ASSUMPTION_CONFIRMATION_PREFIX)
        or key.startswith("business_question_")
        or key.startswith("business_clarification_")
        or key.startswith("business_edit_value_")
    )

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    reset_chat_state()


# ---------------------------------------------------------------------------
# Workflow-Aktionen
# ---------------------------------------------------------------------------
# Formulare und Chat-Assistent lösen dieselben Aktionen aus. Jede Aktion
# aktualisiert graph_state und step; das anschließende st.rerun() erfolgt
# durch den Aufrufer.


def ensure_graph_state() -> dict:
    """Stellt sicher, dass ein Geschäftseinstieg vorhanden ist."""

    graph_state = st.session_state.graph_state

    if graph_state is None or "business_intake" not in graph_state:
        graph_state = {
            **(graph_state or {}),
            "business_intake": (
                create_initial_business_intake_state()
            ),
        }
        st.session_state.graph_state = graph_state

    return graph_state


def invoke_event(
    event: str,
    *,
    graph_state: dict | None = None,
    **updates,
) -> dict:
    """Startet den Graphen mit einem Event und übernimmt das Ergebnis."""

    graph_state = (
        graph_state
        if graph_state is not None
        else st.session_state.graph_state or {}
    )
    invocation_state = {
            **graph_state,
            **updates,
            "control": {
                **graph_state.get("control", {}),
                "event": event,
            },
        }
    try:
        workflow = build_graph(get_or_create_llm())
        result_state = workflow.invoke(invocation_state)
    except Exception as error:
        if event not in {EVENT_MODULAR_MODEL_PREPARATION_REQUESTED, EVENT_MODULAR_PRODUCT_SCHEMA_REQUESTED}:
            raise
        result_state = failed_preparation_state(invocation_state, error,
            "product_schema" if event == EVENT_MODULAR_PRODUCT_SCHEMA_REQUESTED else "preparation")
    st.session_state.graph_state = result_state
    st.session_state.step = route_ui_from_graph_state(result_state)
    return result_state


def run_business_intake(
    raw_input: str,
    answers: list,
    *, chat_verified: bool | None = None,
) -> None:
    """Wertet Beschreibung und Antworten des Geschäftseinstiegs aus."""

    graph_state = ensure_graph_state()
    intake_state = graph_state["business_intake"]
    if chat_verified is None:
        chat_verified = bool(intake_state.get("chat_verified_mode"))
    # The retry form does not render questions answered in the failed turn.
    # Replay those submissions only for the same description; fresh answers
    # from the form take precedence by question ID.
    if raw_input == graph_state.get("raw_input"):
        retained = {
            answer["question_id"]: answer
            for answer in intake_state.get("pending_answers", [])
        }
        retained.update({answer["question_id"]: answer for answer in answers})
        answers = list(retained.values())
    invocation_state = {
            **graph_state,
            "control": {
                **graph_state.get("control", {}),
                "event": EVENT_BUSINESS_INTAKE_REQUESTED,
            },
            "raw_input": raw_input,
            "business_intake": {
                **intake_state,
                "pending_answers": answers,
                "chat_verified_mode": chat_verified,
            },
        }
    st.session_state.raw_problem_description = raw_input
    try:
        workflow = build_graph(get_or_create_llm())
        result_state = workflow.invoke(invocation_state)
    except Exception as error:
        st.session_state.graph_state = failed_preparation_state(invocation_state, error, "business_analysis")
        st.session_state.step = "preparation_failed"
        return
    advance_after_business_intake(workflow, result_state)


def retry_preparation() -> None:
    graph = st.session_state.graph_state or {}
    model = graph.get("modular_model", {})
    stage = (model.get("preparation_error_details") or {}).get("stage")
    if stage == "business_analysis":
        run_business_intake(graph.get("raw_input", st.session_state.raw_problem_description), [],
                            chat_verified=graph.get("business_intake", {}).get("chat_verified_mode", False))
    else:
        event = (EVENT_MODULAR_PRODUCT_SCHEMA_REQUESTED if stage == "product_schema"
                 or not model.get("products") else EVENT_MODULAR_MODEL_PREPARATION_REQUESTED)
        invoke_event(event)


def submit_products(products: list) -> None:
    graph_state = st.session_state.graph_state or {}
    invoke_event(
        EVENT_MODULAR_MODEL_PREPARATION_REQUESTED,
        modular_model={
            **graph_state.get("modular_model", {}),
            "products": products,
            "form_3_review_confirmed": False,
        },
    )


def submit_model_inputs(submissions: list) -> None:
    graph_state = st.session_state.graph_state or {}
    next_state = invalidate_optimization_state(
        {
            **graph_state,
            "modular_model": {
                **graph_state.get("modular_model", {}),
                "pending_input_submissions": submissions,
                "form_3_review_confirmed": True,
            },
        }
    )
    invoke_event(
        EVENT_MODULAR_INPUTS_SUBMITTED,
        graph_state=next_state,
    )


def confirm_exclusions() -> None:
    invoke_event(EVENT_MODULAR_REQUIREMENT_EXCLUSIONS_CONFIRMED)


def start_optimization() -> None:
    graph_state = st.session_state.graph_state or {}
    modular_state = graph_state.get("modular_model", {})
    pending_assumptions = modular_state.get(
        "pending_core_assumptions", []
    )
    invoke_event(
        EVENT_MODULAR_OPTIMIZATION_REQUESTED,
        modular_model={
            **modular_state,
            "review_confirmed": True,
            "core_assumptions_confirmed": all(
                st.session_state.get(
                    CORE_ASSUMPTION_CONFIRMATION_PREFIX
                    + condition_id,
                    False,
                )
                for condition_id in pending_assumptions
            ),
        },
    )


def request_result_explanation() -> None:
    graph_state = st.session_state.graph_state or {}
    workflow = build_graph(get_or_create_llm())
    st.session_state.graph_state = workflow.invoke(
        {
            **graph_state,
            "control": {
                **graph_state.get("control", {}),
                "event": EVENT_MODULAR_RESULT_EXPLANATION_REQUESTED,
            },
        }
    )


def edit_description() -> None:
    st.session_state.step = "business_intake"


st.set_page_config(
    page_title="Agentic Optimization",
    page_icon="⚙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

if "step" not in st.session_state:
    st.session_state.step = "business_intake"
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "raw_problem_description" not in st.session_state:
    st.session_state.raw_problem_description = ""

ensure_graph_state()

# Der Chat wird vor den Formularen verarbeitet, damit vom Chat
# ausgelöste Schritte direkt im Formular sichtbar werden.
render_chat_panel(
    ChatWorkflowHandlers(
        get_llm=get_or_create_llm,
        run_business_intake=lambda text, answers: run_business_intake(text, answers, chat_verified=True),
        submit_products=submit_products,
        submit_model_inputs=submit_model_inputs,
        confirm_exclusions=confirm_exclusions,
        start_optimization=start_optimization,
        request_result_explanation=request_result_explanation,
        edit_description=edit_description,
        restart=reset_workflow,
        retry_preparation=retry_preparation,
    )
)

render_progress()

if st.session_state.get("chat_pending_proposal"):
    st.info("Bitte prüfen Sie den Vorschlag im Chat. Sie können ihn übernehmen, ändern oder weitere Angaben ergänzen.")
    st.stop()

if st.session_state.step in {"review", "result"}:
    edited, editing = render_answer_summary(st.session_state.graph_state.get("business_intake", {}))
    if edited:
        run_business_intake(st.session_state.raw_problem_description, edited)
        st.rerun()
    if editing:
        st.stop()


if st.session_state.step == "business_intake":
    graph_state = ensure_graph_state()

    modular_state = graph_state.get("modular_model", {})
    preparation_error = modular_state.get("preparation_error")

    if (
        get_workflow_status(graph_state)
        == WORKFLOW_MODEL_PREPARATION_FAILED
    ):
        st.error(
            preparation_error
            or (
                "Die automatische Modellzusammenstellung "
                "konnte nicht zuverlässig abgeschlossen werden."
            )
        )

    intake_state = graph_state["business_intake"]
    submission = render_business_intake_page(
        intake_state,
        raw_input_value=(
            st.session_state.raw_problem_description
        ),
        show_questions=not bool(
            intake_state.get("asked_question_ids")
            or intake_state.get("analyzed_raw_input")
        ),
    )

    if submission is not None:
        try:
            run_business_intake(
                submission["raw_input"],
                submission["answers"],
            )
        except WORKFLOW_ERRORS as error:
            show_workflow_error(error)
            st.stop()

        st.rerun()

    st.caption("Beim kompletten Neustart werden die bisherigen Angaben und der Chat zurückgesetzt.")
    if st.button("Komplett neu starten", key="restart_business_intake", width="stretch"):
        reset_workflow()
        st.rerun()


elif st.session_state.step == "preparation_failed":
    model = (st.session_state.graph_state or {}).get("modular_model", {})
    action = render_preparation_error_page(model.get("preparation_error_details") or {
        "message": model.get("preparation_error")})
    if action == "retry":
        with st.spinner("Vorbereitung wird erneut versucht …"):
            retry_preparation()
        st.rerun()
    elif action == "edit":
        edit_description()
        st.rerun()
    elif action == "restart":
        reset_workflow()
        st.rerun()


elif st.session_state.step == "business_clarification":
    graph_state = st.session_state.graph_state or {}
    intake_state = graph_state.get("business_intake")

    if intake_state is None:
        st.error("Die offenen Rückfragen konnten nicht geladen werden.")
        st.stop()

    answers = render_business_clarification_page(intake_state)

    if answers is not None:
        try:
            run_business_intake(
                st.session_state.raw_problem_description,
                answers,
            )
        except WORKFLOW_ERRORS as error:
            show_workflow_error(error)
            st.stop()

        st.rerun()

    if st.button(
        "Angaben im Chat ergänzen",
        width="stretch",
    ):
        st.session_state.step = "business_intake"
        st.rerun()


elif st.session_state.step == "product_table":
    graph_state = st.session_state.graph_state or {}
    modular_state = graph_state.get("modular_model", {})
    core_id = modular_state.get("core_id")
    proposal = modular_state.get("preliminary_proposal")

    if not core_id or proposal is None:
        st.error("Die Produkttabelle konnte nicht vorbereitet werden.")
        st.stop()

    chat_rows, chat_rows_version = get_chat_product_rows()
    products = render_dynamic_product_table_page(
        core_id=core_id,
        proposal=proposal,
        prefilled_rows=chat_rows,
        editor_version=chat_rows_version,
    )

    if products is None:
        st.stop()

    try:
        submit_products(products)
    except WORKFLOW_ERRORS as error:
        show_workflow_error(error)
        st.stop()

    st.rerun()


elif st.session_state.step == "model_unsupported":
    graph_state = st.session_state.graph_state or {}
    modular_state = graph_state.get("modular_model", {})
    proposal = modular_state.get("proposal")

    if (
        proposal is not None
        and proposal["selection_status"]
        == "needs_clarification"
    ):
        st.subheader("Offene Anforderung klären")
    else:
        st.subheader("Anforderung derzeit nicht unterstützt")

    if proposal is None:
        st.warning(
            "Für diese Problemstruktur ist derzeit keine "
            "Berechnungsmethode verfügbar."
        )
    else:
        render_modular_selection_issues(proposal)

        if render_modular_exclusion_confirmation(proposal):
            try:
                confirm_exclusions()
            except WORKFLOW_ERRORS as error:
                show_workflow_error(error)
                st.stop()

            st.rerun()

    if st.button("Situation bearbeiten", width="stretch"):
        st.session_state.step = "business_intake"
        st.rerun()


elif st.session_state.step == "model_input_form":
    st.subheader("Angaben zur Berechnung")
    graph_state = st.session_state.graph_state or {}
    modular_state = graph_state.get("modular_model")
    specification = (
        modular_state.get("specification")
        if modular_state is not None
        else None
    )

    if modular_state is None or specification is None:
        st.error("Der modulare Modellzustand ist unvollständig.")
        st.stop()

    try:
        submissions = render_modular_input_page(specification)
    except (TypeError, ValueError) as error:
        st.error("Bitte korrigieren Sie die Eingaben:")
        st.error(str(error))
        st.stop()

    if submissions is None:
        st.stop()

    try:
        submit_model_inputs(submissions)
    except WORKFLOW_ERRORS as error:
        show_workflow_error(error)
        st.stop()

    st.rerun()


elif st.session_state.step == "review":
    st.subheader("Angaben prüfen")
    st.write(
        "Bitte prüfen Sie die Berechnungsgrundlage, "
        "Modellbausteine und Modellannahmen. Die Berechnung "
        "kann gestartet werden, sobald alle Angaben bestätigt sind."
    )

    graph_state = st.session_state.graph_state or {}
    modular_state = graph_state.get("modular_model")
    proposal = (
        modular_state.get("proposal")
        if modular_state is not None
        else None
    )
    specification = (
        modular_state.get("specification")
        if modular_state is not None
        else None
    )

    if modular_state is None or proposal is None or specification is None:
        st.error("Die modulare Modellprüfung ist unvollständig.")
        st.stop()

    explanation = modular_state.get("review_explanation")

    if not isinstance(explanation, str) or not explanation.strip():
        try:
            workflow = build_graph(get_or_create_llm())
            graph_state = workflow.invoke(
                {
                    **graph_state,
                    "control": {
                        **graph_state.get("control", {}),
                        "event": (
                            EVENT_MODULAR_REVIEW_EXPLANATION_REQUESTED
                        ),
                    },
                }
            )
        except WORKFLOW_ERRORS as error:
            show_workflow_error(error)
            st.stop()

        st.session_state.graph_state = graph_state
        modular_state = graph_state["modular_model"]
        proposal = modular_state["proposal"]
        specification = modular_state["specification"]
        explanation = modular_state.get("review_explanation")

    try:
        start_optimization_clicked = render_modular_review_page(
            proposal,
            specification,
            explanation=explanation,
            pending_core_assumptions=modular_state.get(
                "pending_core_assumptions", []
            ),
            core_assumption_evidence=modular_state.get(
                "core_assumption_evidence", []
            ),
        )
    except ValueError as error:
        st.error("Die modulare Modellprüfung konnte nicht aufgebaut werden.")
        st.error(str(error))
        st.stop()

    if start_optimization_clicked:
        try:
            start_optimization()
        except WORKFLOW_ERRORS as error:
            show_workflow_error(error)
            st.stop()

        st.rerun()


elif st.session_state.step == "result":
    graph_state = st.session_state.graph_state or {}
    modular_state = graph_state.get("modular_model")
    specification = (
        modular_state.get("specification")
        if modular_state is not None
        else None
    )
    result = (
        modular_state.get("optimization_result")
        if modular_state is not None
        else None
    )

    if modular_state is None or specification is None or result is None:
        st.error("Das modulare Optimierungsergebnis ist unvollständig.")
        st.stop()

    explanation_requested = render_modular_result_page(
        specification,
        result,
        explanation=modular_state.get("result_explanation"),
    )

    if explanation_requested:
        try:
            request_result_explanation()
        except WORKFLOW_ERRORS as error:
            show_workflow_error(error)
            st.stop()

        st.rerun()

    st.divider()

    if st.button(
        "Neue Optimierung starten",
        width="stretch",
    ):
        reset_workflow()
        st.rerun()


else:
    st.session_state.step = "business_intake"
    st.rerun()
