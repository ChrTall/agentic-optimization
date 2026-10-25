"""
Streamlit-Darstellung des begleitenden Chat-Assistenten.

Der Chat liegt in der Seitenleiste und ist vom ersten Schritt an
sichtbar. Er kann Eingaben übernehmen und dieselben
Workflow-Aktionen auslösen wie die Formulare. Dafür erhält er
die Aktionen der App als Handler, damit keine Workflowlogik
doppelt existiert.
"""

import logging
from copy import deepcopy
from agentic_optimization.models.product_names import clean_product_name
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import streamlit as st
from langchain_core.language_models import BaseChatModel

from agentic_optimization.agents.chat_assistant_agent import (
    ChatAssistantDecision,
    ChatMessage,
    create_chat_assistant_agent,
)
from agentic_optimization.agents.chat_intake_validation import validate_chat_intake
from agentic_optimization.ui.chat_proposals import (
    PENDING_KEY, EDIT_KEY, is_direct_answer, match_current_option, proposal_message, proposal_sentence,
)
from agentic_optimization.ui.chat_diagnostics import (
    ENABLED_KEY, RECORDS_KEY, diagnostic_sink, intake_snapshot,
    render_diagnostics, start_diagnostic,
)
from agentic_optimization.llm.errors import (
    describe_llm_error,
    is_llm_service_error,
)
from agentic_optimization.models.specification import (
    ProductSpecification,
)
from agentic_optimization.state import (
    BusinessQuestionAnswer,
    ModularInputSubmission,
)
from agentic_optimization.ui.chat_assistant_builder import (
    append_to_description,
    build_chat_context,
    build_follow_up_messages,
    check_product_rows,
    get_open_business_questions,
    get_answerable_business_questions,
    business_update_message,
    get_product_table_schema,
    merge_product_rows,
    resolve_input_submissions,
    resolve_question_answers,
)
from agentic_optimization.ui.modular_components import (
    add_component, component_options, editable_chat_inputs, input_address, update_model_draft,
)
from agentic_optimization.models.specification_updates import apply_modular_input_value

LOGGER = logging.getLogger(__name__)

MESSAGES_KEY = "chat_messages"
STEP_KEY = "chat_announced_step"
QUESTION_KEY = "chat_question_key"
CONFIRM_FOLLOW_UP_KEY = "chat_confirm_follow_up"
PRODUCT_ROWS_KEY = "chat_product_rows"
PRODUCT_ROWS_VERSION_KEY = "chat_product_rows_version"

SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: min(420px, 50vw) !important;
    max-width: min(600px, 50vw) !important;
}
@media (max-width: 900px) {
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 100vw !important;
        min-width: 0 !important;
        max-width: 100vw !important;
    }
}
</style>
"""

HALF_WIDTH_CSS = """
<style>
@media (min-width: 901px) {
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 50vw !important;
        min-width: 50vw !important;
        max-width: 50vw !important;
    }
}
</style>
"""


@dataclass(frozen=True)
class ChatWorkflowHandlers:
    """Aktionen der App, die der Chat auslösen darf."""

    get_llm: Callable[[], BaseChatModel]
    run_business_intake: Callable[
        [str, list[BusinessQuestionAnswer]],
        None,
    ]
    submit_products: Callable[[list[ProductSpecification]], None]
    submit_model_inputs: Callable[
        [list[ModularInputSubmission]],
        None,
    ]
    confirm_exclusions: Callable[[], None]
    start_optimization: Callable[[], None]
    request_result_explanation: Callable[[], None]
    edit_description: Callable[[], None]
    restart: Callable[[], None]
    retry_preparation: Callable[[], None] | None = None


# ---------------------------------------------------------------------------
# Session-State
# ---------------------------------------------------------------------------


def initialize_chat_state() -> None:
    st.session_state.setdefault(MESSAGES_KEY, [])
    st.session_state.setdefault(STEP_KEY, None)
    st.session_state.setdefault(QUESTION_KEY, None)
    st.session_state.setdefault(PRODUCT_ROWS_KEY, [])
    st.session_state.setdefault(PRODUCT_ROWS_VERSION_KEY, 0)


def reset_chat_state() -> None:
    """Beginnt ein neues Gespräch (z. B. bei neuer Optimierung)."""

    for key in (
        MESSAGES_KEY,
        STEP_KEY,
        QUESTION_KEY,
        CONFIRM_FOLLOW_UP_KEY,
        PRODUCT_ROWS_KEY,
        ENABLED_KEY,
        RECORDS_KEY,
        PENDING_KEY,
        EDIT_KEY,
        "modular_input_drafts",
        "modular_input_known_values",
        "modular_table_base",
        "modular_product_input_table",
        "modular_additions_confirmed",
        "modular_component_selection",
        "modular_component_removal",
    ):
        st.session_state.pop(key, None)

    # Die Version wird bewusst weitergezählt, damit der
    # Tabelleneditor garantiert neu aufgebaut wird.
    st.session_state[PRODUCT_ROWS_VERSION_KEY] = (
        st.session_state.get(PRODUCT_ROWS_VERSION_KEY, 0) + 1
    )


def get_chat_product_rows() -> tuple[list[dict[str, Any]], int]:
    """Vom Chat erfasste Produktzeilen und Editor-Version."""

    return (
        list(st.session_state.get(PRODUCT_ROWS_KEY, [])),
        st.session_state.get(PRODUCT_ROWS_VERSION_KEY, 0),
    )


def _messages() -> list[ChatMessage]:
    return st.session_state[MESSAGES_KEY]


def _add_message(role: str, content: str) -> None:
    content = content.strip()

    if content:
        _messages().append({"role": role, "content": content})


def _announce_current_step() -> None:
    """
    Führt das Gespräch: Bei einem neuen Schritt folgt eine kurze
    Einleitung, danach wird immer genau die nächste offene
    Angabe erfragt.
    """

    if st.session_state.get(PENDING_KEY):
        return
    step = st.session_state.get("step", "business_intake")
    product_rows, _ = get_chat_product_rows()
    confirmed = st.session_state.get(CONFIRM_FOLLOW_UP_KEY, False)

    try:
        messages, question_key = build_follow_up_messages(
            step,
            st.session_state.get("graph_state"),
            product_rows=product_rows,
            previous_step=step if confirmed and step == "business_clarification" else st.session_state.get(STEP_KEY),
            previous_question_key=None if confirmed else st.session_state.get(QUESTION_KEY),
            additional_inputs_confirmed=st.session_state.get("modular_additions_confirmed", False),
        )
    except (KeyError, TypeError, ValueError):
        LOGGER.exception("Chat-Folgefrage konnte nicht erzeugt werden.")
        return

    st.session_state[STEP_KEY] = step
    st.session_state[QUESTION_KEY] = question_key
    if confirmed:
        st.session_state.pop(CONFIRM_FOLLOW_UP_KEY, None)
        messages = ["Ihre Angaben sind übernommen." + ("\n\n" + "\n\n".join(messages) if messages else "")]

    for message in messages:
        if not _messages() or _messages()[-1]["content"] != message:
            _add_message("assistant", message)


# ---------------------------------------------------------------------------
# Verarbeitung einer Nutzernachricht
# ---------------------------------------------------------------------------


def _apply_business_decision(
    decision: ChatAssistantDecision,
    user_message: str,
    handlers: ChatWorkflowHandlers,
    diagnostic=None,
) -> list[str]:
    graph_state = st.session_state.get("graph_state") or {}
    description = st.session_state.get(
        "raw_problem_description",
        "",
    )
    pending = st.session_state.get(PENDING_KEY)
    if pending:
        description = pending["description"]
    questions = get_answerable_business_questions(graph_state)
    context = build_chat_context(st.session_state.get("step", "business_intake"), graph_state)
    current_question_id = (context.get("current_question") or {}).get("target", {}).get("question_id")
    explicit_option = (match_current_option(user_message, next(
        (q for q in questions if q["question_id"] == current_question_id), None))
        if not pending else None)
    try:
        answers = [explicit_option] if explicit_option else validate_chat_intake(
            handlers.get_llm(), message=user_message,
            questions=questions,
            proposed_answers=decision["question_answers"],
            profile=graph_state.get("business_intake", {}).get("profile") or {},
            current_question_id=current_question_id if not pending else None,
            diagnostic=diagnostic,
            allow_assumptions=True,
        )
        if explicit_option and diagnostic is not None:
            diagnostic({"stage": "explicit_option_match", "answer": explicit_option})
    except Exception:
        LOGGER.exception("Die automatische Antwortzuordnung konnte nicht geprüft werden.")
        return ["Ihre Antwort konnte nicht zuverlässig geprüft werden. Es wurden keine Angaben "
                "automatisch übernommen. Bitte versuchen Sie es erneut oder nutzen Sie das Formular."]
    new_description = (
        append_to_description(description, user_message)
        if decision["use_message_as_description"] or answers
        else description
    )

    if new_description != description or answers:
        if not is_direct_answer(answers, current_question_id, pending):
            merged = {a["question_id"]: a for a in (pending or {}).get("answers", [])}
            merged.update({a["question_id"]: a for a in answers})
            proposed = list(merged.values())
            st.session_state[PENDING_KEY] = {
                "description": new_description, "answers": proposed, "questions": questions,
                "base": deepcopy(graph_state.get("business_intake", {})),
            }
            st.session_state[EDIT_KEY] = False
            st.session_state["chat_proposal_version"] = st.session_state.get("chat_proposal_version", 0) + 1
            if diagnostic is not None:
                diagnostic({"stage": "pending_confirmation", "answers": proposed})
            return [proposal_message(proposed, questions)]
        before = deepcopy(graph_state)
        if answers:
            intake = graph_state.get("business_intake", {})
            intake["pending_question_ids"] = list(dict.fromkeys([
                *intake.get("pending_question_ids", []), *[a["question_id"] for a in answers],
            ]))
        st.session_state["graph_state"] = graph_state
        handlers.run_business_intake(new_description, answers)
        message = business_update_message(before, st.session_state.get("graph_state"),
                                          changed_description=new_description != description)
        if answers:
            return (["Ihre Antwort ist übernommen."] if answers[0]["question_id"].startswith("core_condition:")
                    else [proposal_sentence(answers[0], questions)])
        return [message or "Ihre Antwort wurde gespeichert."]
    if decision["question_answers"]:
        return ["Ich konnte die Antwort nicht eindeutig zuordnen. Bitte präzisieren Sie die noch offene Angabe."]
    return [decision["reply"]] if decision["reply"] else []


def _confirm_business_proposal(handlers, answers=None):
    pending = st.session_state.get(PENDING_KEY)
    if not pending:
        return
    graph = st.session_state.get("graph_state") or {}
    if graph.get("business_intake", {}) != pending["base"]:
        st.session_state.pop(PENDING_KEY, None)
        _add_message("assistant", "Ihre Angaben wurden inzwischen geändert. Bitte ergänzen Sie Ihre Beschreibung erneut.")
        return
    selected = pending["answers"] if answers is None else answers
    confirmed = [{**a, "source_id": "business_clarification_form"} for a in selected]
    graph["business_intake"]["pending_question_ids"] = list(dict.fromkeys([
        *graph["business_intake"].get("pending_question_ids", []),
        *[a["question_id"] for a in confirmed],
    ]))
    st.session_state["graph_state"] = graph
    handlers.run_business_intake(pending["description"], confirmed)
    records = st.session_state.get(RECORDS_KEY, [])
    if records:
        records[-1]["events"].append({"stage": "user_confirmed_proposal", "answers": deepcopy(confirmed)})
        records[-1]["step_after"] = st.session_state.get("step")
        records[-1]["after"] = intake_snapshot(st.session_state.get("graph_state"))
    st.session_state.pop(PENDING_KEY, None)
    st.session_state.pop(EDIT_KEY, None)
    _add_message("user", "Diese Angaben übernehmen.")
    st.session_state[CONFIRM_FOLLOW_UP_KEY] = True


def _render_business_proposal(handlers):
    pending = st.session_state.get(PENDING_KEY)
    if not pending:
        return
    if (st.session_state.get("graph_state") or {}).get("business_intake", {}) != pending["base"]:
        st.session_state.pop(PENDING_KEY, None)
        st.session_state.pop(EDIT_KEY, None)
        _add_message("assistant", "Das Formular wurde inzwischen geändert. Der bisherige Vorschlag ist verworfen; bitte ergänzen Sie Ihre Angaben erneut.")
        st.rerun()
    st.caption("Diese Angaben sind vorgemerkt und noch nicht übernommen.")
    if st.session_state.get(EDIT_KEY):
        edited = []
        with st.form("edit_chat_proposal"):
            for index, answer in enumerate(pending["answers"]):
                question = next(q for q in pending["questions"] if q["question_id"] == answer["question_id"])
                labels = {o["option_id"]: o["label"] for o in question["options"]}
                labels["omit"] = "Diese Angabe nicht übernehmen"
                value = st.radio(question["prompt"], list(labels),
                    index=list(labels).index(answer["option_id"]), format_func=lambda v, labels=labels: labels[v],
                    key=f"proposal_edit_{st.session_state.get('chat_proposal_version', 0)}_{index}")
                if value != "omit":
                    edited.append({**answer, "option_id": value})
            if st.form_submit_button("Änderungen übernehmen"):
                _confirm_business_proposal(handlers, edited)
                st.rerun()
        st.caption("Weitere Details oder Korrekturen können Sie auch direkt in den Chat schreiben.")
    else:
        if st.button("Übernehmen", key="confirm_chat_proposal", type="primary"):
            _confirm_business_proposal(handlers)
            st.rerun()
        if st.button("Ändern", key="edit_chat_proposal"):
            st.session_state[EDIT_KEY] = True
            st.rerun()
    if st.button("Vorschlag verwerfen", key="discard_chat_proposal"):
        st.session_state.pop(PENDING_KEY, None)
        st.session_state.pop(EDIT_KEY, None)
        _add_message("assistant", "Der Vorschlag wurde verworfen. Sie können neue Angaben machen oder das Formular nutzen.")
        st.rerun()


def _apply_product_decision(
    decision: ChatAssistantDecision,
    handlers: ChatWorkflowHandlers,
    user_message: str = "",
) -> list[str]:
    graph_state = st.session_state.get("graph_state") or {}
    schema = get_product_table_schema(graph_state)

    if schema is None:
        return []

    rows = st.session_state.get(PRODUCT_ROWS_KEY, [])

    if decision["products"]:
        updates = deepcopy(decision["products"])
        for update in updates:
            name = clean_product_name(update.get("display_name"))
            quote = update.get("display_name_evidence")
            if not (name and isinstance(quote, str) and quote.strip()
                    and name.casefold() in quote.casefold()
                    and quote.casefold() in user_message.casefold()):
                update.pop("display_name", None)
        rows = merge_product_rows(rows, updates, schema)
        st.session_state[PRODUCT_ROWS_KEY] = rows
        st.session_state[PRODUCT_ROWS_VERSION_KEY] += 1

    if not rows:
        return []

    check = check_product_rows(rows, schema)

    if decision["action"] == "submit_products" and not check["errors"]:
        handlers.submit_products(check["products"])
        return []

    if decision["action"] == "submit_products" and check["errors"]:
        return [
            "Bevor es weitergeht, fehlt noch:\n"
            + "\n".join(f"- {error}" for error in check["errors"][:6])
        ]

    return []


def _apply_model_input_decision(
    decision: ChatAssistantDecision,
    handlers: ChatWorkflowHandlers,
) -> list[str]:
    graph_state = st.session_state.get("graph_state") or {}
    specification = (graph_state.get("modular_model") or {}).get(
        "specification"
    )

    if specification is None or not decision["input_values"]:
        return []

    targets = editable_chat_inputs(specification)
    missing_inputs = [target["input"] for target in targets]
    inputs_by_id = {f"i{index + 1}": target for index, target in enumerate(targets)}
    submissions = []
    errors = []
    seen = set()
    for value in decision["input_values"]:
        input_id = value["input_id"]
        if input_id not in inputs_by_id or input_id in seen:
            continue
        seen.add(input_id)
        parsed, problems = resolve_input_submissions([value], missing_inputs)
        errors.extend(problems)
        submissions.extend((item, inputs_by_id[input_id]) for item in parsed)

    if submissions:
        options = []
        available = {o["option_id"]: o for o in component_options(specification)}
        for submission, target in submissions:
            if target["option_id"] and target["option_id"] not in options:
                options.append(target["option_id"])
        for option_id in options:
            specification = add_component(specification, option_id)
        for submission, target in submissions:
            item = submission["missing_input"]
            # Mehrere neu aktivierte Constraints erhalten erst jetzt ihre
            # endgültigen Indizes im gemeinsamen Entwurf.
            if item["source"] == "constraint_parameter":
                if target["option_id"]:
                    option = available[target["option_id"]]
                    item = {**item, "constraint_index": next(
                        i for i, c in enumerate(specification["constraints"])
                        if c["component_id"] == option["component_id"]
                        and c.get("product_id") == option["product_id"])}
            specification = apply_modular_input_value(specification, item, submission["value"])
            st.session_state.setdefault("modular_input_known_values", {}).pop(
                input_address(specification, item), None)
        st.session_state.pop("modular_product_input_table", None)
        st.session_state["graph_state"] = update_model_draft(graph_state, specification)
        if options:
            st.session_state["modular_additions_confirmed"] = False

    if errors:
        return [
            "Diese Werte konnte ich nicht übernehmen:\n"
            + "\n".join(f"- {error}" for error in errors)
        ]

    return []


def _process_user_message(
    user_message: str,
    handlers: ChatWorkflowHandlers,
) -> None:
    step = st.session_state.get("step", "business_intake")
    graph_state = st.session_state.get("graph_state")
    product_rows, _ = get_chat_product_rows()

    try:
        llm = handlers.get_llm()
    except ValueError as error:
        _add_message(
            "assistant",
            f"Ich kann gerade nicht antworten: {error}",
        )
        return

    context = build_chat_context(
        step,
        graph_state,
        raw_description=st.session_state.get(
            "raw_problem_description",
            "",
        ),
        product_rows=product_rows,
    )
    if st.session_state.get(PENDING_KEY):
        context["pending_proposal"] = st.session_state[PENDING_KEY]
        if user_message.strip().casefold().strip(".! ") in {"ja", "übernehmen", "ja übernehmen", "ja, übernehmen", "passt"}:
            _confirm_business_proposal(handlers)
            return
    trace = start_diagnostic(user_message, step, graph_state, context)
    diagnostic = diagnostic_sink(trace)
    agent = create_chat_assistant_agent(llm, diagnostic=diagnostic)

    try:
        decision = agent(
            context=context,
            history=_messages()[:-1],
            user_message=user_message,
        )
        if diagnostic is not None:
            diagnostic({"stage": "sanitized_decision", "decision": decision})
    except Exception as error:  # noqa: BLE001 – Provider-Fehler sind vielfältig
        if trace is not None:
            trace["error_type"] = type(error).__name__
            trace["after"] = intake_snapshot(st.session_state.get("graph_state"))
        LOGGER.exception("Chat-Assistent ist fehlgeschlagen.")
        _add_message(
            "assistant",
            describe_llm_error(error)
            if is_llm_service_error(error)
            else (
                "Entschuldigung, da ist gerade etwas schiefgelaufen. "
                "Bitte versuchen Sie es noch einmal oder nutzen Sie "
                "das Formular."
            ),
        )
        return

    if step not in {"business_intake", "business_clarification"} and not decision["question_answers"]:
        _add_message("assistant", decision["reply"] or "Alles klar.")
    action = decision["action"]

    try:
        if action == "restart":
            handlers.restart()
            return

        if action == "edit_description":
            handlers.edit_description()
            return

        if step == "preparation_failed" and action == "retry_preparation":
            if handlers.retry_preparation is not None:
                handlers.retry_preparation()
            return

        notes: list[str] = []

        if step in {"business_intake", "business_clarification"} or (
            step in {"model_input_form", "review", "result"} and decision["question_answers"]
        ):
            notes = _apply_business_decision(decision, user_message, handlers, diagnostic)
        elif step == "product_table":
            notes = _apply_product_decision(decision, handlers, user_message)
        elif step == "model_input_form":
            notes = _apply_model_input_decision(decision, handlers)
        elif step == "model_unsupported" and (
            action == "confirm_exclusions"
        ):
            handlers.confirm_exclusions()
        elif step == "review" and action == "start_optimization":
            handlers.start_optimization()
        elif step == "result" and action == "explain_result":
            handlers.request_result_explanation()
            explanation = (
                (st.session_state.get("graph_state") or {})
                .get("modular_model", {})
                .get("result_explanation")
            )

            if isinstance(explanation, str) and explanation.strip():
                notes.append(explanation)

        for note in notes:
            _add_message("assistant", note)

    except Exception as error:  # noqa: BLE001
        if diagnostic is not None:
            diagnostic({"stage": "apply_failed", "error_type": type(error).__name__})
        LOGGER.exception("Chat-Aktion ist fehlgeschlagen.")
        _add_message(
            "assistant",
            describe_llm_error(error)
            if is_llm_service_error(error)
            else (
                "Beim Übernehmen ist leider ein Fehler aufgetreten: "
                f"{error}"
            ),
        )
    finally:
        if trace is not None:
            trace["step_after"] = st.session_state.get("step")
            trace["after"] = intake_snapshot(st.session_state.get("graph_state"))


# ---------------------------------------------------------------------------
# Darstellung
# ---------------------------------------------------------------------------


def render_chat_panel(handlers: ChatWorkflowHandlers) -> None:
    """
    Rendert den Chat in der Seitenleiste.

    Muss vor den Formularen aufgerufen werden, damit vom Chat
    ausgelöste Workflowschritte schon im selben Durchlauf
    berücksichtigt werden.
    """

    initialize_chat_state()
    _announce_current_step()

    with st.sidebar:
        st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
        st.subheader("💬 Ihr Assistent")
        half_width = st.toggle("Chat auf halber Fensterbreite", key="chat_half_width")
        if half_width:
            st.markdown(HALF_WIDTH_CSS, unsafe_allow_html=True)
        else:
            st.caption("Zum Vergrößern den rechten Rand des Chats nach rechts ziehen.")
        st.caption(
            "Schreiben Sie einfach drauflos – ich schlage Angaben zur "
            "Übernahme vor und führe Sie durch die "
            "Schritte."
        )
        render_diagnostics()

        message_box = st.container(height=560, border=False)

        with message_box:
            for message in _messages():
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            _render_business_proposal(handlers)

        user_message = st.chat_input(
            "Nachricht an den Assistenten …",
            key="chat_assistant_input",
        )

        if not user_message or not user_message.strip():
            return

        _add_message("user", user_message)

        with message_box:
            with st.chat_message("user"):
                st.markdown(user_message)

            with st.chat_message("assistant"):
                with st.spinner("Einen Moment …"):
                    _process_user_message(
                        user_message.strip(),
                        handlers,
                    )

    st.rerun()
