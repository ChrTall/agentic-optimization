"""
Streamlit-Darstellung des begleitenden Chat-Assistenten.

Der Chat liegt in der Seitenleiste und ist vom ersten Schritt an
sichtbar. Er kann Eingaben übernehmen und dieselben
Workflow-Aktionen auslösen wie die Formulare. Dafür erhält er
die Aktionen der App als Handler, damit keine Workflowlogik
doppelt existiert.
"""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import streamlit as st
from langchain_core.language_models import BaseChatModel

from agentic_optimization.agents.chat_assistant_agent import (
    ChatAssistantDecision,
    ChatMessage,
    create_chat_assistant_agent,
    recall_from_statements,
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
    MAX_UNANSWERED_REPEATS,
    QuickReply,
    build_known_facts_summary,
    build_next_question,
    build_progress_label,
    build_quick_replies,
    build_recall_items,
    build_stuck_message,
    collect_user_statements,
    format_recall_note,
    check_product_rows,
    flatten_missing_inputs,
    get_open_business_questions,
    get_product_table_schema,
    merge_product_rows,
    resolve_input_submissions,
    resolve_question_answers,
)
from agentic_optimization.ui.modular_input_builder import (
    build_modular_input_layout,
)

LOGGER = logging.getLogger(__name__)

MESSAGES_KEY = "chat_messages"
STEP_KEY = "chat_announced_step"
QUESTION_KEY = "chat_question_key"
PRODUCT_ROWS_KEY = "chat_product_rows"
PRODUCT_ROWS_VERSION_KEY = "chat_product_rows_version"
RECALL_ATTEMPTS_KEY = "chat_recall_attempts"
REPEAT_KEY = "chat_repeat_state"
ASKED_BEFORE_KEY = "chat_question_before_message"

SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"] {
    width: 420px !important;
    min-width: 420px !important;
}
@media (max-width: 900px) {
    section[data-testid="stSidebar"] {
        width: 100vw !important;
        min-width: 0 !important;
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
        PRODUCT_ROWS_KEY,
        RECALL_ATTEMPTS_KEY,
        REPEAT_KEY,
        ASKED_BEFORE_KEY,
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


def _recall_known_answers(handlers: ChatWorkflowHandlers) -> None:
    """
    Übernimmt offene Angaben, die der Nutzer bereits genannt hat.

    Läuft vor jeder automatischen Frage. Jede offene Angabe wird pro
    Gesprächsstand höchstens einmal geprüft, damit keine unnötigen
    Modellaufrufe entstehen.
    """

    step = st.session_state.get("step")

    if step not in {"business_clarification", "model_input_form"}:
        return

    statements = collect_user_statements(_messages())

    if not statements:
        return

    graph_state = st.session_state.get("graph_state") or {}

    try:
        items = build_recall_items(step, graph_state)
    except (KeyError, TypeError, ValueError):
        LOGGER.exception("Offene Angaben für das Erinnern fehlen.")
        return

    digest = hashlib.sha1(statements.encode("utf-8")).hexdigest()
    attempts = st.session_state.setdefault(RECALL_ATTEMPTS_KEY, {})
    done = set(attempts.get(digest, []))
    pending = [item for item in items if item["key"] not in done]

    if not pending:
        return

    attempts[digest] = sorted(done | {item["key"] for item in pending})

    try:
        recalled = recall_from_statements(
            handlers.get_llm(),
            statements=statements,
            open_items=[
                {key: value for key, value in item.items() if key != "key"}
                for item in pending
            ],
        )
    except Exception:  # noqa: BLE001 – Erinnern ist eine Komfortfunktion
        LOGGER.exception("Erinnern an bisherige Angaben fehlgeschlagen.")
        return

    if not recalled:
        return

    accepted: list[dict[str, Any]] = []

    try:
        if step == "business_clarification":
            answers = resolve_question_answers(
                [
                    {"question_id": answer["id"], "option_id": answer["value"]}
                    for answer in recalled
                ],
                get_open_business_questions(graph_state),
            )
            accepted_ids = {answer["question_id"] for answer in answers}
            accepted = [a for a in recalled if a["id"] in accepted_ids]

            if answers:
                handlers.run_business_intake(
                    st.session_state.get("raw_problem_description", ""),
                    answers,
                )
        else:
            missing_inputs = flatten_missing_inputs(
                build_modular_input_layout(
                    graph_state["modular_model"]["specification"]
                )
            )
            submissions = []

            for answer in recalled:
                resolved, _ = resolve_input_submissions(
                    [{"input_id": answer["id"], "value": answer["value"]}],
                    missing_inputs,
                )

                if resolved:
                    submissions.extend(resolved)
                    accepted.append(answer)

            if submissions:
                st.session_state.pop("modular_product_input_table", None)
                handlers.submit_model_inputs(submissions)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Übernahme erinnerter Angaben fehlgeschlagen.")
        return

    note = format_recall_note(pending, accepted)

    if note:
        _add_message("assistant", note)


def _announce_current_step() -> None:
    """
    Führt das Gespräch: Bei einem neuen Schritt folgt eine kurze
    Einleitung, danach wird immer genau die nächste offene
    Angabe erfragt.
    """

    step = st.session_state.get("step", "business_intake")
    product_rows, _ = get_chat_product_rows()

    try:
        messages, question_key = build_follow_up_messages(
            step,
            st.session_state.get("graph_state"),
            product_rows=product_rows,
            previous_step=st.session_state.get(STEP_KEY),
            previous_question_key=st.session_state.get(QUESTION_KEY),
        )
    except (KeyError, TypeError, ValueError):
        LOGGER.exception("Chat-Folgefrage konnte nicht erzeugt werden.")
        return

    st.session_state[STEP_KEY] = step
    st.session_state[QUESTION_KEY] = question_key

    for message in messages:
        if not _messages() or _messages()[-1]["content"] != message:
            _add_message("assistant", message)

    _offer_way_out_if_stuck(step, question_key, product_rows)


def _offer_way_out_if_stuck(
    step: str,
    question_key: str | None,
    product_rows: list[dict[str, Any]],
) -> None:
    """Bietet Auswege an, wenn dieselbe Frage offen bleibt."""

    if ASKED_BEFORE_KEY not in st.session_state:
        return

    asked_before = st.session_state.pop(ASKED_BEFORE_KEY)
    repeat = st.session_state.get(REPEAT_KEY) or {}

    if question_key is None or question_key != asked_before:
        st.session_state[REPEAT_KEY] = {}
        return

    unanswered = (
        repeat.get("unanswered", 0) + 1
        if repeat.get("question_key") == question_key
        else 1
    )
    notified = (
        repeat.get("notified", False)
        if repeat.get("question_key") == question_key
        else False
    )

    if unanswered >= MAX_UNANSWERED_REPEATS and not notified:
        message = build_stuck_message(
            build_next_question(
                step,
                st.session_state.get("graph_state"),
                product_rows=product_rows,
            )
        )

        if message:
            _add_message("assistant", message)
            notified = True

    st.session_state[REPEAT_KEY] = {
        "question_key": question_key,
        "unanswered": unanswered,
        "notified": notified,
    }


# ---------------------------------------------------------------------------
# Verarbeitung einer Nutzernachricht
# ---------------------------------------------------------------------------


def _apply_business_decision(
    decision: ChatAssistantDecision,
    user_message: str,
    handlers: ChatWorkflowHandlers,
) -> None:
    graph_state = st.session_state.get("graph_state") or {}
    description = st.session_state.get(
        "raw_problem_description",
        "",
    )
    answers = resolve_question_answers(
        decision["question_answers"],
        get_open_business_questions(graph_state),
    )
    new_description = (
        append_to_description(description, user_message)
        if decision["use_message_as_description"]
        else description
    )

    if new_description != description or answers:
        handlers.run_business_intake(new_description, answers)


def _apply_product_decision(
    decision: ChatAssistantDecision,
    handlers: ChatWorkflowHandlers,
) -> list[str]:
    graph_state = st.session_state.get("graph_state") or {}
    schema = get_product_table_schema(graph_state)

    if schema is None:
        return []

    rows = st.session_state.get(PRODUCT_ROWS_KEY, [])

    if decision["products"]:
        rows = merge_product_rows(rows, decision["products"], schema)
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

    missing_inputs = flatten_missing_inputs(
        build_modular_input_layout(specification)
    )
    submissions, errors = resolve_input_submissions(
        decision["input_values"],
        missing_inputs,
    )

    if submissions:
        # Der Tabelleneditor speichert Änderungen relativ zu den
        # Ausgangszeilen. Nach einer Übernahme ändern sich die
        # Zeilen, daher wird sein Zustand verworfen.
        st.session_state.pop("modular_product_input_table", None)
        handlers.submit_model_inputs(submissions)

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
        user_statements=collect_user_statements(_messages()),
    )
    agent = create_chat_assistant_agent(llm)

    try:
        decision = agent(
            context=context,
            history=_messages()[:-1],
            user_message=user_message,
        )
    except Exception as error:  # noqa: BLE001 – Provider-Fehler sind vielfältig
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

    _add_message("assistant", decision["reply"] or "Alles klar.")
    action = decision["action"]

    try:
        if action == "restart":
            handlers.restart()
            return

        if action == "edit_description":
            handlers.edit_description()
            return

        notes: list[str] = []

        if step in {"business_intake", "business_clarification"}:
            _apply_business_decision(decision, user_message, handlers)
        elif step == "product_table":
            notes = _apply_product_decision(decision, handlers)
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


# ---------------------------------------------------------------------------
# Antwort-Buttons
# ---------------------------------------------------------------------------


def _handle_quick_reply(
    reply: QuickReply,
    handlers: ChatWorkflowHandlers,
) -> None:
    """Führt einen angeklickten Antwortvorschlag aus."""

    if reply["kind"] == "message":
        _add_message("user", reply["value"])
        st.session_state[ASKED_BEFORE_KEY] = st.session_state.get(
            QUESTION_KEY
        )
        _process_user_message(reply["value"], handlers)
        return

    try:
        if reply["kind"] == "answer":
            _add_message("user", reply["label"])
            graph_state = st.session_state.get("graph_state") or {}
            answers = resolve_question_answers(
                [reply["value"]],
                get_open_business_questions(graph_state),
            )

            if answers:
                handlers.run_business_intake(
                    st.session_state.get("raw_problem_description", ""),
                    answers,
                )
            return

        action = reply["value"]

        if action == "restart":
            handlers.restart()
        elif action == "start_optimization":
            _add_message("user", "Bitte die Berechnung starten.")
            handlers.start_optimization()
        elif action == "explain_result":
            _add_message("user", "Bitte erklären Sie mir das Ergebnis.")
            handlers.request_result_explanation()
            explanation = (
                (st.session_state.get("graph_state") or {})
                .get("modular_model", {})
                .get("result_explanation")
            )

            if isinstance(explanation, str) and explanation.strip():
                _add_message("assistant", explanation)
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Antwort-Button konnte nicht ausgeführt werden.")
        _add_message(
            "assistant",
            describe_llm_error(error)
            if is_llm_service_error(error)
            else f"Das hat leider nicht geklappt: {error}",
        )


def _render_quick_replies(handlers: ChatWorkflowHandlers) -> bool:
    """Zeigt Antwortvorschläge; True, wenn einer angeklickt wurde."""

    step = st.session_state.get("step", "business_intake")
    graph_state = st.session_state.get("graph_state")
    product_rows, _ = get_chat_product_rows()

    try:
        replies = build_quick_replies(
            step,
            graph_state,
            build_next_question(step, graph_state, product_rows=product_rows),
        )
    except (KeyError, TypeError, ValueError):
        LOGGER.exception("Antwortvorschläge konnten nicht erzeugt werden.")
        return False

    if not replies:
        return False

    if step == "business_intake":
        st.caption("Oder probieren Sie ein Beispiel aus:")

    for reply in replies:
        if st.button(
            reply["label"],
            key=f"chat_quick_reply:{reply['key']}",
            width="stretch",
        ):
            with st.spinner("Einen Moment …"):
                _handle_quick_reply(reply, handlers)
            return True

    return False


def _render_known_facts() -> None:
    """Macht sichtbar, was der Assistent schon weiß."""

    try:
        lines = build_known_facts_summary(st.session_state.get("graph_state"))
    except (KeyError, TypeError, ValueError):
        LOGGER.exception("Bekannte Angaben konnten nicht angezeigt werden.")
        return

    with st.expander(f"🧠 Das weiß ich schon ({len(lines)})"):
        if lines:
            st.markdown("\n".join(lines))
        else:
            st.caption(
                "Noch nichts – erzählen Sie mir von Ihrer Situation."
            )


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
    _recall_known_answers(handlers)
    _announce_current_step()

    with st.sidebar:
        st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
        st.subheader("💬 Ihr Assistent")
        progress = build_progress_label(
            st.session_state.get("step", "business_intake")
        )
        st.caption(
            (f"**{progress}** · " if progress else "")
            + "Schreiben Sie einfach drauflos – ich trage Ihre "
            "Angaben ins Formular ein und führe Sie durch die "
            "Schritte."
        )
        _render_known_facts()

        message_box = st.container(height=500, border=False)

        with message_box:
            for message in _messages():
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Fester Container: Die Zahl der Buttons wechselt, die Position
        # des Eingabefelds darunter soll stabil bleiben.
        with st.container():
            if _render_quick_replies(handlers):
                st.rerun()

        user_message = st.chat_input(
            "Nachricht an den Assistenten …",
            key="chat_assistant_input",
        )

        if not user_message or not user_message.strip():
            return

        _add_message("user", user_message)
        st.session_state[ASKED_BEFORE_KEY] = st.session_state.get(
            QUESTION_KEY
        )

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