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
from agentic_optimization.models.specification_updates import (
    list_optional_components,
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
from agentic_optimization.i18n import t
from agentic_optimization.ui.chat_mentions import (
    detect_component_mentions,
    value_conflicts_with_text,
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
TURN_KEY = "chat_turn"
HANDLED_MENTIONS_KEY = "chat_handled_component_mentions"
NEW_TURN_KEY = "chat_new_turn_pending"

LANGUAGE_KEY = "ui_language"
LANGUAGE_LABELS = {"de": "Deutsch", "en": "English"}
CHAT_WIDTH_KEY = "chat_width"
CHAT_WIDTHS = {
    "Schmal": "360px",
    "Mittel": "420px",
    "Breit": "33vw",
    "Halbe Breite": "50vw",
}
DEFAULT_CHAT_WIDTH = "Mittel"


def build_sidebar_css(width_option: str | None) -> str:
    """CSS für die Chatbreite (bis zur halben Fensterbreite)."""

    width = CHAT_WIDTHS.get(width_option or "", CHAT_WIDTHS[DEFAULT_CHAT_WIDTH])

    return f"""
<style>
section[data-testid="stSidebar"] {{
    width: {width} !important;
    min-width: min({width}, 50vw) !important;
    max-width: 50vw !important;
}}
@media (max-width: 900px) {{
    section[data-testid="stSidebar"] {{
        width: 100vw !important;
        min-width: 0 !important;
        max-width: 100vw !important;
    }}
}}
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
    submit_specification_edits: Callable[[list, list | None], None]
    continue_to_review: Callable[[], None]
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


def _on_language_change() -> None:
    """Zeigt die automatischen Texte des Assistenten in der neuen Sprache.

    Hat der Nutzer noch nichts geschrieben, beginnt das Gespräch neu
    (Begrüßung in der neuen Sprache). Sonst bleibt der Verlauf erhalten,
    und die aktuell offene Frage wird in der neuen Sprache erneut gestellt.
    """

    from agentic_optimization.i18n import set_language

    set_language(st.session_state.get(LANGUAGE_KEY) or "de")
    messages = st.session_state.get(MESSAGES_KEY) or []

    if not any(message["role"] == "user" for message in messages):
        st.session_state[MESSAGES_KEY] = []
        st.session_state[STEP_KEY] = None

    st.session_state[QUESTION_KEY] = None


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
        HANDLED_MENTIONS_KEY,
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
    """Fügt eine Nachricht hinzu.

    Alles, was der Assistent innerhalb einer Runde sagt (Antwort,
    Hinweise, Einleitung, nächste Frage), landet in **einer** Sprechblase.
    Eine neue Runde beginnt mit jeder Nutzernachricht und mit jeder
    Eingabe im Formular.
    """

    content = content.strip()

    if not content:
        return

    turn = st.session_state.get(TURN_KEY, 0)
    messages = _messages()

    if (
        role == "assistant"
        and messages
        and messages[-1]["role"] == "assistant"
        and messages[-1].get("turn") == turn
    ):
        last = messages[-1]
        last["parts"] = [*last.get("parts", [last["content"]]), content]
        last["content"] = "\n\n".join(last["parts"])
        return

    messages.append({
        "role": role, "content": content, "turn": turn, "parts": [content],
    })


def _last_assistant_part() -> str | None:
    """Letzter Abschnitt der letzten Sprechblase des Assistenten."""

    messages = _messages()

    if not messages or messages[-1]["role"] != "assistant":
        return None

    return messages[-1].get("parts", [messages[-1]["content"]])[-1]


def _start_turn_if_pending() -> None:
    """Neue Runde, wenn seit dem letzten Durchlauf etwas Neues passiert
    ist (Nutzernachricht, Klick oder Formulareingabe).

    Durchläufe, die der Chat selbst nach einer Nachricht auslöst,
    gehören zur selben Runde.
    """

    if st.session_state.pop(NEW_TURN_KEY, False):
        st.session_state[TURN_KEY] = st.session_state.get(TURN_KEY, 0) + 1


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

            ids_by_index = {
                item["id"]: missing_input
                for item, missing_input in zip(
                    build_recall_items(step, graph_state), missing_inputs
                )
            }

            for answer in recalled:
                missing_input = ids_by_index.get(answer["id"])

                # Ein Zitat wie „Verkaufspreis von 25 €“ gehört nicht zu
                # den Einkaufskosten – solche Zuordnungen verwerfen.
                if missing_input is not None and (
                    value_conflicts_with_text(
                        missing_input["field"]["name"],
                        answer["value"],
                        answer["quote"],
                    )
                    or value_conflicts_with_text(
                        missing_input["field"]["name"],
                        answer["value"],
                        _sentence_around(answer["quote"], statements),
                    )
                ):
                    LOGGER.info(
                        "Erinnerter Wert verworfen (passt nicht zum Feld): %s",
                        answer,
                    )
                    continue

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


def _sentence_around(quote: str, text: str) -> str:
    """Der ganze Satz, in dem ``quote`` steht (sonst das Zitat selbst)."""

    position = text.find(quote)

    if position < 0:
        return quote

    start = max(
        text.rfind(mark, 0, position) for mark in (". ", "! ", "? ", "\n")
    )
    ends = [
        index
        for index in (
            text.find(mark, position + len(quote))
            for mark in (". ", "! ", "? ", "\n")
        )
        if index >= 0
    ]
    end = min(ends) if ends else len(text)
    return text[start + 1 if start >= 0 else 0:end + 1].strip()


def _missing_inputs_for(specification: ProductSpecification | Any) -> list[Any]:
    return flatten_missing_inputs(build_modular_input_layout(specification))


def _current_specification() -> Any:
    graph_state = st.session_state.get("graph_state") or {}
    return (graph_state.get("modular_model") or {}).get("specification")


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _apply_component_mentions(
    texts: list[tuple[str, str]],
    handlers: "ChatWorkflowHandlers",
) -> list[str]:
    """Wählt ausdrücklich genannte Bausteine an und trägt Werte ein.

    ``texts`` sind (Schlüssel, Text)-Paare; jeder Schlüssel wird nur
    einmal verarbeitet. Bausteine, die der Nutzer später im Formular
    abwählt, werden daher nicht erneut hinzugefügt.
    """

    specification = _current_specification()

    if specification is None:
        return []

    handled = st.session_state.setdefault(HANDLED_MENTIONS_KEY, [])
    notes: list[str] = []

    for key, text in texts:
        if key in handled:
            continue

        handled.append(key)

        for mention in detect_component_mentions(text):
            specification = _current_specification()
            optional = {
                component["component_id"]: component
                for component in list_optional_components(specification)
                if not component["locked"]
            }
            component_id = next(
                (cid for cid in mention["component_ids"] if cid in optional),
                None,
            )

            if component_id is None:
                continue

            name = optional[component_id]["name"]
            added = False

            if not optional[component_id]["active"]:
                active_ids = [
                    cid for cid, component in optional.items()
                    if component["active"]
                ]
                handlers.submit_specification_edits(
                    [], active_ids + [component_id]
                )
                added = True
                specification = _current_specification()

            filled = False

            if mention["value"] is not None:
                missing_inputs = _missing_inputs_for(specification)
                product_ids = [
                    product["product_id"]
                    for product in specification["products"]
                ]
                values = []

                for index, missing_input in enumerate(missing_inputs):
                    if missing_input.get("component_id") != component_id:
                        continue

                    constraint = specification["constraints"][
                        missing_input["constraint_index"]
                    ]
                    product_id = constraint.get("product_id")

                    # Bei mehreren Produkten nur, wenn das Produkt im
                    # Satz genannt ist.
                    if (
                        product_id is not None
                        and len(product_ids) > 1
                        and product_id.lower() not in text.lower()
                    ):
                        continue

                    values.append({
                        "input_id": f"i{index + 1}",
                        "value": mention["value"],
                    })

                submissions, _ = resolve_input_submissions(
                    values, missing_inputs
                )

                if submissions:
                    st.session_state.pop("modular_product_input_table", None)
                    handlers.submit_model_inputs(submissions)
                    filled = True

            if filled:
                notes.append(t(
                    "Übernommen: **{name}** = {value} _(„{quote}“)_",
                    name=name,
                    value=_format_number(mention["value"]),
                    quote=mention["segment"],
                ))
            elif added:
                notes.append(t(
                    "Ergänzt: **{name}** _(„{quote}“)_ – den Wert frage ich gleich ab.",
                    name=name,
                    quote=mention["segment"],
                ))

    return notes


def _apply_mentioned_components(handlers: "ChatWorkflowHandlers") -> None:
    """Übernimmt im Gespräch genannte Vorgaben, sobald Formular 3 offen ist.

    Erfasst auch Vorgaben aus früheren Nachrichten (z. B. aus der
    Beschreibung), die bei der Modellauswahl nicht berücksichtigt wurden.
    """

    if st.session_state.get("step") != "model_input_form":
        return

    # Beim (erneuten) Betreten von Formular 3 gilt ein frisches Modell.
    if st.session_state.get(STEP_KEY) != "model_input_form":
        st.session_state[HANDLED_MENTIONS_KEY] = []

    texts = [
        (f"{index}:{message['content']}", message["content"])
        for index, message in enumerate(_messages())
        if message["role"] == "user"
    ]

    try:
        notes = _apply_component_mentions(texts, handlers)
    except Exception:  # noqa: BLE001 – Komfortfunktion
        LOGGER.exception("Genannte Bausteine konnten nicht übernommen werden.")
        return

    for note in notes:
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
        if _last_assistant_part() != message.strip():
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
            t("Bevor es weitergeht, fehlt noch:\n")
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

    if specification is None:
        return []

    notes: list[str] = []
    errors: list[str] = []
    user_message = next(
        (
            message["content"]
            for message in reversed(_messages())
            if message["role"] == "user"
        ),
        "",
    )

    # 1. Werte zuerst – ihre Kennungen beziehen sich auf die aktuellen
    #    offenen Eingaben, die sich durch neue Bausteine verschieben.
    if decision["input_values"]:
        missing_inputs = _missing_inputs_for(specification)
        inputs_by_id = {
            f"i{index + 1}": missing_input
            for index, missing_input in enumerate(missing_inputs)
        }
        # Werte verwerfen, die laut Nachricht zu einer anderen Größe
        # gehören (z. B. Verkaufspreis als Einkaufskosten).
        values = [
            item
            for item in decision["input_values"]
            if item["input_id"] not in inputs_by_id
            or not value_conflicts_with_text(
                inputs_by_id[item["input_id"]]["field"]["name"],
                item["value"],
                user_message,
            )
        ]
        submissions, errors = resolve_input_submissions(values, missing_inputs)

        if submissions:
            # Der Tabelleneditor speichert Änderungen relativ zu den
            # Ausgangszeilen. Nach einer Übernahme ändern sich die
            # Zeilen, daher wird sein Zustand verworfen.
            st.session_state.pop("modular_product_input_table", None)
            handlers.submit_model_inputs(submissions)

    # 2. Ausdrücklich genannte Vorgaben anwählen und Wert eintragen.
    message_index = max(
        (
            index
            for index, message in enumerate(_messages())
            if message["role"] == "user"
        ),
        default=-1,
    )
    notes += _apply_component_mentions(
        [(f"{message_index}:{user_message}", user_message)],
        handlers,
    )

    # 3. Weitere vom Assistenten erkannte Bausteine.
    if decision["add_components"]:
        specification = _current_specification()
        optional = {
            component["component_id"]: component
            for component in list_optional_components(specification)
            if not component["locked"]
        }
        to_add = [
            component_id
            for component_id in decision["add_components"]
            if component_id in optional
            and not optional[component_id]["active"]
        ]

        if to_add:
            active_ids = [
                cid for cid, component in optional.items()
                if component["active"]
            ]
            handlers.submit_specification_edits([], active_ids + to_add)
            notes.append(
                t("Ergänzt: ")
                + ", ".join(optional[cid]["name"] for cid in to_add)
                + t(". Die dafür nötigen Werte frage ich gleich ab.")
            )

    if (
        decision["action"] == "continue_to_review"
        and not decision["input_values"]
        and not notes
    ):
        handlers.continue_to_review()
        return notes

    if errors:
        notes.append(
            t("Diese Werte konnte ich nicht übernehmen:\n")
            + "\n".join(f"- {error}" for error in errors)
        )

    return notes


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
            t("Ich kann gerade nicht antworten: {error}", error=error),
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
                t("Entschuldigung, da ist gerade etwas schiefgelaufen. Bitte versuchen Sie es noch einmal oder nutzen Sie das Formular.")
            ),
        )
        return

    _add_message("assistant", decision["reply"] or t("Alles klar."))
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
                t("Beim Übernehmen ist leider ein Fehler aufgetreten: {error}", error=error)
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
        elif action == "continue_to_review":
            _add_message("user", t("Weiter zur Überprüfung."))
            handlers.continue_to_review()
        elif action == "start_optimization":
            _add_message("user", t("Bitte die Berechnung starten."))
            handlers.start_optimization()
        elif action == "explain_result":
            _add_message("user", t("Bitte erklären Sie mir das Ergebnis."))
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
            else t("Das hat leider nicht geklappt: {error}", error=error),
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

    for reply in replies:
        if st.button(
            reply["label"],
            key=f"chat_quick_reply:{reply['key']}",
            width="stretch",
        ):
            with st.spinner(t("Einen Moment …")):
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

    with st.expander(t("🧠 Das weiß ich schon ({value1})", value1=len(lines))):
        if lines:
            st.markdown("\n".join(lines))
        else:
            st.caption(
                t("Noch nichts – erzählen Sie mir von Ihrer Situation.")
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
    _start_turn_if_pending()
    _apply_mentioned_components(handlers)
    _recall_known_answers(handlers)
    _announce_current_step()

    with st.sidebar:
        st.segmented_control(
            "Sprache / Language",
            options=list(LANGUAGE_LABELS),
            format_func=LANGUAGE_LABELS.get,
            default="de",
            key=LANGUAGE_KEY,
            on_change=_on_language_change,
            label_visibility="collapsed",
        )
        width_option = st.segmented_control(
            t("Chatbreite"),
            options=list(CHAT_WIDTHS),
            format_func=t,
            default=DEFAULT_CHAT_WIDTH,
            key=CHAT_WIDTH_KEY,
            label_visibility="collapsed",
        )
        st.markdown(build_sidebar_css(width_option), unsafe_allow_html=True)
        st.subheader(t("💬 Ihr Assistent"))
        progress = build_progress_label(
            st.session_state.get("step", "business_intake")
        )
        st.caption(
            (f"**{progress}** · " if progress else "")
            + t("Schreiben Sie einfach drauflos – ich trage Ihre Angaben ins Formular ein und führe Sie durch die Schritte.")
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
            t("Nachricht an den Assistenten …"),
            key="chat_assistant_input",
        )

        if not user_message or not user_message.strip():
            # Der nächste Durchlauf kommt von einer neuen Aktion.
            st.session_state[NEW_TURN_KEY] = True
            return

        _add_message("user", user_message)
        st.session_state[ASKED_BEFORE_KEY] = st.session_state.get(
            QUESTION_KEY
        )

        with message_box:
            with st.chat_message("user"):
                st.markdown(user_message)

            with st.chat_message("assistant"):
                with st.spinner(t("Einen Moment …")):
                    _process_user_message(
                        user_message.strip(),
                        handlers,
                    )

    st.rerun()