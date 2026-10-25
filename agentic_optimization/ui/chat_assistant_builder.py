"""
UI-unabhängige Logik des begleitenden Chat-Assistenten.

Dieses Modul übersetzt den Workflowzustand in einen kompakten
Kontext für das Sprachmodell und die (bereits bereinigte)
Antwort des Sprachmodells zurück in die bestehenden
Eingabeformate des Workflows. Alle Werte laufen dabei durch
dieselben Prüfungen wie die Formulareingaben.
"""

from copy import deepcopy
from typing import Any, Literal, NotRequired, TypedDict

from agentic_optimization.agents.chat_assistant_agent import (
    ChatInputValue,
    ChatProductRow,
    ChatQuestionAnswer,
)
from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
    OBJECTIVE_COMPONENTS,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)
from agentic_optimization.models.specification_updates import (
    list_optional_components,
)
from agentic_optimization.models.validator import (
    MissingInput,
)
from agentic_optimization.state import (
    BusinessQuestionAnswer,
    ModularInputSubmission,
)
from agentic_optimization.ui.business_intake_view_builder import (
    build_business_intake_view_data,
)
from agentic_optimization.ui.business_problem_form import (
    BUSINESS_QUESTIONS,
    BusinessQuestion,
)
from agentic_optimization.ui.field_renderer import (
    _format_existing_forecast_quantiles,
    _format_existing_number_list,
    parse_input_value,
)
from agentic_optimization.ui.modular_input_builder import (
    DynamicProductTableSchema,
    ModularInputLayout,
    build_dynamic_product_table_schema,
    build_modular_input_layout,
)
from agentic_optimization.ui.modular_product_list import (
    parse_dynamic_product_table,
)
from agentic_optimization.ui.modular_result_builder import (
    METRIC_LABELS,
    build_objective_label,
    STATUS_LABELS,
)
from agentic_optimization.i18n import t

CHAT_SOURCE_ID = "chat_assistant"

MAX_CONTEXT_TEXT = 12_000
MAX_USER_STATEMENTS_CHARS = 16_000

GREETING = (
    "Hallo! Ich begleite Sie Schritt für Schritt zu einer "
    "Bestellempfehlung.\n\n"
    "Erzählen Sie mir einfach in eigenen Worten, worum es "
    "geht – zum Beispiel welches Produkt, wie oft Sie "
    "bestellen und was Ihnen wichtig ist. Ich übernehme Ihre "
    "Angaben direkt in die Anwendung.\n\n"
    "Sie können genauso gut das Formular rechts nutzen – "
    "beides lässt sich jederzeit kombinieren."
)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _field_label(field: dict[str, Any]) -> str:
    unit = t(field.get("unit"))
    label = t(field["label"])

    return f"{label} ({unit})" if unit else label


def _core_name(core_id: str | None) -> str | None:
    if not core_id:
        return None

    registered_core = CORE_REGISTRY.get(core_id)

    if registered_core is None:
        return None

    return t(registered_core["definition"]["name"])


def _truncate(text: str, limit: int = MAX_CONTEXT_TEXT) -> str:
    if len(text) <= limit:
        return text

    return text[: limit - 1] + "…"


def _describe_field_for_llm(
    field: dict[str, Any],
) -> dict[str, Any]:
    description: dict[str, Any] = {
        "name": field["name"],
        "label": field["label"],
        "type": field["type"],
        "required": field["required"],
    }

    for key in ("unit", "help", "min_value", "max_value"):
        if field.get(key) is not None:
            description[key] = field[key]

    return description


def _describe_question_for_llm(
    question: BusinessQuestion,
) -> dict[str, Any]:
    return {
        "question_id": question["question_id"],
        "prompt": question["prompt"],
        "options": [
            {
                "option_id": option["option_id"],
                "label": option["label"],
                **(
                    {"help": option["help"]}
                    if option.get("help")
                    else {}
                ),
            }
            for option in question["options"]
        ],
    }


# ---------------------------------------------------------------------------
# Fehlende Modelleingaben
# ---------------------------------------------------------------------------


def flatten_missing_inputs(
    layout: ModularInputLayout,
) -> list[MissingInput]:
    """
    Liefert alle offenen Eingaben in stabiler Reihenfolge.

    Die Reihenfolge entspricht der Prüfung im Workflowknoten
    apply_modular_model_inputs_node.
    """

    return [
        *(
            item
            for group in layout["product_groups"]
            for item in [
                *group["forecast_inputs"],
                *group["fact_inputs"],
            ]
        ),
        *layout["global_inputs"],
        *(
            item
            for group in layout["constraint_groups"]
            for item in group["inputs"]
        ),
    ]


def describe_missing_input(
    missing_input: MissingInput,
) -> str:
    """Beschreibt eine offene Eingabe verständlich."""

    label = _field_label(missing_input["field"])
    source = missing_input["source"]

    if source in {"forecast", "product_fact"}:
        return t("{label} – Produkt „{product_id}“", label=label, product_id=missing_input['product_id'])

    if source == "constraint_parameter":
        component = CONSTRAINT_COMPONENTS.get(
            missing_input.get("component_id", ""),
        )

        if component is not None:
            return f"{label} – {component['name']}"

    return label


def _input_id(index: int) -> str:
    return f"i{index + 1}"


def resolve_input_submissions(
    values: list[ChatInputValue],
    missing_inputs: list[MissingInput],
) -> tuple[list[ModularInputSubmission], list[str]]:
    """
    Übersetzt Chatwerte in reguläre Formular-Einreichungen.

    Ungültige Werte werden nicht übernommen, sondern als
    verständliche Fehlermeldung zurückgegeben.
    """

    inputs_by_id = {
        _input_id(index): missing_input
        for index, missing_input in enumerate(missing_inputs)
    }
    submissions: list[ModularInputSubmission] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for item in values:
        input_id = item["input_id"]
        missing_input = inputs_by_id.get(input_id)

        if missing_input is None or input_id in seen_ids:
            continue

        seen_ids.add(input_id)
        raw_value = item["value"]

        # Zahlen in Textform ("12,50") tolerant annehmen.
        if (
            missing_input["field"]["type"]
            in {"number", "integer"}
            and isinstance(raw_value, str)
        ):
            try:
                raw_value = float(
                    raw_value.strip().replace(",", ".")
                )
            except ValueError:
                pass

        try:
            parsed_value = parse_input_value(
                missing_input["field"],
                raw_value,
            )
        except (TypeError, ValueError) as error:
            errors.append(
                f"{describe_missing_input(missing_input)}: {error}"
            )
            continue

        if parsed_value is None or parsed_value == []:
            continue

        submissions.append(
            {
                "missing_input": deepcopy(missing_input),
                "value": parsed_value,
            }
        )

    return submissions, errors


# ---------------------------------------------------------------------------
# Geschäftsfragen
# ---------------------------------------------------------------------------


def get_open_business_questions(
    graph_state: dict[str, Any],
) -> list[BusinessQuestion]:
    intake_state = graph_state.get("business_intake")

    if not intake_state or intake_state.get("profile") is None:
        return []

    return build_business_intake_view_data(
        intake_state
    )["questions"]


def resolve_question_answers(
    answers: list[ChatQuestionAnswer],
    questions: list[BusinessQuestion],
) -> list[BusinessQuestionAnswer]:
    """Übernimmt nur Antworten auf aktuell offene Fragen."""

    options_by_question = {
        question["question_id"]: {
            option["option_id"]
            for option in question["options"]
        }
        for question in questions
    }
    resolved: list[BusinessQuestionAnswer] = []
    seen_questions: set[str] = set()

    for answer in answers:
        question_id = answer["question_id"]
        option_id = answer["option_id"]

        if (
            question_id in seen_questions
            or option_id
            not in options_by_question.get(question_id, set())
        ):
            continue

        seen_questions.add(question_id)
        resolved.append(
            {
                "question_id": question_id,
                "option_id": option_id,
                "source_id": CHAT_SOURCE_ID,
            }
        )

    return resolved


def append_to_description(
    description: str,
    message: str,
) -> str:
    """Ergänzt die Problembeschreibung um eine Chatnachricht."""

    message = message.strip()

    if not message:
        return description

    if not description.strip():
        return message

    return f"{description.rstrip()}\n{message}"


# ---------------------------------------------------------------------------
# Produkttabelle
# ---------------------------------------------------------------------------


def get_product_table_schema(
    graph_state: dict[str, Any],
) -> DynamicProductTableSchema | None:
    modular_state = graph_state.get("modular_model", {})
    core_id = modular_state.get("core_id")
    proposal = modular_state.get("preliminary_proposal")

    if not core_id or proposal is None:
        return None

    return build_dynamic_product_table_schema(
        core_id,
        proposal,
    )


def _schema_fields(
    schema: DynamicProductTableSchema,
) -> list[dict[str, Any]]:
    return [
        *schema["forecast_fields"],
        *schema["fact_fields"],
    ]


def _format_table_value(
    field: dict[str, Any],
    value: Any,
) -> Any:
    """Bringt einen Wert in das Format der Tabellenzelle."""

    if field["type"] == "forecast_quantiles" and isinstance(
        value,
        list,
    ):
        return _format_existing_forecast_quantiles(value)

    if field["type"] == "number_list" and isinstance(
        value,
        list,
    ):
        return _format_existing_number_list(value)

    if field["type"] in {"number", "integer"} and isinstance(
        value,
        str,
    ):
        try:
            return float(value.strip().replace(",", "."))
        except ValueError:
            return None

    return value


def merge_product_rows(
    existing_rows: list[dict[str, Any]],
    updates: list[ChatProductRow],
    schema: DynamicProductTableSchema,
) -> list[dict[str, Any]]:
    """
    Ergänzt Tabellenzeilen um neue Produktangaben aus dem Chat.

    Produkte werden anhand ihrer Kennung (ohne Groß- und
    Kleinschreibung) zusammengeführt.
    """

    fields = _schema_fields(schema)
    fields_by_name = {
        field["name"]: field
        for field in fields
    }
    rows = [dict(row) for row in existing_rows]

    for update in updates:
        product_id = update["product_id"].strip()

        if not product_id:
            continue

        row = next(
            (
                candidate
                for candidate in rows
                if str(candidate.get("product_id", ""))
                .strip()
                .casefold()
                == product_id.casefold()
            ),
            None,
        )

        if row is None:
            row = {
                "product_id": product_id,
                **{field["name"]: None for field in fields},
            }
            rows.append(row)

        for field_name, value in update["values"].items():
            field = fields_by_name.get(field_name)

            if field is None or value in (None, ""):
                continue

            row[field_name] = _format_table_value(field, value)

    return rows


def check_product_rows(
    rows: list[dict[str, Any]],
    schema: DynamicProductTableSchema,
):
    """Prüft Tabellenzeilen mit derselben Logik wie das Formular."""

    return parse_dynamic_product_table(rows, schema)


# ---------------------------------------------------------------------------
# Nächste Frage: Der Assistent fragt Schritt für Schritt nach
# ---------------------------------------------------------------------------


SUPPORTED_CASES_TEXT = (
    "Berechnen kann ich derzeit diese drei Arten von "
    "Bestellentscheidungen für **ein Produkt**:\n"
    "- **Einmalige Bestellung vor einer Saison** bei unsicherer "
    "Nachfrage, ohne Nachbestellung (z. B. Weihnachtsbäume, "
    "Saisonware, Tageszeitungen)\n"
    "- **Regelmäßige Nachbestellung bei gleichmäßigem, bekanntem "
    "Bedarf** – wie viel pro Bestellung (z. B. Büromaterial, "
    "Verbrauchsteile)\n"
    "- **Regelmäßige Nachbestellung bei schwankender Nachfrage** "
    "mit laufender Bestandsüberwachung – wie viel und ab welchem "
    "Bestand nachbestellt wird"
)

UNSUPPORTED_MESSAGE = (
    "**Diese Situation kann ich leider noch nicht berechnen.**\n\n"
    "{reason}"
    + SUPPORTED_CASES_TEXT
    + "\n\nWenn Ihre Situation doch zu einem dieser Fälle passt, "
    "beschreiben Sie sie gern noch einmal mit diesem Schwerpunkt. "
    "Sonst können Sie über **Neue Optimierung starten** neu "
    "beginnen."
)

MISSING_DECISIVE_INFO_MESSAGE = (
    "**Ohne diese Angabe kann ich kein passendes Rechenmodell "
    "auswählen** – raten möchte ich hier nicht, weil die Empfehlung "
    "sonst falsch sein könnte.\n\n"
    "Sie können die fehlende Information später ergänzen: Schreiben "
    "Sie mir einfach, sobald Sie sie kennen. "
    + SUPPORTED_CASES_TEXT
)


def _inputs_complete_text(specification: dict[str, Any]) -> str:
    """Hinweis auf zusätzliche Bausteine, sobald alles ausgefüllt ist."""

    inactive = [
        t(component["name"])
        for component in list_optional_components(specification)
        if not component["locked"] and not component["active"]
    ]
    text = t("**Alle Pflichtangaben sind vollständig.** ")

    if inactive:
        text += (
            t("Falls bei Ihnen weitere Vorgaben gelten, kann ich diese noch berücksichtigen:\n")
            + "\n".join(f"- {name}" for name in inactive)
            + t("\n\nSagen Sie mir einfach, was zutrifft (z. B. „höchstens 250 Stück“), oder wählen Sie es in Formular 3 unter **Zusätzliche Bausteine** aus. ")
        )

    return text + t("**Sonst geht es weiter zur Überprüfung.**")


class NextQuestion(TypedDict):
    """Die eine Angabe, nach der der Assistent als Nächstes fragt."""

    key: str
    text: str
    target: dict[str, Any]
    remaining: int


def _format_help(help_text: str | None) -> str:
    return f"\n\n_{help_text}_" if help_text else ""


def _business_question_text(question: BusinessQuestion) -> str:
    options = [
        option
        for option in question["options"]
        if option["option_id"] != "user_unknown"
    ]
    option_lines = "\n".join(
        f"- {t(option['label'])}"
        + (f" – {t(option['help'])}" if option.get("help") else "")
        for option in options
    )

    return (
        t("**{prompt}**\n\n{option_lines}\n\nAntworten Sie gern in eigenen Worten – „weiß ich nicht“ ist auch in Ordnung.", prompt=t(question['prompt']), option_lines=option_lines)
    )


def _field_question_text(
    field: dict[str, Any],
    product_id: str | None,
    *,
    mention_product: bool,
) -> str:
    product_part = (
        t(" für „{product_id}“", product_id=product_id)
        if mention_product and product_id
        else ""
    )

    return (
        t("Wie hoch ist Ihr Wert für **{value1}**{product_part}?", value1=_field_label(field), product_part=product_part)
        + _format_help(field.get("help"))
    )


def _next_product_table_question(
    graph_state: dict[str, Any],
    product_rows: list[dict[str, Any]],
) -> NextQuestion | None:
    schema = get_product_table_schema(graph_state)

    if schema is None:
        return None

    if not product_rows:
        return {
            "key": "product_table:names",
            "text": (
                t("**Für welche Produkte möchten Sie eine Empfehlung?** Nennen Sie mir einfach die Bezeichnungen oder Artikelnummern.")
            ),
            "target": {"kind": "product_names"},
            "remaining": 1,
        }

    open_cells = [
        (row, field)
        for row in product_rows
        for field in _schema_fields(schema)
        if field["required"]
        and row.get(field["name"]) in (None, "")
    ]

    if open_cells:
        row, field = open_cells[0]
        product_id = str(row.get("product_id", ""))

        return {
            "key": f"product_table:{product_id}:{field['name']}",
            "text": _field_question_text(
                field,
                product_id,
                mention_product=True,
            ),
            "target": {
                "kind": "product_value",
                "product_id": product_id,
                "field": field["name"],
            },
            "remaining": len(open_cells),
        }

    return {
        "key": f"product_table:more:{len(product_rows)}",
        "text": (
            t("Alle Angaben zu ")
            + ", ".join(
                f"„{row['product_id']}“" for row in product_rows
            )
            + t(" sind da. **Kommen noch weitere Produkte dazu, oder sind das alle?**")
        ),
        "target": {"kind": "more_products"},
        "remaining": 0,
    }


def build_next_question(
    step: str,
    graph_state: dict[str, Any] | None,
    *,
    product_rows: list[dict[str, Any]] | None = None,
) -> NextQuestion | None:
    """
    Bestimmt die eine Angabe, die als Nächstes erfragt wird.

    None bedeutet: In diesem Schritt gibt es nichts zu fragen.
    """

    graph_state = graph_state or {}
    modular_state = graph_state.get("modular_model") or {}

    if step == "business_intake":
        intake_state = graph_state.get("business_intake") or {}

        if intake_state.get("analyzed_raw_input"):
            view = build_business_intake_view_data(intake_state)

            if view["message_type"] != "error":
                return None

            core_selection = intake_state.get("core_selection") or {}

            if (
                not intake_state.get("analysis_error")
                and core_selection.get("status") == "unsupported"
            ):
                # Keine weitere Rückfrage: klar sagen, was nicht geht.
                reason = core_selection.get("reason") or ""
                return {
                    "key": "business_intake:unsupported",
                    "text": t(UNSUPPORTED_MESSAGE,
                        reason=f"{reason}\n\n" if reason else ""
                    ),
                    "target": {"kind": "none"},
                    "remaining": 0,
                }

            return {
                "key": f"business_intake:retry:{view['message']}",
                "text": (
                    t("{message}\n\n**Können Sie Ihre Situation etwas genauer beschreiben?**", message=view['message'])
                ),
                "target": {"kind": "description"},
                "remaining": 1,
            }

        return {
            "key": "business_intake:description",
            "text": (
                t("**Worum geht es bei Ihnen?** Beschreiben Sie kurz, welches Produkt Sie bestellen, wie oft und was Ihnen dabei wichtig ist.")
            ),
            "target": {"kind": "description"},
            "remaining": 1,
        }

    if step == "business_clarification":
        questions = get_open_business_questions(graph_state)

        if not questions:
            intake_state = graph_state.get("business_intake") or {}

            if (
                intake_state.get("profile") is not None
                and build_business_intake_view_data(intake_state)[
                    "message_type"
                ] == "warning"
            ):
                return {
                    "key": "business_clarification:missing_decisive_info",
                    "text": t(MISSING_DECISIVE_INFO_MESSAGE),
                    "target": {"kind": "none"},
                    "remaining": 0,
                }

            return None

        question = questions[0]

        return {
            "key": f"business:{question['question_id']}",
            "text": _business_question_text(question),
            "target": {
                "kind": "business_question",
                "question_id": question["question_id"],
            },
            "remaining": len(questions),
        }

    if step == "product_table":
        return _next_product_table_question(
            graph_state,
            product_rows or [],
        )

    if step == "model_input_form":
        specification = modular_state.get("specification")

        if specification is None:
            return None

        layout = build_modular_input_layout(specification)
        missing_inputs = flatten_missing_inputs(layout)

        if not missing_inputs:
            if layout["validation_errors"]:
                return None

            return {
                "key": "model_input:complete",
                "text": _inputs_complete_text(specification),
                "target": {"kind": "inputs_complete"},
                "remaining": 0,
            }

        # Pflichtfelder zuerst fragen.
        index, missing_input = min(
            enumerate(missing_inputs),
            key=lambda item: (
                not item[1]["field"]["required"],
                item[0],
            ),
        )
        mention_product = (
            len(specification.get("products", [])) > 1
        )

        return {
            "key": (
                "model_input:"
                + describe_missing_input(missing_input)
            ),
            "text": _field_question_text(
                missing_input["field"],
                missing_input.get("product_id"),
                mention_product=mention_product,
            ),
            "target": {
                "kind": "model_input",
                "input_id": _input_id(index),
            },
            "remaining": len(missing_inputs),
        }

    return None


# ---------------------------------------------------------------------------
# Kontext für das Sprachmodell
# ---------------------------------------------------------------------------

PROFILE_FIELD_IDS = (
    "recommendation",
    "procurement_pattern",
    "replenishment_during_period",
    "inventory_review",
    "product_scope",
    "primary_goal",
)

MAX_CONTEXT_PRODUCTS = 50


def _constraint_summaries(
    specification: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "name": t(CONSTRAINT_COMPONENTS.get(
                constraint["component_id"],
                {"name": constraint["component_id"]},
            )["name"]),
            "parameters": constraint.get("parameters", {}),
            "product_id": constraint.get("product_id"),
        }
        for constraint in specification.get("constraints", [])
    ]


def _known_parameters(
    graph_state: dict[str, Any],
) -> dict[str, Any]:
    """Alle bereits erfassten Angaben, unabhängig vom Schritt."""

    intake_state = graph_state.get("business_intake") or {}
    profile = intake_state.get("profile") or {}
    modular_state = graph_state.get("modular_model") or {}
    specification = modular_state.get("specification") or {}

    situation = {
        field_id: profile[field_id]["value"]
        for field_id in PROFILE_FIELD_IDS
        if field_id in profile
        and profile[field_id].get("status")
        in {"inferred", "confirmed"}
    }
    products = (
        specification.get("products")
        or modular_state.get("products")
        or []
    )

    return {
        "situation": situation,
        "model": _core_name(
            specification.get("core_id")
            or modular_state.get("core_id")
        ),
        "products": products[:MAX_CONTEXT_PRODUCTS],
        "product_count": len(products),
        "global_facts": (
            specification.get("global_facts")
            or modular_state.get("global_facts")
            or {}
        ),
        "constraints": _constraint_summaries(specification),
        "excluded_requirements": specification.get(
            "excluded_requirements",
            [],
        ),
    }


def _result_summary(
    graph_state: dict[str, Any],
) -> dict[str, Any]:
    modular_state = graph_state.get("modular_model", {})
    result = modular_state.get("optimization_result") or {}

    return {
        "status": t(STATUS_LABELS.get(
            result.get("status", ""),
            result.get("status"),
        )),
        "objective_value": result.get("objective_value"),
        "objective_label": build_objective_label(
            modular_state.get("specification") or {}
        ),
        "decisions": result.get("decisions", []),
        "metrics": [
            {
                **metric,
                "label": t(METRIC_LABELS.get(
                    metric.get("metric_id", ""),
                    metric.get("metric_id"),
                )),
            }
            for metric in result.get("metrics", [])
        ],
        "errors": result.get("errors", []),
        "explanation": _truncate(
            modular_state.get("result_explanation") or ""
        ),
    }


def _build_step_context(
    step: str,
    graph_state: dict[str, Any],
    *,
    raw_description: str,
    product_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    modular_state = graph_state.get("modular_model", {})

    if step in {"business_intake", "business_clarification"}:
        return {
            "step_description": (
                t("Der Nutzer beschreibt seine betriebliche Bestellsituation. Übernimm Situationsangaben und beantwortete Rückfragen.")
            ),
            "current_description": _truncate(raw_description),
            "open_questions": [
                _describe_question_for_llm(question)
                for question in get_open_business_questions(
                    graph_state
                )
            ],
            "allowed_outputs": [
                "use_message_as_description",
                "question_answers",
            ],
            "allowed_actions": (
                ["none", "edit_description", "restart"]
                if step == "business_clarification"
                else ["none"]
            ),
        }

    if step == "product_table":
        schema = get_product_table_schema(graph_state)

        return {
            "step_description": (
                t("Der Nutzer nennt seine Produkte und deren Daten. Übernimm jedes Produkt mit seiner Kennung und den genannten Werten. Setze submit_products, wenn der Nutzer bestätigt, dass keine weiteren Produkte dazukommen.")
            ),
            "product_fields": [
                _describe_field_for_llm(field)
                for field in (
                    _schema_fields(schema) if schema else []
                )
            ],
            "collected_products": product_rows,
            "allowed_outputs": ["products"],
            "allowed_actions": [
                "none",
                "submit_products",
                "restart",
            ],
        }

    if step == "model_input_form":
        specification = modular_state.get("specification")
        missing_inputs: list[MissingInput] = []
        validation_errors: list[str] = []

        optional_components: list[dict[str, Any]] = []

        if specification is not None:
            layout = build_modular_input_layout(specification)
            missing_inputs = flatten_missing_inputs(layout)
            validation_errors = layout["validation_errors"]
            optional_components = [
                {
                    "component_id": component["component_id"],
                    "name": component["name"],
                    "active": component["active"],
                }
                for component in list_optional_components(specification)
                if not component["locked"]
            ]

        return {
            "step_description": (
                t("Für die Berechnung fehlen noch Zahlenwerte. Übernimm genannte Werte in input_values. Sind alle Pflichtangaben vorhanden, kann der Nutzer zusätzliche Bausteine ergänzen (add_components) oder mit continue_to_review zur Überprüfung weitergehen.")
            ),
            "optional_components": optional_components,
            "missing_inputs": [
                {
                    "input_id": _input_id(index),
                    "description": describe_missing_input(
                        missing_input
                    ),
                    "product_id": missing_input.get("product_id"),
                    **_describe_field_for_llm(
                        missing_input["field"]
                    ),
                }
                for index, missing_input in enumerate(
                    missing_inputs
                )
            ],
            "validation_errors": validation_errors,
            "allowed_outputs": ["input_values", "add_components"],
            "allowed_actions": (
                ["none", "restart", "continue_to_review"]
                if specification is not None and not missing_inputs
                and not validation_errors
                else ["none", "restart"]
            ),
        }

    if step == "model_unsupported":
        proposal = modular_state.get("proposal") or {}
        selection_status = proposal.get("selection_status")

        return {
            "step_description": (
                t("Einige Anforderungen lassen sich mit den verfügbaren Bausteinen nicht abbilden. Der Nutzer kann ohne sie fortfahren (confirm_exclusions, nur nach ausdrücklicher Zustimmung) oder die Beschreibung ändern (edit_description).")
            ),
            "selection_status": selection_status,
            "unresolved_requirements": proposal.get(
                "unresolved_requirements",
                [],
            ),
            "allowed_outputs": [],
            "allowed_actions": [
                "none",
                "edit_description",
                "restart",
                *(
                    ["confirm_exclusions"]
                    if selection_status == "unsupported"
                    else []
                ),
            ],
        }

    if step == "review":
        pending_assumptions = modular_state.get(
            "pending_core_assumptions", []
        )
        return {
            "step_description": (
                t("Alle Angaben liegen vor. Der Nutzer prüft die Berechnungsgrundlage (siehe known_parameters). Beantworte Fragen dazu. Bei offenen Grundmodell-Annahmen muss der Nutzer diese zunächst im Review einzeln per Checkbox bestätigen. Setze start_optimization nur nach ausdrücklichem Berechnungswunsch.")
            ),
            "pending_core_assumptions": pending_assumptions,
            "review_explanation": _truncate(
                modular_state.get("review_explanation") or ""
            ),
            "allowed_outputs": [],
            "allowed_actions": [
                "none",
                "start_optimization",
                "edit_description",
                "restart",
            ],
        }

    if step == "result":
        return {
            "step_description": (
                t("Die Berechnung ist abgeschlossen. Erkläre das Ergebnis verständlich und beantworte Fragen. explain_result erzeugt eine ausführliche Erklärung; restart beginnt eine neue Optimierung.")
            ),
            "result": _result_summary(graph_state),
            "allowed_outputs": [],
            "allowed_actions": [
                "none",
                "explain_result",
                "restart",
            ],
        }

    return {
        "allowed_outputs": [],
        "allowed_actions": ["none"],
    }


def build_chat_context(
    step: str,
    graph_state: dict[str, Any] | None,
    *,
    raw_description: str = "",
    product_rows: list[dict[str, Any]] | None = None,
    user_statements: str = "",
) -> dict[str, Any]:
    """Kontext für das LLM: Schritt, aktuelle Frage, bekannte Werte."""

    graph_state = graph_state or {}
    product_rows = product_rows or []
    next_question = build_next_question(
        step,
        graph_state,
        product_rows=product_rows,
    )

    return {
        "step": step,
        **_build_step_context(
            step,
            graph_state,
            raw_description=raw_description,
            product_rows=product_rows,
        ),
        "current_question": (
            {
                "text": next_question["text"],
                "target": next_question["target"],
                "remaining_open_items": next_question["remaining"],
            }
            if next_question
            else None
        ),
        "known_parameters": _known_parameters(graph_state),
        "user_statements": user_statements,
    }


# ---------------------------------------------------------------------------
# Festgefahrene Fragen
# ---------------------------------------------------------------------------

MAX_UNANSWERED_REPEATS = 2
"""Nach so vielen Antworten ohne Fortschritt bietet der Chat Auswege an."""


def build_stuck_message(next_question: NextQuestion | None) -> str | None:
    """Hilft weiter, wenn eine Frage mehrfach unbeantwortet bleibt."""

    if next_question is None:
        return None

    kind = next_question["target"].get("kind")

    if kind == "business_question":
        return (
            t("Wir kommen bei dieser Frage gerade nicht weiter – kein Problem. **Tippen Sie einfach auf eine der Antworten unter der Frage.** Wenn Sie es nicht wissen, wählen Sie „Noch nicht bekannt“. Ich sage Ihnen dann ehrlich, ob ich trotzdem rechnen kann.")
        )

    if kind == "model_input":
        return (
            t("Bei dieser Angabe hänge ich fest. So geht es weiter:\n- **Nennen Sie einfach die Zahl**, z. B. „120“ – die Einheit steht in der Frage.\n- **Sie kennen den Wert nicht?** Schreiben Sie „Wie schätze ich das?“ – ich erkläre Ihnen, woher Sie ihn bekommen.\n- Sie können den Wert auch direkt **im Formular rechts** eintragen.")
        )

    if kind == "description":
        return (
            t("Ich konnte Ihre Situation bisher keinem Rechenmodell zuordnen.\n\n")
            + t(SUPPORTED_CASES_TEXT)
            + t("\n\nWenn Ihre Situation zu keinem dieser Fälle passt, kann ich sie leider noch nicht berechnen. Über **Neue Optimierung starten** können Sie neu beginnen.")
        )

    if kind in {"product_names", "product_table", "more_products"}:
        return (
            t("Bei den Produktdaten hänge ich fest. Am einfachsten tragen Sie die Werte **direkt in die Tabelle rechts** ein – ich übernehme den Rest.")
        )

    return None


# ---------------------------------------------------------------------------
# Gedächtnis: bisherige Aussagen des Nutzers
# ---------------------------------------------------------------------------


def collect_user_statements(
    messages: list[dict[str, Any]],
    *,
    limit: int = MAX_USER_STATEMENTS_CHARS,
) -> str:
    """Fasst alle bisherigen Nutzernachrichten ungekürzt zusammen.

    Bei sehr langen Gesprächen bleiben die neuesten Aussagen erhalten.
    """

    statements = "\n\n".join(
        message["content"].strip()
        for message in messages
        if message.get("role") == "user" and message.get("content", "").strip()
    )

    if len(statements) <= limit:
        return statements

    return "…" + statements[-(limit - 1):]


class RecallItem(TypedDict):
    """Eine offene Angabe, die aus bisherigen Aussagen kommen kann."""

    id: str
    key: str
    kind: Literal["choice", "value"]
    question: str
    options: NotRequired[list[dict[str, str]]]
    field: NotRequired[dict[str, Any]]


def build_recall_items(
    step: str,
    graph_state: dict[str, Any] | None,
) -> list[RecallItem]:
    """Offene Angaben des aktuellen Schritts für den Erinnerungsschritt.

    ``id`` ist die Kennung für das Sprachmodell (bei Modelleingaben
    positionsabhängig), ``key`` bleibt über Schritte hinweg stabil.
    """

    graph_state = graph_state or {}

    if step == "business_clarification":
        return [
            {
                "id": question["question_id"],
                "key": f"business:{question['question_id']}",
                "kind": "choice",
                "question": question["prompt"],
                "options": [
                    {"option_id": option["option_id"], "label": option["label"]}
                    for option in question["options"]
                    if option["option_id"] != "user_unknown"
                ],
            }
            for question in get_open_business_questions(graph_state)
        ]

    if step == "model_input_form":
        specification = (graph_state.get("modular_model") or {}).get(
            "specification"
        )

        if specification is None:
            return []

        missing_inputs = flatten_missing_inputs(
            build_modular_input_layout(specification)
        )

        return [
            {
                "id": _input_id(index),
                "key": "model_input:" + describe_missing_input(missing_input),
                "kind": "value",
                "question": describe_missing_input(missing_input),
                "field": _describe_field_for_llm(missing_input["field"]),
            }
            for index, missing_input in enumerate(missing_inputs)
        ]

    return []


def _format_recalled_value(item: RecallItem, value: Any) -> str:
    if item["kind"] == "choice":
        for option in item.get("options", []):
            if option["option_id"] == value:
                return t(option["label"])

    if isinstance(value, list):
        return "; ".join(
            f"{entry.get('level')}: {entry.get('value')}"
            if isinstance(entry, dict)
            else str(entry)
            for entry in value
        )

    unit = (item.get("field") or {}).get("unit")
    return f"{value} {unit}" if unit else str(value)


def format_recall_note(
    items: list[RecallItem],
    recalled: list[dict[str, Any]],
) -> str | None:
    """Teilt dem Nutzer mit, was aus seinen Angaben übernommen wurde."""

    items_by_id = {item["id"]: item for item in items}
    lines = []

    for answer in recalled:
        item = items_by_id.get(answer["id"])

        if item is None:
            continue

        lines.append(
            f"- **{t(item['question'])}**: "
            f"{_format_recalled_value(item, answer['value'])} "
            f"_(„{answer['quote']}“)_"
        )

    if not lines:
        return None

    return (
        t("Das haben Sie mir schon gesagt – ich habe es übernommen, Sie müssen es nicht noch einmal angeben:\n") + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# Proaktive Nachrichten
# ---------------------------------------------------------------------------


def _remaining_hint(remaining: int) -> str:
    if remaining <= 1:
        return ""

    return (
        t("\n\n_Noch {remaining} Angaben offen – Sie können auch mehrere auf einmal nennen._", remaining=remaining)
    )


def build_step_intro(
    step: str,
    graph_state: dict[str, Any] | None,
    *,
    product_rows: list[dict[str, Any]] | None = None,
) -> str | None:
    """Kurze Einleitung, wenn ein neuer Schritt beginnt."""

    graph_state = graph_state or {}
    modular_state = graph_state.get("modular_model", {})

    if step == "business_intake":
        intake_state = graph_state.get("business_intake") or {}

        if not intake_state.get("analyzed_raw_input"):
            return t(GREETING)

        return (
            t("Sie können Ihre Beschreibung jederzeit ergänzen. Schreiben Sie mir einfach, was sich geändert hat.")
        )

    if step == "business_clarification":
        return (
            t("Danke, das habe ich ausgewertet. Ich habe noch ein paar kurze Rückfragen.")
        )

    if step == "product_table":
        return (
            t("Gut, jetzt geht es um Ihre Produkte. Ich frage die Daten der Reihe nach ab.")
        )

    if step == "model_input_form":
        return (
            t("Prima, jetzt brauche ich noch ein paar Zahlen für die Berechnung. Ich frage sie nacheinander ab.")
        )

    if step == "model_unsupported":
        proposal = modular_state.get("proposal") or {}
        lines = [
            t("Einige Ihrer Anforderungen kann ich mit den verfügbaren Bausteinen noch nicht abbilden:"),
            "",
            *(
                f"- **{requirement['description']}**: "
                f"{requirement['reason']}"
                for requirement in proposal.get(
                    "unresolved_requirements",
                    [],
                )
            ),
            "",
        ]

        if proposal.get("selection_status") == "unsupported":
            lines.append(
                t("**Möchten Sie ohne diese Anforderungen fortfahren, oder Ihre Beschreibung anpassen?**")
            )
        else:
            lines.append(
                t("**Können Sie genauer beschreiben, was gemeint ist?**")
            )

        return "\n".join(lines)

    if step == "review":
        return (
            t("Damit habe ich alles, was ich brauche. Rechts sehen Sie die Zusammenfassung – fragen Sie mich gern, wenn etwas unklar ist.\n\n**Stimmen die Angaben, und soll ich die Berechnung starten?**")
        )

    if step == "result":
        result = modular_state.get("optimization_result") or {}
        status = result.get("status", "")
        status_label = (
            t(STATUS_LABELS[status])
            if status in STATUS_LABELS
            else t("Die Berechnung ist abgeschlossen")
        )

        return (
            t("**{status_label}.** Das Ergebnis sehen Sie rechts. **Soll ich es Ihnen erklären?** Oder fragen Sie mich direkt zu einzelnen Zahlen.", status_label=status_label)
        )

    return None


def build_follow_up_messages(
    step: str,
    graph_state: dict[str, Any] | None,
    *,
    product_rows: list[dict[str, Any]] | None,
    previous_step: str | None,
    previous_question_key: str | None,
) -> tuple[list[str], str | None]:
    """
    Bestimmt, was der Assistent von sich aus sagen soll.

    Bei einem neuen Schritt: Einleitung plus erste Frage.
    Im selben Schritt: nur die nächste Frage, sobald sich die
    offene Angabe geändert hat. Liefert zusätzlich den Schlüssel
    der aktuellen Frage zurück.
    """

    next_question = build_next_question(
        step,
        graph_state,
        product_rows=product_rows,
    )
    question_key = next_question["key"] if next_question else None
    messages: list[str] = []

    if step != previous_step:
        if previous_step in {"business_intake", "business_clarification"}:
            summary = build_proposal_summary(
                (graph_state or {}).get("business_intake")
            )

            if summary:
                messages.append(summary)

        intro = build_step_intro(
            step,
            graph_state,
            product_rows=product_rows,
        )

        if intro:
            messages.append(intro)

        # Die Begrüßung enthält die Einstiegsfrage bereits.
        if next_question and intro != t(GREETING):
            messages.append(
                next_question["text"]
                + _remaining_hint(next_question["remaining"])
            )

    elif next_question and question_key != previous_question_key:
        messages.append(
            next_question["text"]
            + _remaining_hint(next_question["remaining"])
        )

    return messages, question_key


# ---------------------------------------------------------------------------
# Nutzerführung: Fortschritt, Antwort-Buttons, bekanntes Wissen
# ---------------------------------------------------------------------------

STEP_PROGRESS = {
    "business_intake": (1, "Situation beschreiben"),
    "business_clarification": (2, "Rückfragen"),
    "model_unsupported": (2, "Rückfragen"),
    "product_table": (3, "Zahlen für die Berechnung"),
    "model_input_form": (3, "Zahlen für die Berechnung"),
    "review": (4, "Angaben prüfen"),
    "result": (5, "Ergebnis"),
}
TOTAL_STEPS = 5

def build_progress_label(step: str) -> str | None:
    """Kurze Fortschrittsangabe für den Chat, z. B. „Schritt 2 von 5“."""

    progress = STEP_PROGRESS.get(step)

    if progress is None:
        return None

    number, label = progress
    return t("Schritt {number} von {TOTAL_STEPS} · {label}", number=number, TOTAL_STEPS=TOTAL_STEPS, label=t(label))


class QuickReply(TypedDict):
    """Ein Antwort-Button unter der aktuellen Frage."""

    key: str
    label: str
    kind: Literal["message", "answer", "action"]
    value: Any


def build_quick_replies(
    step: str,
    graph_state: dict[str, Any] | None,
    next_question: NextQuestion | None,
) -> list[QuickReply]:
    """Antwortvorschläge, damit der Nutzer nicht tippen muss."""

    graph_state = graph_state or {}
    target = (next_question or {}).get("target") or {}
    question_key = (next_question or {}).get("key", step)

    if target.get("kind") == "none":
        return [
            {
                "key": f"{question_key}:restart",
                "label": t("🔄 Neu beginnen"),
                "kind": "action",
                "value": "restart",
            }
        ]

    if target.get("kind") == "business_question":
        question_id = target["question_id"]

        for question in get_open_business_questions(graph_state):
            if question["question_id"] != question_id:
                continue

            return [
                {
                    "key": f"{question_key}:{option['option_id']}",
                    "label": t(option["label"]),
                    "kind": "answer",
                    "value": {
                        "question_id": question_id,
                        "option_id": option["option_id"],
                    },
                }
                for option in question["options"]
            ]

    if target.get("kind") == "inputs_complete":
        return [
            {
                "key": "model_input:continue",
                "label": t("➡️ Weiter zur Überprüfung"),
                "kind": "action",
                "value": "continue_to_review",
            }
        ]

    if target.get("kind") == "model_input":
        return [
            {
                "key": f"{question_key}:estimate",
                "label": t("🤔 Wie schätze ich das?"),
                "kind": "message",
                "value": t("Ich kenne den Wert nicht genau. Wie kann ich ihn schätzen?"),
            }
        ]

    if step == "review":
        return [
            {
                "key": "review:start",
                "label": t("✅ Berechnung starten"),
                "kind": "action",
                "value": "start_optimization",
            }
        ]

    if step == "result":
        return [
            {
                "key": "result:explain",
                "label": t("💡 Ergebnis erklären"),
                "kind": "action",
                "value": "explain_result",
            },
            {
                "key": "result:restart",
                "label": t("🔄 Neue Optimierung"),
                "kind": "action",
                "value": "restart",
            },
        ]

    return []


def _field_catalog(core_id: str | None) -> dict[str, dict[str, Any]]:
    """Bezeichnungen und Einheiten aller Eingabefelder eines Modells."""

    catalog: dict[str, dict[str, Any]] = {}
    registered = CORE_REGISTRY.get(core_id or "")

    if registered is not None:
        definition = registered["definition"]

        for field in [
            *definition["required_forecast_fields"],
            *definition["required_product_facts"],
            *definition["required_global_facts"],
        ]:
            catalog.setdefault(field["name"], field)

    for component in [
        *OBJECTIVE_COMPONENTS.values(),
        *CONSTRAINT_COMPONENTS.values(),
    ]:
        for field in [
            *component["required_product_facts"],
            *component["required_global_facts"],
        ]:
            catalog.setdefault(field["name"], field)

    return catalog


def _format_known_value(value: Any, field: dict[str, Any] | None) -> str:
    if isinstance(value, list):
        return "; ".join(
            f"{entry.get('level')}: {entry.get('value')}"
            if isinstance(entry, dict)
            else str(entry)
            for entry in value
        )

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    unit = t((field or {}).get("unit"))
    return f"{value} {unit}" if unit else str(value)


def build_known_facts_summary(
    graph_state: dict[str, Any] | None,
) -> list[str]:
    """Alles, was die App schon weiß – als lesbare Liste (Markdown)."""

    graph_state = graph_state or {}
    intake_state = graph_state.get("business_intake") or {}
    profile = intake_state.get("profile") or {}
    modular_state = graph_state.get("modular_model") or {}
    specification = modular_state.get("specification") or {}
    lines: list[str] = []

    for field_id in PROFILE_FIELD_IDS:
        field = profile.get(field_id) or {}

        if field.get("status") not in {"inferred", "confirmed"}:
            continue

        question = BUSINESS_QUESTIONS.get(field_id)
        label = field.get("value")

        if question is not None:
            label = next(
                (
                    t(option["label"])
                    for option in question["options"]
                    if option.get("profile_value", option["option_id"])
                    == field.get("value")
                ),
                label,
            )

        lines.append(f"- {label}")

    core_id = specification.get("core_id") or modular_state.get("core_id")
    core_name = _core_name(core_id)

    if core_name:
        lines.append(t("- **Rechenmodell:** {core_name}", core_name=core_name))

    catalog = _field_catalog(core_id)
    products = specification.get("products") or modular_state.get("products") or []

    for product in products[:MAX_CONTEXT_PRODUCTS]:
        values = {
            **(product.get("forecast") or {}),
            **(product.get("facts") or {}),
        }
        details = [
            f"{t((catalog.get(name) or {}).get('label', name))}: "
            f"{_format_known_value(value, catalog.get(name))}"
            for name, value in values.items()
            if value not in (None, "", [])
        ]
        lines.append(
            t("- **Produkt „{product_id}“**", product_id=product['product_id'])
            + (":\n" + "\n".join(f"  - {detail}" for detail in details)
               if details else "")
        )

    for constraint in _constraint_summaries(specification):
        parameters = ", ".join(
            _format_known_value(value, None)
            for value in constraint["parameters"].values()
        )
        lines.append(
            f"- **{constraint['name']}**" + (f": {parameters}" if parameters else "")
        )

    return lines


def build_proposal_summary(
    intake_state: dict[str, Any] | None,
) -> str | None:
    """Fasst erkannte, noch unbestätigte Anforderungen für den Chat zusammen."""

    from agentic_optimization.ui.requirements_panel import (
        build_requirement_items,
    )

    proposed = [
        item
        for item in build_requirement_items(intake_state)
        if item["status"] == "proposed"
    ]

    if not proposed:
        return None

    lines = []

    for item in proposed:
        if item["kind"] == "assumption":
            lines.append(t("- _Annahme:_ {title} ({value_label})", title=item['title'], value_label=item['value_label']))
        else:
            lines.append(f"- **{item['title']}:** {item['value_label']}")

    return (
        t("Das habe ich aus Ihren Angaben erkannt:\n")
        + "\n".join(lines)
        + t("\n\nBitte prüfen Sie die Punkte oben unter **Erkannte Anforderungen** bzw. **Bekannte Angaben** – Sie können sie bestätigen, ändern oder verwerfen.")
    )
