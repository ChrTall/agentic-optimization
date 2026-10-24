"""
UI-unabhängige Logik des begleitenden Chat-Assistenten.

Dieses Modul übersetzt den Workflowzustand in einen kompakten
Kontext für das Sprachmodell und die (bereits bereinigte)
Antwort des Sprachmodells zurück in die bestehenden
Eingabeformate des Workflows. Alle Werte laufen dabei durch
dieselben Prüfungen wie die Formulareingaben.
"""

from copy import deepcopy
from typing import Any, TypedDict

from agentic_optimization.agents.chat_assistant_agent import (
    ChatInputValue,
    ChatProductRow,
    ChatQuestionAnswer,
)
from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)
from agentic_optimization.models.validator import (
    MissingInput,
    validate_model_specification,
)
from agentic_optimization.models.product_names import clean_product_name, fallback_product_name, product_name
from agentic_optimization.state import (
    BusinessQuestionAnswer,
    ModularInputSubmission,
)
from agentic_optimization.ui.business_intake_view_builder import (
    build_business_intake_view_data,
)
from agentic_optimization.ui.business_problem_form import (
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
from agentic_optimization.ui.modular_components import component_options, editable_chat_inputs, input_address
from agentic_optimization.ui.modular_product_list import (
    parse_dynamic_product_table,
)
from agentic_optimization.ui.modular_result_builder import (
    METRIC_LABELS,
    STATUS_LABELS,
    objective_presentation,
)

CHAT_SOURCE_ID = "chat_assistant"

MAX_CONTEXT_TEXT = 4_000

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
    unit = field.get("unit")
    label = field["label"]

    return f"{label} ({unit})" if unit else label


def _core_name(core_id: str | None) -> str | None:
    if not core_id:
        return None

    registered_core = CORE_REGISTRY.get(core_id)

    if registered_core is None:
        return None

    return registered_core["definition"]["name"]


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
    specification: dict | None = None,
) -> str:
    """Beschreibt eine offene Eingabe verständlich."""

    label = _field_label(missing_input["field"])
    source = missing_input["source"]

    if source in {"forecast", "product_fact"}:
        name = product_name(specification or {}, missing_input["product_id"])
        return f"{label} – Produkt „{name}“"

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
        if clean_product_name(update.get("display_name")):
            row["display_name"] = clean_product_name(update["display_name"])

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
        f"- {option['label']}"
        + (f" – {option['help']}" if option.get("help") else "")
        for option in options
    )

    return (
        f"**{question['prompt']}**\n\n{option_lines}\n\n"
        "Antworten Sie gern in eigenen Worten – "
        "„weiß ich nicht“ ist auch in Ordnung."
    )


def _field_question_text(
    field: dict[str, Any],
    product_id: str | None,
    *,
    mention_product: bool,
) -> str:
    product_part = (
        f" für „{product_id}“"
        if mention_product and product_id
        else ""
    )

    return (
        f"Wie hoch ist Ihr Wert für **{_field_label(field)}**"
        f"{product_part}?"
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
                "**Für welche Produkte möchten Sie eine "
                "Empfehlung?** Nennen Sie mir einfach die "
                "Bezeichnungen oder Artikelnummern."
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
                clean_product_name(row.get("display_name")) or fallback_product_name(product_id),
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
            "Alle Angaben zu "
            + ", ".join(
                f"„{clean_product_name(row.get('display_name')) or fallback_product_name(row['product_id'])}“" for row in product_rows
            )
            + " sind da. **Kommen noch weitere Produkte "
            "dazu, oder sind das alle?**"
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

            return {
                "key": f"business_intake:retry:{view['message']}",
                "text": (
                    f"{view['message']}\n\n"
                    "**Können Sie Ihre Situation etwas genauer "
                    "beschreiben?**"
                ),
                "target": {"kind": "description"},
                "remaining": 1,
            }

        return {
            "key": "business_intake:description",
            "text": (
                "**Worum geht es bei Ihnen?** Beschreiben Sie "
                "kurz, welches Produkt Sie bestellen, wie oft "
                "und was Ihnen dabei wichtig ist."
            ),
            "target": {"kind": "description"},
            "remaining": 1,
        }

    if step == "business_clarification":
        questions = get_open_business_questions(graph_state)

        if not questions:
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

        missing_inputs = validate_model_specification(specification)["missing_inputs"]

        if not missing_inputs:
            return None

        # Pflichtfelder zuerst fragen.
        index, missing_input = min(
            enumerate(missing_inputs),
            key=lambda item: (
                not item[1]["field"]["required"],
                item[0],
            ),
        )
        index = next(i for i, target in enumerate(editable_chat_inputs(specification))
                     if target["option_id"] is None
                     and input_address(specification, target["input"])
                     == input_address(specification, missing_input))
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
                product_name(specification, missing_input.get("product_id")),
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
            "name": CONSTRAINT_COMPONENTS.get(
                constraint["component_id"],
                {"name": constraint["component_id"]},
            )["name"],
            "parameters": constraint.get("parameters", {}),
            "product_id": constraint.get("product_id"),
            "product_name": product_name(specification, constraint.get("product_id")),
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
        "products": [{**p, "display_name": product_name({"products": products}, p["product_id"])}
                     for p in products[:MAX_CONTEXT_PRODUCTS]],
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
    specification = modular_state.get("specification") or {}

    return {
        "status": STATUS_LABELS.get(
            result.get("status", ""),
            result.get("status"),
        ),
        "objective_value": result.get("objective_value"),
        "objective_label": objective_presentation(specification)[0] if specification.get("objective") else None,
        "decisions": [{**d, "product_name": product_name(specification, d.get("product_id"))}
                      for d in result.get("decisions", [])],
        "metrics": [
            {
                **metric,
                "product_name": product_name(specification, metric.get("product_id")),
                "label": METRIC_LABELS.get(
                    metric.get("metric_id", ""),
                    metric.get("metric_id"),
                ),
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
                "Der Nutzer beschreibt seine betriebliche "
                "Bestellsituation. Übernimm Situationsangaben "
                "und beantwortete Rückfragen."
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
                "Der Nutzer nennt seine Produkte und deren "
                "Daten. Übernimm jedes Produkt mit seiner "
                "Kennung und den genannten Werten. Setze "
                "submit_products, wenn der Nutzer bestätigt, "
                "dass keine weiteren Produkte dazukommen."
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
        targets = []

        if specification is not None:
            layout = build_modular_input_layout(specification)
            targets = editable_chat_inputs(specification)
            missing_inputs = [target["input"] for target in targets]
            validation_errors = layout["validation_errors"]

        return {
            "step_description": (
                "Übernimm oder korrigiere genannte Werte in input_values. "
                "Felder mit activates_component aktivieren einen zusätzlichen "
                "kompatiblen Baustein: nur bei ausdrücklichem Nutzerwunsch. "
                "Bei mehreren Produkten und unklarer Zuordnung nachfragen. "
                "Die Felder bleiben sichtbar. Der Nutzer prüft die zusätzlichen "
                "Möglichkeiten und geht anschließend im Formular weiter."
            ),
            "missing_inputs": [
                {
                    "input_id": _input_id(index),
                    "description": describe_missing_input(
                        missing_input, specification
                    ),
                    "product_id": missing_input.get("product_id"),
                    "product_name": product_name(specification or {}, missing_input.get("product_id")),
                    "activates_component": targets[index]["option_id"],
                    **_describe_field_for_llm(
                        missing_input["field"]
                    ),
                }
                for index, missing_input in enumerate(
                    missing_inputs
                )
            ],
            "validation_errors": validation_errors,
            "allowed_outputs": ["input_values"],
            "allowed_actions": ["none", "restart"],
        }

    if step == "model_unsupported":
        proposal = modular_state.get("proposal") or {}
        selection_status = proposal.get("selection_status")

        return {
            "step_description": (
                "Einige Anforderungen lassen sich mit den "
                "verfügbaren Bausteinen nicht abbilden. Der "
                "Nutzer kann ohne sie fortfahren "
                "(confirm_exclusions, nur nach ausdrücklicher "
                "Zustimmung) oder die Beschreibung ändern "
                "(edit_description)."
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
                "Alle Angaben liegen vor. Der Nutzer prüft die "
                "Berechnungsgrundlage (siehe known_parameters). "
                "Beantworte Fragen dazu. Bei offenen "
                "Grundmodell-Annahmen muss der Nutzer diese "
                "zunächst im Review einzeln per Checkbox "
                "bestätigen. Setze start_optimization nur nach "
                "ausdrücklichem Berechnungswunsch."
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
                "Die Berechnung ist abgeschlossen. Erkläre das "
                "Ergebnis verständlich und beantworte Fragen. "
                "explain_result erzeugt eine ausführliche "
                "Erklärung; restart beginnt eine neue "
                "Optimierung."
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
    }


# ---------------------------------------------------------------------------
# Proaktive Nachrichten
# ---------------------------------------------------------------------------


def _remaining_hint(remaining: int) -> str:
    if remaining <= 1:
        return ""

    return (
        f"\n\n_Noch {remaining} Angaben offen – Sie können "
        "auch mehrere auf einmal nennen._"
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
            return GREETING

        return (
            "Sie können Ihre Beschreibung jederzeit ergänzen. "
            "Schreiben Sie mir einfach, was sich geändert hat."
        )

    if step == "business_clarification":
        return (
            "Danke, das habe ich ausgewertet. Ich habe noch ein "
            "paar kurze Rückfragen."
        )

    if step == "product_table":
        return (
            "Gut, jetzt geht es um Ihre Produkte. Ich frage die "
            "Daten der Reihe nach ab."
        )

    if step == "model_input_form":
        return (
            "Prima, jetzt brauche ich noch ein paar Zahlen für "
            "die Berechnung. Ich frage sie nacheinander ab."
        )

    if step == "model_unsupported":
        proposal = modular_state.get("proposal") or {}
        lines = [
            "Einige Ihrer Anforderungen kann ich mit den "
            "verfügbaren Bausteinen noch nicht abbilden:",
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
                "**Möchten Sie ohne diese Anforderungen "
                "fortfahren, oder Ihre Beschreibung anpassen?**"
            )
        else:
            lines.append(
                "**Können Sie genauer beschreiben, was gemeint "
                "ist?**"
            )

        return "\n".join(lines)

    if step == "review":
        return (
            "Damit habe ich alles, was ich brauche. Rechts sehen "
            "Sie die Zusammenfassung – fragen Sie mich gern, "
            "wenn etwas unklar ist.\n\n"
            "**Stimmen die Angaben, und soll ich die Berechnung "
            "starten?**"
        )

    if step == "result":
        result = modular_state.get("optimization_result") or {}
        status_label = STATUS_LABELS.get(
            result.get("status", ""),
            "Die Berechnung ist abgeschlossen",
        )

        return (
            f"**{status_label}.** Das Ergebnis sehen Sie "
            "rechts. **Soll ich es Ihnen erklären?** Oder "
            "fragen Sie mich direkt zu einzelnen Zahlen."
        )

    return None


def build_follow_up_messages(
    step: str,
    graph_state: dict[str, Any] | None,
    *,
    product_rows: list[dict[str, Any]] | None,
    previous_step: str | None,
    previous_question_key: str | None,
    additional_inputs_confirmed: bool = False,
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
        intro = build_step_intro(
            step,
            graph_state,
            product_rows=product_rows,
        )

        if intro:
            messages.append(intro)

        # Die Begrüßung enthält die Einstiegsfrage bereits.
        if next_question and intro != GREETING:
            messages.append(
                next_question["text"]
                + _remaining_hint(next_question["remaining"])
            )

    elif next_question and question_key != previous_question_key:
        messages.append(
            next_question["text"]
            + _remaining_hint(next_question["remaining"])
        )

    if step == "model_input_form" and not next_question:
        specification = (graph_state or {}).get("modular_model", {}).get("specification")
        errors = validate_model_specification(specification)["errors"] if specification else []
        if errors:
            question_key = "model_input:validation:" + "|".join(errors)
            if question_key != previous_question_key:
                messages.append("Die Eingabeprüfung meldet noch Unstimmigkeiten. "
                                "Bitte korrigieren Sie die Angaben im Formular:\n"
                                + "\n".join(f"- {error}" for error in errors))
        else:
            question_key = "model_input:complete:" + str(additional_inputs_confirmed)
            if question_key == previous_question_key:
                return messages, question_key
            options = component_options(specification) if specification else []
            examples = list(dict.fromkeys(o["label"].split(" – ")[0] for o in options))[:3]
            additions = (" Möchten Sie weitere Angaben berücksichtigen, beispielsweise "
                         + ", ".join(examples) + "?" if examples and not additional_inputs_confirmed else "")
            next_step = (
                " Prüfen Sie den Bereich ‚Weitere Angaben berücksichtigen‘ und "
                "bestätigen Sie, dass alle relevanten Angaben berücksichtigt sind."
                if not additional_inputs_confirmed else " Sie können zur Überprüfung weitergehen."
            )
            messages.append(
                "Die Pflichtfelder sind ausgefüllt." + additions + next_step
                + " Prüfen Sie die Werte "
                "anschließend in der Überprüfung."
            )

    return messages, question_key
