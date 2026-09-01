import json
import re
from collections.abc import Callable
from typing import get_args

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_optimization.state import (
    WORKFLOW_GENERAL_REQUIREMENTS_EXTRACTED,
    DecisionType,
    DemandType,
    GraphState,
    ObjectiveType,
    PlanningHorizon,
    ProductScope,
    ReviewType,
    RequirementConflict,
    UnsupportedRequirement,
    with_control,
)

ALLOWED_DECISION_TYPES = set(get_args(DecisionType))
ALLOWED_OBJECTIVE_TYPES = set(get_args(ObjectiveType))
ALLOWED_DEMAND_TYPES = set(get_args(DemandType))
ALLOWED_PRODUCT_SCOPES = set(get_args(ProductScope))
ALLOWED_PLANNING_HORIZONS = set(get_args(PlanningHorizon))
ALLOWED_REVIEW_TYPES = set(get_args(ReviewType))

REQUIRED_FIELDS = ["decision", "objective", "demand_type", "planning_horizon"]
STRUCTURAL_FIELDS = [*REQUIRED_FIELDS, "product_scope", "review_type"]

SYSTEM_PROMPT = """Du analysierst ausschließlich die freie Beschreibung
eines Optimierungsproblems.

Gib deine Antwort ausschließlich als JSON-Objekt zurück. Verwende keine
Markdown-Backticks und keine zusätzlichen Erklärungen.

Format:
{
  "decision": "order_quantity | production_quantity | inventory_level | null",
  "objective": "maximize_profit | minimize_cost | reach_service_level | null",
  "demand_type": "known | uncertain | historical_estimate | null",
  "product_scope": "single_product | multiple_products | null",
  "planning_horizon": "single_period | multiple_periods | null",
  "review_type": "continuous_review | periodic_review | null",
  "constraints": ["...", "..."],
  "unsupported_requirements": [
    {
      "field": "decision",
      "user_value": "Bestellzeitpunkt optimieren",
      "message": "Die Optimierung des Bestellzeitpunkts wird derzeit nicht unterstützt."
    }
  ]
}

Bedeutung:

decision:
- order_quantity: Bestellmenge oder Einkaufsmenge festlegen
- production_quantity: Produktionsmenge festlegen
- inventory_level: Lagerbestand festlegen

objective:
- maximize_profit: Gewinn maximieren
- minimize_cost: Kosten minimieren
- reach_service_level: ein Servicelevel erreichen

demand_type:
- known: sichere oder bekannte Nachfrage
- uncertain: unsichere oder saisonale Nachfrage
- historical_estimate: aus historischen Daten geschätzte Nachfrage

product_scope:
- single_product: genau ein Produkt; dies ist die feste Systemannahme
- multiple_products: mehrere Produkte; diese Problemstruktur wird derzeit
  nicht unterstützt

planning_horizon:
- single_period: eine einmalige Entscheidung
- multiple_periods: mehrere aufeinanderfolgende Zeiträume

review_type:
- continuous_review: der Bestand wird laufend überwacht und bei Erreichen
  eines Meldebestands wird bestellt
- periodic_review: der Bestand wird in festen Zeitabständen überprüft

review_type ist nur relevant, wenn planning_horizon multiple_periods ist.
Bei single_period muss review_type null sein.


constraints enthält ausschließlich konkrete betriebliche Vorgaben oder
Grenzen, welche die zulässige Optimierungsentscheidung einschränken.

Beispiele für Constraints:
- ein maximales Budget
- begrenzter Lagerplatz oder begrenzte Kapazität
- eine Mindestmenge
- eine Höchstmenge
- eine Lieferfrist

Keine Constraints sind:
- unsichere Nachfrage
- saisonale Nachfrage
- historische Nachfrage
- die Anzahl der Produkte
- der Planungshorizont
- das Optimierungsziel
- eine allgemeine Beschreibung des Produkts
- eine einmalige Bestellung ohne Möglichkeit zur Nachbestellung; das
  beschreibt einen single_period-Planungshorizont
- die laufende Bestandsüberwachung und das Nachbestellen beim Erreichen eines
  Meldebestands; das beschreibt einen continuous_review-Prozess

Übernimm einen Text nur in constraints, wenn darin tatsächlich eine konkrete
Vorgabe oder Grenze beschrieben wird. Merkmale der Problemstruktur gehören
ausschließlich in die dafür vorgesehenen strukturierten Felder.

unsupported_requirements enthält ausdrücklich genannte Anforderungen an
die Problemstruktur, die keinem der erlaubten Werte zugeordnet werden können.

Betroffen sind ausschließlich:
- decision
- objective
- demand_type
- product_scope
- planning_horizon
- review_type

Beispiele:
- "Bestellmenge und Meldebestand bestimmen" kann als order_quantity mit
  continuous_review eingeordnet werden.
- Eine ausdrücklich genannte, andere Überprüfungsart ist ein nicht
  unterstütztes review_type.

Wenn eine solche Anforderung ausdrücklich genannt wird:
- setze das entsprechende strukturierte Feld auf null,
- übernimm die Formulierung in user_value,
- füge eine verständliche deutsche message hinzu,
- trage sie in unsupported_requirements ein.

Nicht als unsupported_requirements behandeln:
- Preise, Kosten oder andere Modellparameter,
- betriebliche Vorgaben wie Budget oder Lagerkapazität,
- bloße Produkteigenschaften,
- Informationen, die lediglich fehlen.

Insbesondere bedeuten genannte Einkaufskosten nicht automatisch, dass
Kosten minimiert werden sollen. Ein objective darf nur gesetzt werden,
wenn der Nutzer ein Optimierungsziel ausdrücklich nennt.

Wenn keine nicht unterstützte strukturelle Anforderung genannt wurde,
gib eine leere Liste zurück.

Erfinde keine Angaben. Wenn ein strukturelles Feld nicht eindeutig aus dem
Freitext erkennbar ist, gib dafür null zurück.

Analysiere den Freitext unabhängig von vorhandenen Formularangaben.
Widersprüche zwischen Formular und Freitext werden anschließend durch
deterministischen Python-Code geprüft.
"""


def _normalize_constraints(
    value: object,
) -> list[str]:
    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        return []

    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _normalize_unsupported_requirements(
    value: object,
) -> list[UnsupportedRequirement]:
    if not isinstance(value, list):
        return []

    allowed_fields = set(STRUCTURAL_FIELDS)
    normalized_requirements: list[UnsupportedRequirement] = []

    for item in value:
        if not isinstance(item, dict):
            continue

        field = item.get("field")
        user_value = item.get("user_value")
        message = item.get("message")

        if not isinstance(field, str):
            continue

        field = field.strip()

        if field not in allowed_fields:
            continue

        if not isinstance(user_value, str):
            continue

        user_value = user_value.strip()

        if not user_value:
            continue

        if not isinstance(message, str) or not message.strip():
            message = (
                f"Die Anforderung „{user_value}“ wird "
                "derzeit nicht unterstützt."
            )
        else:
            message = message.strip()

        normalized_requirements.append(
            {
                "field": field,
                "user_value": user_value,
                "message": message,
            }
        )

    return normalized_requirements


def create_requirement_agent(llm: BaseChatModel) -> Callable[[GraphState], GraphState]:
    """
    Erstellt den Graph-Node zur Extraktion allgemeiner Anforderungen.

    Das LLM analysiert ausschließlich den Freitext. Formularwerte werden
    anschließend deterministisch priorisiert und mit der Extraktion
    zusammengeführt.
    """

    def requirement_agent(state: GraphState) -> GraphState:
        form_data = state.get("form_data", {})
        raw_input_value = state.get("raw_input", "")
        raw_input = (
            raw_input_value
            if isinstance(raw_input_value, str)
            else ""
        )

        user_message = _build_user_message(raw_input)

        # Das LLM wertet nur den Freitext aus. Formularwerte werden
        # anschließend separat und regelbasiert berücksichtigt.
        try:
            extracted = _extract_with_llm(llm, user_message)
        except Exception:
            # Schlägt die LLM-Auswertung fehl, bleiben die Formularwerte
            # erhalten. Aus dem Freitext werden dann keine Angaben übernommen.
            extracted = _extract_rule_based(raw_input)

        form_constraints = _normalize_constraints(
            form_data.get("constraints"),
        )

        extracted_constraints = _normalize_constraints(
            extracted.get("constraints"),
        )

        unsupported_requirements = (
            _normalize_unsupported_requirements(
                extracted.get("unsupported_requirements"),
            )
        )

        constraints = list(
            dict.fromkeys(
                form_constraints + extracted_constraints
            )
        )

        requirement_warnings: list[str] = []
        requirement_conflicts: list[RequirementConflict] = []

        decision = _resolve_structured_field(
            field_name="decision",
            form_data=form_data,
            extracted=extracted,
            allowed_values=ALLOWED_DECISION_TYPES,
            conflicts=requirement_conflicts,
        )

        objective = _resolve_structured_field(
            field_name="objective",
            form_data=form_data,
            extracted=extracted,
            allowed_values=ALLOWED_OBJECTIVE_TYPES,
            conflicts=requirement_conflicts,
        )

        demand_type = _resolve_structured_field(
            field_name="demand_type",
            form_data=form_data,
            extracted=extracted,
            allowed_values=ALLOWED_DEMAND_TYPES,
            conflicts=requirement_conflicts,
        )

        extracted_product_scope = _normalize_choice(
            extracted.get("product_scope"),
            ALLOWED_PRODUCT_SCOPES,
        )

        if (
            extracted_product_scope == "multiple_products"
            and not any(
                requirement.get("field") == "product_scope"
                for requirement in unsupported_requirements
            )
        ):
            unsupported_requirements.append(
                {
                    "field": "product_scope",
                    "user_value": "mehrere Produkte",
                    "message": (
                        "Die aktuelle Version unterstützt "
                        "ausschließlich Optimierungen für ein "
                        "einzelnes Produkt."
                    ),
                }
            )

        # Alle aktuell registrierten Modellvarianten sind Einproduktmodelle.
        product_scope = "single_product"

        planning_horizon = _resolve_structured_field(
            field_name="planning_horizon",
            form_data=form_data,
            extracted=extracted,
            allowed_values=ALLOWED_PLANNING_HORIZONS,
            conflicts=requirement_conflicts,
        )

        review_type = _resolve_structured_field(
            field_name="review_type",
            form_data=form_data,
            extracted=extracted,
            allowed_values=ALLOWED_REVIEW_TYPES,
            conflicts=requirement_conflicts,
        )

        if planning_horizon != "multiple_periods":
            review_type = None

        requirements = {
            "decision": decision,
            "objective": objective,
            "demand_type": demand_type,
            "product_scope": product_scope,
            "planning_horizon": planning_horizon,
            "review_type": review_type,
            "constraints": constraints,
            "raw_description": raw_input,
        }

        missing_fields = [
            field
            for field in REQUIRED_FIELDS
            if not requirements.get(field)
        ]

        if (
            planning_horizon == "multiple_periods"
            and not review_type
        ):
            missing_fields.append("review_type")

        missing_fields = list(dict.fromkeys(missing_fields))

        return with_control(
            {
                **state,
                "requirements": requirements,
                "missing_fields": missing_fields,
                "clarification_question": "",
                "general_validation_errors": [],
                "consistency_warnings": [],
                "requirement_warnings": requirement_warnings,
                "requirement_conflicts": requirement_conflicts,
                "unsupported_requirements": unsupported_requirements,
            },
            event=None,
            workflow_status=WORKFLOW_GENERAL_REQUIREMENTS_EXTRACTED,
        )

    return requirement_agent


# ── LLM ────────────────────────────────────────────────────────────────────

def _extract_with_llm(llm: BaseChatModel, user_message: str) -> dict:
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw_text = str(response.content)
    return _parse_json(raw_text)


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _build_user_message(raw_input: str) -> str:
    if not raw_input.strip():
        return (
            "Der Nutzer hat keine zusätzliche freie Beschreibung "
            "angegeben."
        )

    return (
        "Freie Beschreibung des Nutzers:\n"
        f"{raw_input.strip()}"
    )


def _parse_json(text: str) -> dict:
    """Extrahiert JSON aus LLM-Antwort, auch wenn Markdown-Backticks vorhanden."""

    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    parsed = json.loads(clean)

    if not isinstance(parsed, dict):
        raise TypeError(
            "Die LLM-Antwort muss ein JSON-Objekt sein."
        )

    return parsed


def _normalize_choice(
    value: object,
    allowed_values: set[str],
) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()

    if normalized not in allowed_values:
        return None

    return normalized


def _resolve_structured_field(
    *,
    field_name: str,
    form_data: dict,
    extracted: dict,
    allowed_values: set[str],
    conflicts: list[RequirementConflict],
) -> str | None:
    """
    Führt einen Formularwert und einen extrahierten Freitextwert zusammen.

    Ein vorhandener Formularwert bleibt führend. Abweichende LLM-Werte
    werden als Konflikt gespeichert und überschreiben das Formular nicht.
    """

    form_value = _normalize_choice(
        form_data.get(field_name),
        allowed_values,
    )

    extracted_value = _normalize_choice(
        extracted.get(field_name),
        allowed_values,
    )

    if (
        form_value is not None
        and extracted_value is not None
        and form_value != extracted_value
    ):
        conflicts.append(
            {
                "field": field_name,
                "form_value": form_value,
                "raw_input_value": extracted_value,
                "message": (
                    f"Die Angaben zu '{field_name}' widersprechen "
                    "sich zwischen Formular und Freitext."
                ),
            }
        )

    if form_value is not None:
        return form_value

    return extracted_value


def _extract_rule_based(
    raw_input: str,
) -> dict:
    """
    Sicherer Fallback, wenn die LLM-Auswertung fehlschlägt.

    Der Fallback erfindet keine Informationen aus dem Freitext.
    Formularwerte werden anschließend separat mit den extrahierten
    Werten zusammengeführt.
    """

    return {
        "decision": None,
        "objective": None,
        "demand_type": None,
        "product_scope": None,
        "planning_horizon": None,
        "review_type": None,
        "constraints": [],
        "unsupported_requirements": [],
    }
