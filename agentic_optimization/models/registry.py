from typing import Any, Callable, TypedDict
import math


class ModelInputField(TypedDict):
    """
    Beschreibt ein einzelnes Eingabefeld für ein Modellformular.
    """

    name: str
    label: str
    type: str
    required: bool
    help: str | None


class ModelDefinition(TypedDict):
    """
    Beschreibt ein unterstütztes Optimierungsmodell.

    Enthält die benötigten Pflichtfelder, Formularfelder
    und die passende Validierungsfunktion.
    """

    model_type: str
    model_name: str
    required_fields: list[str]
    input_fields: list[ModelInputField]
    validator: Callable[[dict[str, Any]], list[str]]


def validate_newsvendor_inputs(inputs: dict[str, Any]) -> list[str]:
    """
    Prüft die eingegebenen Parameter für das Newsvendor-Modell.

    Gibt eine Liste mit Fehlermeldungen zurück.
    Ist die Liste leer, sind die Eingaben gültig.
    """

    errors: list[str] = []

    selling_price = inputs.get("selling_price")
    purchase_cost = inputs.get("purchase_cost")
    salvage_value = inputs.get("salvage_value")
    demand_scenarios = inputs.get("demand_scenarios") or []
    scenario_probabilities = inputs.get("scenario_probabilities") or []

    if not isinstance(demand_scenarios, list):
        errors.append("Nachfrageszenarien müssen als Liste angegeben werden.")
        demand_scenarios = []

    if not isinstance(scenario_probabilities, list):
        errors.append("Wahrscheinlichkeiten müssen als Liste angegeben werden.")
        scenario_probabilities = []

    numeric_values = {
        "selling_price": selling_price,
        "purchase_cost": purchase_cost,
        "salvage_value": salvage_value,
    }
    numeric_labels = {
        "selling_price": "Der Verkaufspreis",
        "purchase_cost": "Die Einkaufskosten",
        "salvage_value": "Der Restwert",
    }

    for field_name, value in numeric_values.items():
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            errors.append(f"{numeric_labels[field_name]} muss eine endliche Zahl sein.")

    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in demand_scenarios
    ):
        errors.append("Nachfrageszenarien müssen endliche Zahlen sein.")

    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in scenario_probabilities
    ):
        errors.append("Wahrscheinlichkeiten müssen endliche Zahlen sein.")

    if errors:
        return errors

    if selling_price is None or selling_price <= 0:
        errors.append("Der Verkaufspreis muss größer als 0 sein.")

    if purchase_cost is None or purchase_cost <= 0:
        errors.append("Die Einkaufskosten müssen größer als 0 sein.")

    if (
        selling_price is not None
        and purchase_cost is not None
        and selling_price <= purchase_cost
    ):
        errors.append("Der Verkaufspreis muss größer als die Einkaufskosten sein.")

    if salvage_value is None:
        errors.append("Der Restwert muss angegeben werden.")

    if (
        salvage_value is not None
        and purchase_cost is not None
        and salvage_value > purchase_cost
    ):
        errors.append("Der Restwert darf nicht größer als die Einkaufskosten sein.")

    if not demand_scenarios:
        errors.append("Bitte gib mindestens ein Nachfrageszenario ein.")

    if not scenario_probabilities:
        errors.append("Bitte gib mindestens eine Wahrscheinlichkeit ein.")

    if (
        demand_scenarios
        and scenario_probabilities
        and len(demand_scenarios) != len(scenario_probabilities)
    ):
        errors.append(
            "Die Anzahl der Nachfrageszenarien muss zur Anzahl der Wahrscheinlichkeiten passen."
        )

    if any(demand < 0 for demand in demand_scenarios):
        errors.append("Nachfrageszenarien dürfen nicht negativ sein.")

    if any(
        probability < 0 or probability > 1
        for probability in scenario_probabilities
    ):
        errors.append("Wahrscheinlichkeiten müssen zwischen 0 und 1 liegen.")

    if scenario_probabilities and abs(sum(scenario_probabilities) - 1.0) > 0.001:
        errors.append("Die Wahrscheinlichkeiten müssen zusammen 1 ergeben.")

    return errors


# Zentrale Registry aller unterstützten Modelltypen.
MODEL_REGISTRY: dict[str, ModelDefinition] = {
    "newsvendor": {
        "model_type": "newsvendor",
        "model_name": "Newsvendor-Modell",
        "required_fields": [
            "selling_price",
            "purchase_cost",
            "salvage_value",
            "demand_scenarios",
            "scenario_probabilities",
        ],
        "input_fields": [
            {
                "name": "selling_price",
                "label": "Verkaufspreis pro Einheit",
                "type": "number",
                "required": True,
                "help": None,
            },
            {
                "name": "purchase_cost",
                "label": "Einkaufskosten pro Einheit",
                "type": "number",
                "required": True,
                "help": None,
            },
            {
                "name": "salvage_value",
                "label": "Restwert pro unverkaufter Einheit",
                "type": "number",
                "required": True,
                "help": None,
            },
            {
                "name": "demand_scenarios",
                "label": "Nachfrageszenarien",
                "type": "number_list",
                "required": True,
                "help": "Kommagetrennt, z.B. 80, 100, 120",
            },
            {
                "name": "scenario_probabilities",
                "label": "Wahrscheinlichkeiten der Szenarien",
                "type": "number_list",
                "required": True,
                "help": "Kommagetrennt, z.B. 0.25, 0.5, 0.25",
            },
        ],
        "validator": validate_newsvendor_inputs,
    }
}


def get_model_definition(model_type: str | None) -> ModelDefinition | None:
    """
    Gibt die ModelDefinition für einen Modelltyp zurück.

    Falls der Modelltyp unbekannt oder leer ist, wird None zurückgegeben.
    """

    if not model_type:
        return None

    return MODEL_REGISTRY.get(model_type)