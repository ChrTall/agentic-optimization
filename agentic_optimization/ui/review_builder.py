"""
Erstellt die Anzeigedaten für Formular 4 aus Graph-State und Registry-Metadaten.
"""

from typing import Any, TypedDict

from agentic_optimization.models.types import (
    ModelVariantDefinition,
)
from agentic_optimization.state import GraphState


class ReviewItem(TypedDict):
    """Eine einzelne Angabe, die im Review mit Bezeichnung und Wert angezeigt wird."""

    label: str
    value: str


class ReviewData(TypedDict):
    """Datenvertrag für die drei Review-Bereiche in Formular 4."""
    problem: list[ReviewItem]
    parameters: list[ReviewItem]
    constraints: list[ReviewItem]
    unsupported_constraints: list[str]
    model_name: str


DECISION_LABELS = {
    "order_quantity": "Bestellmenge bestimmen",
    "production_quantity": "Produktionsmenge bestimmen",
    "inventory_level": "Lagerbestand bestimmen",
}

OBJECTIVE_LABELS = {
    "maximize_profit": "Gewinn maximieren",
    "minimize_cost": "Kosten minimieren",
    "reach_service_level": "Servicelevel erreichen",
}

DEMAND_TYPE_LABELS = {
    "known": "Sicher oder bekannt",
    "uncertain": "Unsicher",
    "historical_estimate": "Historisch geschätzt",
}

PLANNING_HORIZON_LABELS = {
    "single_period": "Einmalige Entscheidung",
    "multiple_periods": "Mehrere Zeiträume",
}

REVIEW_TYPE_LABELS = {
    "continuous_review": "Kontinuierliche Überprüfung",
    "periodic_review": "Periodische Überprüfung",
}


def _format_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def _format_value(
    value: Any,
    unit: str | None = None,
) -> str:
    if value is None:
        formatted_value = "–"

    elif isinstance(value, bool):
        formatted_value = "Ja" if value else "Nein"

    elif isinstance(value, (int, float)):
        formatted_value = _format_number(value)

    elif isinstance(value, (list, tuple)):
        formatted_value = ", ".join(
            _format_number(item)
            if isinstance(item, (int, float))
            and not isinstance(item, bool)
            else str(item)
            for item in value
        )

    else:
        formatted_value = str(value)

    if unit and formatted_value != "–":
        return f"{formatted_value} {unit}"

    return formatted_value


def _build_problem_items(
    requirements: dict[str, Any],
) -> list[ReviewItem]:
    """Übersetzt die Requirements aus Formular 1 in Review-Zeilen."""

    definitions = [
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
            "Planungszeitraum",
            "planning_horizon",
            PLANNING_HORIZON_LABELS,
        ),
    ]

    if requirements.get("review_type") is not None:
        definitions.append(
            (
                "Bestandsüberprüfung",
                "review_type",
                REVIEW_TYPE_LABELS,
            )
        )

    return [
        {
            "label": label,
            "value": labels.get(
                requirements.get(field_name),
                "–",
            ),
        }
        for label, field_name, labels in definitions
    ]


def _build_parameter_items(
    inputs: dict[str, Any],
    variant_definition: ModelVariantDefinition,
) -> list[ReviewItem]:
    """Erstellt Review-Zeilen aus Modellinputs und Input-Field-Metadaten."""

    items: list[ReviewItem] = []

    for field_definition in variant_definition[
        "input_fields"
    ]:
        field_name = field_definition["name"]

        items.append(
            {
                "label": field_definition["label"],
                "value": _format_value(
                    inputs.get(field_name),
                    field_definition.get("unit"),
                ),
            }
        )

    return items


def _build_constraint_items(
    constraints: list[dict[str, Any]],
    variant_definition: ModelVariantDefinition,
) -> list[ReviewItem]:
    """Erstellt Review-Zeilen für die bestätigten structured_constraints."""

    items: list[ReviewItem] = []

    supported_constraints = variant_definition[
        "supported_constraints"
    ]

    for constraint in constraints:
        constraint_type = constraint.get("type")
        constraint_definition = (
            supported_constraints.get(constraint_type)
        )

        if constraint_definition is None:
            continue

        parameters = constraint.get("parameters", {})
        parameter_definitions = constraint_definition.get(
            "parameters",
            [],
        )

        if not parameter_definitions:
            items.append(
                {
                    "label": constraint_definition["name"],
                    "value": "Berücksichtigt",
                }
            )
            continue

        for parameter_definition in parameter_definitions:
            parameter_name = parameter_definition["name"]

            label = constraint_definition["name"]

            if len(parameter_definitions) > 1:
                label = (
                    f"{label}: "
                    f"{parameter_definition['label']}"
                )

            items.append(
                {
                    "label": label,
                    "value": _format_value(
                        parameters.get(parameter_name),
                        parameter_definition.get("unit"),
                    ),
                }
            )

    return items


def build_review_data(
    state: GraphState,
    variant_definition: ModelVariantDefinition,
) -> ReviewData:
    """
    Erstellt die drei Review-Bereiche aus dem aktuellen Graph-State.

    Verwendet werden die Requirements, die Inputs der ausgewählten Variante
    und die bestätigten structured_constraints.
    """

    family = state.get("selected_model_family")
    variant = state.get("selected_model_variant")

    inputs = (
        state.get("model_specific_inputs", {})
        .get(family or "", {})
        .get(variant or "", {})
    )

    requirements = state.get("requirements", {})
    constraints = state.get(
        "structured_constraints",
        [],
    )
    unsupported_constraints = [
        constraint.strip()
        for constraint in state.get(
            "unsupported_constraints",
            [],
        )
        if isinstance(constraint, str)
        and constraint.strip()
    ]

    model_config = state.get("model_config", {})

    model_name = model_config.get(
        "variant_name",
        variant_definition["variant_name"],
    )

    return {
        "problem": _build_problem_items(
            requirements,
        ),
        "parameters": _build_parameter_items(
            inputs,
            variant_definition,
        ),
        "constraints": _build_constraint_items(
            constraints,
            variant_definition,
        ),
        "unsupported_constraints": unsupported_constraints,
        "model_name": str(model_name),
    }
