"""
Implementiert die klassische EOQ-Variante für ein Produkt.

Die Variante bestimmt die optimale Bestellmenge bei bekannter,
konstanter Nachfrage mit der analytischen EOQ-Formel.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any

from agentic_optimization.models.types import (
    ConstraintDefinition,
    ModelInputField,
    ModelVariantDefinition,
    ResultMetric,
    StructuredConstraint,
    VariantResultPresentation,
)

INPUT_FIELDS: list[ModelInputField] = [
    {
        "name": "annual_demand",
        "label": "Jahresbedarf",
        "type": "number",
        "required": True,
        "help": (
            "Gesamter erwarteter Bedarf innerhalb eines Jahres."
        ),
        "placeholder": None,
        "min_value": 0.01,
        "step": 1.0,
        "unit": "Einheiten pro Jahr",
    },
    {
        "name": "ordering_cost",
        "label": "Bestellkosten pro Bestellung",
        "type": "number",
        "required": True,
        "help": (
            "Fixe Kosten, die bei jeder Bestellung entstehen."
        ),
        "placeholder": None,
        "min_value": 0.01,
        "step": 1.0,
        "unit": "€ pro Bestellung",
    },
    {
        "name": "holding_cost",
        "label": "Lagerhaltungskosten",
        "type": "number",
        "required": True,
        "help": (
            "Jährliche Lagerhaltungskosten pro gelagerter Einheit."
        ),
        "placeholder": None,
        "min_value": 0.01,
        "step": 0.5,
        "unit": "€ pro Einheit und Jahr",
    },
]


SUPPORTED_CONSTRAINTS: dict[
    str,
    ConstraintDefinition,
] = {
    "minimum_order_quantity": {
        "type": "minimum_order_quantity",
        "name": "Mindestbestellmenge",
        "parameters": [
            {
                "name": "quantity",
                "label": "Mindestbestellmenge",
                "type": "number",
                "required": True,
                "help": (
                    "Die mindestens zu bestellende Menge."
                ),
                "placeholder": None,
                "min_value": 1.0,
                "step": 1.0,
                "unit": "Einheiten",
            }
        ],
    },
    "maximum_order_quantity": {
        "type": "maximum_order_quantity",
        "name": "Höchstbestellmenge",
        "parameters": [
            {
                "name": "quantity",
                "label": "Höchstbestellmenge",
                "type": "number",
                "required": True,
                "help": (
                    "Die maximal zulässige Bestellmenge."
                ),
                "placeholder": None,
                "min_value": 1.0,
                "step": 1.0,
                "unit": "Einheiten",
            }
        ],
    },
}


def validate_basic_eoq_inputs(
    inputs: dict[str, Any],
) -> list[str]:
    """Prüft die drei erforderlichen Eingaben des klassischen EOQ-Modells."""

    errors: list[str] = []

    input_labels = {
        "annual_demand": "Der Jahresbedarf",
        "ordering_cost": "Die Bestellkosten",
        "holding_cost": "Die Lagerhaltungskosten",
    }

    for field_name, label in input_labels.items():
        value = inputs.get(field_name)

        if value is None:
            errors.append(
                f"{label} muss angegeben werden."
            )
            continue

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            errors.append(
                f"{label} muss eine endliche Zahl sein."
            )
            continue

        if value <= 0:
            errors.append(
                f"{label} muss größer als 0 sein."
            )

    return errors


def validate_basic_eoq_constraints(
    _inputs: dict[str, Any],
    constraints: list[StructuredConstraint],
) -> list[str]:
    """Prüft Mindest- und Höchstbestellmenge."""

    errors: list[str] = []

    minimum_quantity: float | None = None
    maximum_quantity: float | None = None

    for constraint in constraints:
        constraint_type = constraint["type"]
        parameters = constraint["parameters"]

        if constraint_type not in SUPPORTED_CONSTRAINTS:
            errors.append(
                f"Der Constraint-Typ '{constraint_type}' "
                "wird von Basic EOQ nicht unterstützt."
            )
            continue

        quantity = parameters.get("quantity")

        if quantity is None:
            continue

        if (
            not isinstance(quantity, (int, float))
            or isinstance(quantity, bool)
            or not math.isfinite(quantity)
        ):
            errors.append(
                "Die Bestellmengenbegrenzung muss eine "
                "endliche Zahl sein."
            )
            continue

        if quantity <= 0:
            errors.append(
                "Die Bestellmengenbegrenzung muss größer "
                "als 0 sein."
            )
            continue

        if constraint_type == "minimum_order_quantity":
            minimum_quantity = float(quantity)
        elif constraint_type == "maximum_order_quantity":
            maximum_quantity = float(quantity)

    if (
        minimum_quantity is not None
        and maximum_quantity is not None
        and minimum_quantity > maximum_quantity
    ):
        errors.append(
            "Die Mindestbestellmenge darf die "
            "Höchstbestellmenge nicht überschreiten."
        )

    return errors


@dataclass(frozen=True)
class BasicEoqSpecification:
    """Normalisierte Eingaben für die analytische EOQ-Berechnung."""

    annual_demand: float
    ordering_cost: float
    holding_cost: float
    minimum_order_quantity: float | None = None
    maximum_order_quantity: float | None = None


@dataclass(frozen=True)
class BasicEoqResult:
    """Fachliche Ergebnisse der analytischen EOQ-Berechnung."""

    solver_status: str
    unconstrained_order_quantity: float
    order_quantity: float
    orders_per_year: float
    average_inventory: float
    cycle_length_years: float
    cycle_length_days: float
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_relevant_cost: float
    model_representation: str


def create_basic_eoq_specification(
    inputs: dict[str, Any],
    constraints: list[StructuredConstraint] | None = None,
) -> BasicEoqSpecification:
    """Erstellt eine normalisierte EOQ-Spezifikation."""

    constraints_by_type = {
        constraint["type"]: constraint
        for constraint in constraints or []
    }

    def get_optional_quantity(
        constraint_type: str,
    ) -> float | None:
        constraint = constraints_by_type.get(
            constraint_type
        )

        if constraint is None:
            return None

        quantity = constraint[
            "parameters"
        ].get("quantity")

        if quantity is None:
            return None

        return float(quantity)

    return BasicEoqSpecification(
        annual_demand=float(
            inputs["annual_demand"]
        ),
        ordering_cost=float(
            inputs["ordering_cost"]
        ),
        holding_cost=float(
            inputs["holding_cost"]
        ),
        minimum_order_quantity=(
            get_optional_quantity(
                "minimum_order_quantity"
            )
        ),
        maximum_order_quantity=(
            get_optional_quantity(
                "maximum_order_quantity"
            )
        ),
    )


def solve_basic_eoq(
    specification: BasicEoqSpecification,
) -> BasicEoqResult:
    """Berechnet die optimale Bestellmenge mit der klassischen EOQ-Formel."""

    unconstrained_order_quantity = math.sqrt(
        (
            2
            * specification.annual_demand
            * specification.ordering_cost
        )
        / specification.holding_cost
    )

    order_quantity = unconstrained_order_quantity

    if specification.minimum_order_quantity is not None:
        order_quantity = max(
            order_quantity,
            specification.minimum_order_quantity,
        )

    if specification.maximum_order_quantity is not None:
        order_quantity = min(
            order_quantity,
            specification.maximum_order_quantity,
        )

    orders_per_year = (
        specification.annual_demand
        / order_quantity
    )

    average_inventory = order_quantity / 2

    cycle_length_years = (
        order_quantity
        / specification.annual_demand
    )

    cycle_length_days = (
        cycle_length_years * 365
    )

    annual_ordering_cost = (
        orders_per_year
        * specification.ordering_cost
    )

    annual_holding_cost = (
        average_inventory
        * specification.holding_cost
    )

    annual_relevant_cost = (
        annual_ordering_cost
        + annual_holding_cost
    )

    model_representation = "\n".join(
        [
            "Q* = sqrt(2 * D * S / H)",
            f"D = {specification.annual_demand}",
            f"S = {specification.ordering_cost}",
            f"H = {specification.holding_cost}",
            (
                "Unbeschränkte EOQ = "
                f"{unconstrained_order_quantity}"
            ),
            (
                "Mindestbestellmenge = "
                f"{specification.minimum_order_quantity}"
            ),
            (
                "Höchstbestellmenge = "
                f"{specification.maximum_order_quantity}"
            ),
            (
                "Empfohlene Bestellmenge = "
                f"{order_quantity}"
            ),
        ]
    )

    return BasicEoqResult(
        solver_status="Optimal",
        unconstrained_order_quantity=(
            unconstrained_order_quantity
        ),
        order_quantity=order_quantity,
        orders_per_year=orders_per_year,
        average_inventory=average_inventory,
        cycle_length_years=cycle_length_years,
        cycle_length_days=cycle_length_days,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        annual_relevant_cost=annual_relevant_cost,
        model_representation=model_representation,
    )


def optimize_basic_eoq(
    inputs: dict[str, Any],
    constraints: list[StructuredConstraint],
) -> dict[str, Any]:
    """Validiert und berechnet die begrenzte EOQ."""

    input_errors = validate_basic_eoq_inputs(
        inputs
    )

    constraint_errors = (
        validate_basic_eoq_constraints(
            inputs,
            constraints,
        )
    )

    errors = [
        *input_errors,
        *constraint_errors,
    ]

    if errors:
        raise ValueError(" ".join(errors))

    specification = create_basic_eoq_specification(
        inputs,
        constraints,
    )

    result = solve_basic_eoq(
        specification
    )

    return asdict(result)


def _format_result_number(
    value: float | None,
) -> str:
    """Formatiert eine Ergebniszahl mit zwei Nachkommastellen."""

    if value is None:
        return "–"

    formatted = f"{float(value):,.2f}"

    return (
        formatted
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )


def _format_result_currency(
    value: float | None,
) -> str:
    """Formatiert einen Ergebniswert als Eurobetrag."""

    if value is None:
        return "–"

    return (
        f"{_format_result_number(value)} €"
    )


def build_basic_eoq_result_presentation(
    optimization_result: dict[str, Any],
) -> VariantResultPresentation:
    """Bereitet EOQ-Kennzahlen und Berechnungsdetails für die UI auf."""

    metrics: list[ResultMetric] = [
        {
            "label": "Empfohlene Bestellmenge",
            "value": (
                f"{_format_result_number(
                    optimization_result.get(
                        'order_quantity'
                    )
                )} Einheiten"
            ),
        },
        {
            "label": (
                "Jährliche relevante Gesamtkosten"
            ),
            "value": _format_result_currency(
                optimization_result.get(
                    "annual_relevant_cost"
                )
            ),
        },
    ]

    detail_rows = [
        {
            "Kennzahl": "Bestellungen pro Jahr",
            "Wert": _format_result_number(
                optimization_result.get(
                    "orders_per_year"
                )
            ),
        },
        {
            "Kennzahl": "Durchschnittlicher Lagerbestand",
            "Wert": (
                f"{_format_result_number(
                    optimization_result.get(
                        'average_inventory'
                    )
                )} Einheiten"
            ),
        },
        {
            "Kennzahl": "Durchschnittliche Zyklusdauer",
            "Wert": (
                f"{_format_result_number(
                    optimization_result.get(
                        'cycle_length_days'
                    )
                )} Tage"
            ),
        },
        {
            "Kennzahl": "Jährliche Bestellkosten",
            "Wert": _format_result_currency(
                optimization_result.get(
                    "annual_ordering_cost"
                )
            ),
        },
        {
            "Kennzahl": "Jährliche Lagerhaltungskosten",
            "Wert": _format_result_currency(
                optimization_result.get(
                    "annual_holding_cost"
                )
            ),
        },
    ]

    return {
        "metrics": metrics,
        "detail_title": "EOQ-Berechnungsdetails",
        "detail_rows": detail_rows,
    }


VARIANT_DEFINITION: ModelVariantDefinition = {
    "variant_id": "basic_eoq",
    "variant_name": "Basic EOQ",
    "selection_criteria": {
        "decision": "order_quantity",
        "objective": "minimize_cost",
        "demand_type": "known",
        "product_scope": "single_product",
        "planning_horizon": "repeated_cycles",
    },
    "input_fields": INPUT_FIELDS,
    "supported_constraints": SUPPORTED_CONSTRAINTS,
    "input_validator": validate_basic_eoq_inputs,
    "constraint_validator": (
        validate_basic_eoq_constraints
    ),
    "optimizer": optimize_basic_eoq,
    "result_builder": (
        build_basic_eoq_result_presentation
    ),
}
