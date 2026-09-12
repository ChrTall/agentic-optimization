"""
Fachlicher Vertrag des klassischen EOQ-Grundmodells.
"""

import math
from dataclasses import dataclass

from agentic_optimization.models.execution import (
    DecisionValue,
    MetricValue,
    OptimizationExecutionResult,
)
from agentic_optimization.models.specification import (
    DemandForecast,
    ModelSpecification,
)
from agentic_optimization.models.types import (
    ModelCoreDefinition,
)


CORE_DEFINITION: ModelCoreDefinition = {
    "core_id": "eoq",
    "name": "Wirtschaftliche Bestellmenge (EOQ)",
    "minimum_products": 1,
    "required_product_facts": [],
    "required_global_facts": [],
    "required_objective_component_ids": [
        "annual_ordering_cost",
        "annual_holding_cost",
    ],
    "required_forecast_fields": [
        {
            "name": "expected_value",
            "label": "Erwartete Nachfrage",
            "type": "number",
            "required": True,
            "help": (
                "Erwartete Nachfrage innerhalb des "
                "angegebenen Prognosezeitraums."
            ),
            "min_value": 0.01,
            "unit": "Einheiten",
        },
        {
            "name": "horizon_days",
            "label": "Prognosezeitraum",
            "type": "integer",
            "required": True,
            "help": (
                "Anzahl der Tage, auf die sich die "
                "erwartete Nachfrage bezieht."
            ),
            "min_value": 1,
            "unit": "Tage",
        },
    ],
    "provided_quantities": {
        "order_quantity": {
            "quantity_id": "order_quantity",
            "name": "Bestellmenge",
            "role": "decision",
            "scope": "per_product",
            "description": (
                "Menge, die je Bestellvorgang bestellt wird."
            ),
        },
        "orders_per_year": {
            "quantity_id": "orders_per_year",
            "name": "Bestellungen pro Jahr",
            "role": "derived",
            "scope": "per_product",
            "description": (
                "Jährlicher Bedarf geteilt durch die Bestellmenge."
            ),
        },
        "average_inventory": {
            "quantity_id": "average_inventory",
            "name": "Durchschnittlicher Lagerbestand",
            "role": "derived",
            "scope": "per_product",
            "description": (
                "Hälfte der Bestellmenge im klassischen EOQ-Modell."
            ),
        },
        "cycle_length_days": {
            "quantity_id": "cycle_length_days",
            "name": "Zyklusdauer",
            "role": "derived",
            "scope": "per_product",
            "description": (
                "Durchschnittliche Dauer zwischen zwei Bestellungen."
            ),
        },
    },
}


@dataclass(frozen=True)
class EoqProductResult:
    """Unformatierte EOQ-Ergebnisse für ein Produkt."""

    annual_demand: float
    unconstrained_order_quantity: float
    order_quantity: float
    orders_per_year: float
    average_inventory: float
    cycle_length_days: float
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_relevant_cost: float


def annualize_expected_demand(
    forecast: DemandForecast,
) -> float:
    """Rechnet die erwartete Nachfrage auf ein Jahr hoch."""

    expected_value = float(forecast["expected_value"])
    horizon_days = int(forecast["horizon_days"])

    return expected_value * 365.0 / horizon_days


def calculate_eoq_product(
    annual_demand: float,
    ordering_cost: float,
    holding_cost: float,
    minimum_order_quantity: float | None = None,
    maximum_order_quantity: float | None = None,
) -> EoqProductResult:
    """Berechnet die klassische EOQ für ein Produkt."""

    unconstrained_order_quantity = math.sqrt(
        2.0
        * annual_demand
        * ordering_cost
        / holding_cost
    )

    order_quantity = unconstrained_order_quantity

    if minimum_order_quantity is not None:
        order_quantity = max(
            order_quantity,
            minimum_order_quantity,
        )

    if maximum_order_quantity is not None:
        order_quantity = min(
            order_quantity,
            maximum_order_quantity,
        )

    orders_per_year = annual_demand / order_quantity
    average_inventory = order_quantity / 2.0
    cycle_length_days = (
        order_quantity / annual_demand * 365.0
    )
    annual_ordering_cost = (
        orders_per_year * ordering_cost
    )
    annual_holding_cost = (
        average_inventory * holding_cost
    )
    annual_relevant_cost = (
        annual_ordering_cost
        + annual_holding_cost
    )

    return EoqProductResult(
        annual_demand=annual_demand,
        unconstrained_order_quantity=(
            unconstrained_order_quantity
        ),
        order_quantity=order_quantity,
        orders_per_year=orders_per_year,
        average_inventory=average_inventory,
        cycle_length_days=cycle_length_days,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        annual_relevant_cost=annual_relevant_cost,
    )


def validate_eoq_forecast(
    forecast: DemandForecast,
) -> list[str]:
    """Prüft die EOQ-spezifische Bedeutung des Forecasts."""

    annual_demand = annualize_expected_demand(forecast)

    if not math.isfinite(annual_demand) or annual_demand <= 0:
        return [
            "Der hochgerechnete Jahresbedarf muss "
            "eine positive endliche Zahl sein."
        ]

    return []


def validate_eoq_semantics(
    specification: ModelSpecification,
) -> list[str]:
    """Prüft die fachliche Zusammensetzung des EOQ-Modells."""

    errors: list[str] = []
    objective = specification["objective"]

    if objective["sense"] != "minimize":
        errors.append(
            "Das klassische EOQ-Modell muss die "
            "jährlichen relevanten Kosten minimieren."
        )

    for constraint in specification["constraints"]:
        if (
            constraint["component_id"]
            == "maximum_order_quantity"
            and float(
                constraint["parameters"]["quantity"]
            )
            <= 0
        ):
            errors.append(
                "Die Höchstbestellmenge muss für das "
                "EOQ-Modell größer als null sein."
            )

    return errors


def _get_order_quantity_bounds(
    specification: ModelSpecification,
    product_id: str,
) -> tuple[float | None, float | None]:
    """Liest die Bestellmengengrenzen eines Produkts."""

    minimum: float | None = None
    maximum: float | None = None

    for constraint in specification["constraints"]:
        if constraint.get("product_id") != product_id:
            continue

        component_id = constraint["component_id"]
        quantity = float(
            constraint["parameters"]["quantity"]
        )

        if component_id == "minimum_order_quantity":
            minimum = quantity
        elif component_id == "maximum_order_quantity":
            maximum = quantity

    return minimum, maximum


def execute_eoq(
    specification: ModelSpecification,
) -> OptimizationExecutionResult:
    """Validiert und löst eine modulare EOQ-Spezifikation."""

    from agentic_optimization.models.validator import (
        validate_model_specification,
    )

    validation = validate_model_specification(specification)

    if not validation["is_valid"]:
        messages = list(validation["errors"])

        messages.extend(
            "Fehlende Eingabe: "
            f"{missing_input['field']['name']}"
            for missing_input in validation["missing_inputs"]
        )

        raise ValueError(
            "Die EOQ-Spezifikation ist nicht vollständig: "
            + "; ".join(messages)
        )

    decisions: list[DecisionValue] = []
    metrics: list[MetricValue] = []
    representations: list[str] = []
    total_annual_cost = 0.0

    for product in specification["products"]:
        product_id = product["product_id"]
        facts = product["facts"]
        forecast = product["forecast"]

        annual_demand = annualize_expected_demand(
            forecast
        )
        ordering_cost = float(facts["ordering_cost"])
        holding_cost = float(facts["holding_cost"])

        minimum, maximum = _get_order_quantity_bounds(
            specification,
            product_id,
        )

        result = calculate_eoq_product(
            annual_demand=annual_demand,
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
            minimum_order_quantity=minimum,
            maximum_order_quantity=maximum,
        )

        total_annual_cost += result.annual_relevant_cost

        decisions.append(
            {
                "quantity_id": "order_quantity",
                "product_id": product_id,
                "value": result.order_quantity,
            }
        )

        product_metrics = {
            "annual_demand": result.annual_demand,
            "unconstrained_order_quantity": (
                result.unconstrained_order_quantity
            ),
            "orders_per_year": result.orders_per_year,
            "average_inventory": result.average_inventory,
            "cycle_length_days": result.cycle_length_days,
            "annual_ordering_cost": (
                result.annual_ordering_cost
            ),
            "annual_holding_cost": (
                result.annual_holding_cost
            ),
            "annual_relevant_cost": (
                result.annual_relevant_cost
            ),
        }

        metrics.extend(
            {
                "metric_id": metric_id,
                "product_id": product_id,
                "value": value,
            }
            for metric_id, value in product_metrics.items()
        )

        representations.append(
            "\n".join(
                [
                    f"Produkt: {product_id}",
                    "Q* = sqrt(2 * D * S / H)",
                    f"D = {annual_demand}",
                    f"S = {ordering_cost}",
                    f"H = {holding_cost}",
                    (
                        "Unbeschränkte EOQ = "
                        f"{result.unconstrained_order_quantity}"
                    ),
                    f"Mindestbestellmenge = {minimum}",
                    f"Höchstbestellmenge = {maximum}",
                    (
                        "Empfohlene Bestellmenge = "
                        f"{result.order_quantity}"
                    ),
                ]
            )
        )

    return {
        "status": "optimal",
        "solution_method": "analytical",
        "objective_value": total_annual_cost,
        "decisions": decisions,
        "metrics": metrics,
        "technical_representation": "\n\n".join(
            representations
        ),
        "errors": [],
    }