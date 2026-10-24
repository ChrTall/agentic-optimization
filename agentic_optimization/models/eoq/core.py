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
    "description": (
        "Dieses Modell unterstützt wiederkehrende "
        "Bestellzyklen bei einem gleichmäßig angenommenen "
        "Bedarf."
    ),
    "decision_description": (
        "Berechnet wird, wie viele Einheiten bei jeder "
        "Bestellung bestellt werden sollen."
    ),
    "assumptions": [
        (
            "Der Bedarf wird über die Zeit als gleichmäßig "
            "und planbar angenommen."
        ),
        (
            "Der angegebene Prognosezeitraum wird zur "
            "Ermittlung des Jahresbedarfs hochgerechnet."
        ),
        (
            "Fehlbestände und Mengenrabatte werden im "
            "aktuellen Grundmodell nicht berücksichtigt."
        ),
    ],
    "minimum_products": 1,
    "required_product_facts": [],
    "required_global_facts": [],
    "required_objective_component_ids": [
        "annual_ordering_cost",
        "annual_holding_cost",
    ],
    "required_objective_sense": "minimize",
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
    order_multiple: int | None = None,
) -> EoqProductResult:
    """Berechnet die klassische EOQ für ein Produkt."""

    unconstrained_order_quantity = math.sqrt(
        2.0
        * annual_demand
        * ordering_cost
        / holding_cost
    )

    # Empfohlen werden ganze Einheiten: Ohne Bestellvielfaches gilt das
    # Vielfache 1. Verglichen werden die ganzzahligen Nachbarn der EOQ.
    if order_multiple is None:
        order_multiple = 1

    if order_multiple <= 0:
        raise ValueError(
            "Das Bestellvielfache muss größer als null sein."
        )

    minimum_factor = max(
        1,
        math.ceil(
            (minimum_order_quantity or 0.0)
            / order_multiple
        ),
    )

    maximum_factor = (
        math.floor(
            maximum_order_quantity / order_multiple
        )
        if maximum_order_quantity is not None
        else None
    )

    if (
        maximum_factor is not None
        and minimum_factor > maximum_factor
    ):
        raise ValueError(
            "Innerhalb der Bestellmengengrenzen "
            "existiert kein zulässiges Bestellvielfaches."
        )

    unconstrained_factor = (
        unconstrained_order_quantity / order_multiple
    )

    factor_candidates = {
        minimum_factor,
        max(
            minimum_factor,
            math.floor(unconstrained_factor),
        ),
        max(
            minimum_factor,
            math.ceil(unconstrained_factor),
        ),
    }

    if maximum_factor is not None:
        factor_candidates = {
            min(factor, maximum_factor)
            for factor in factor_candidates
        }
        factor_candidates.add(maximum_factor)

    quantity_candidates = {
        float(factor * order_multiple)
        for factor in factor_candidates
        if factor >= minimum_factor
        and (
            maximum_factor is None
            or factor <= maximum_factor
        )
    }

    def relevant_cost(quantity: float) -> float:
        return (
            annual_demand * ordering_cost / quantity
            + holding_cost * quantity / 2.0
        )

    order_quantity = min(
        quantity_candidates,
        key=lambda quantity: (
            relevant_cost(quantity),
            quantity,
        ),
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

    for product in specification["products"]:
        product_id = product["product_id"]

        minimum, maximum = _get_order_quantity_bounds(
            specification,
            product_id,
        )
        order_multiple = _get_order_multiple(
            specification,
            product_id,
        )

        if (
            maximum is not None
            and order_multiple is not None
        ):
            minimum_factor = max(
                1,
                math.ceil(
                    (minimum or 0.0) / order_multiple
                ),
            )

            if minimum_factor * order_multiple > maximum:
                errors.append(
                    f"Für das EOQ-Produkt '{product_id}' "
                    "existiert innerhalb der Bestellmengengrenzen "
                    "kein positives zulässiges Bestellvielfaches."
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

        if component_id not in {
            "minimum_order_quantity",
            "maximum_order_quantity",
        }:
            continue

        quantity = float(
            constraint["parameters"]["quantity"]
        )

        if component_id == "minimum_order_quantity":
            minimum = quantity
        elif component_id == "maximum_order_quantity":
            maximum = quantity

    return minimum, maximum


def _get_order_multiple(
    specification: ModelSpecification,
    product_id: str,
) -> int | None:
    """Liest das Bestellvielfache eines Produkts."""

    for constraint in specification["constraints"]:
        if (
            constraint.get("product_id") == product_id
            and constraint["component_id"]
            == "order_multiple"
        ):
            return int(
                constraint["parameters"]["multiple"]
            )

    return None


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

        order_multiple = _get_order_multiple(
            specification,
            product_id,
        )

        minimum_text = (
            "nicht gesetzt"
            if minimum is None
            else str(minimum)
        )
        maximum_text = (
            "nicht gesetzt"
            if maximum is None
            else str(maximum)
        )

        result = calculate_eoq_product(
            annual_demand=annual_demand,
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
            minimum_order_quantity=minimum,
            maximum_order_quantity=maximum,
            order_multiple=order_multiple,
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
                    (
                        "Mindestbestellmenge = "
                        f"{minimum_text}"
                    ),
                    (
                        "Höchstbestellmenge = "
                        f"{maximum_text}"
                    ),
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
