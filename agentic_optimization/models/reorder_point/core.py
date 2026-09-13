"""
Fachlicher Vertrag der kontinuierlichen Q,R-Bestandspolitik.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist

from agentic_optimization.models.forecast_adapter import (
    prepare_continuous_review_forecast,
)
from agentic_optimization.models.specification import (
    DemandForecast,
    ModelSpecification,
)
from agentic_optimization.models.types import (
    ModelCoreDefinition,
)
from agentic_optimization.models.execution import (
    DecisionValue,
    MetricValue,
    OptimizationExecutionResult,
)

_STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class QRPolicyEvaluation:
    """Kosten und Kennzahlen eines einzelnen Q,R-Kandidaten."""

    order_quantity: float
    reorder_point: float
    orders_per_year: float
    safety_stock: float
    average_inventory: float
    expected_shortage_per_cycle: float
    annual_shortage_quantity: float
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_shortage_cost: float
    annual_relevant_cost: float


CORE_DEFINITION: ModelCoreDefinition = {
    "core_id": "continuous_review_q_r",
    "name": "Kontinuierliche Bestandsüberprüfung (Q,R)",
    "minimum_products": 1,
    "required_product_facts": [
        {
            "name": "lead_time_days",
            "label": "Lieferzeit",
            "type": "integer",
            "required": True,
            "help": (
                "Zeit zwischen Bestellung und Wareneingang."
            ),
            "min_value": 1,
            "unit": "Tage",
        }
    ],
    "required_global_facts": [],
    "required_objective_component_ids": [
        "annual_ordering_cost",
        "annual_holding_cost",
        "annual_shortage_cost",
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
            "name": "quantiles",
            "label": "Nachfragequantile",
            "type": "forecast_quantiles",
            "required": True,
            "help": (
                "Mindestens ein Quantil unterhalb und "
                "eines oberhalb von 0,5."
            ),
            "unit": "Einheiten",
        },
        {
            "name": "horizon_days",
            "label": "Prognosezeitraum",
            "type": "integer",
            "required": True,
            "help": (
                "Anzahl der Tage, auf die sich der "
                "Forecast bezieht."
            ),
            "min_value": 1,
            "unit": "Tage",
        },
    ],
    "provided_quantities": {
        "order_quantity": {
            "quantity_id": "order_quantity",
            "name": "Bestellmenge Q",
            "role": "decision",
            "scope": "per_product",
            "description": (
                "Menge, die bei einer Bestellung ausgelöst wird."
            ),
        },
        "reorder_point": {
            "quantity_id": "reorder_point",
            "name": "Meldebestand R",
            "role": "decision",
            "scope": "per_product",
            "description": (
                "Bestandsposition, bei der eine Bestellung "
                "ausgelöst wird."
            ),
        },
        "orders_per_year": {
            "quantity_id": "orders_per_year",
            "name": "Bestellungen pro Jahr",
            "role": "derived",
            "scope": "per_product",
            "description": (
                "Jährlicher Bedarf geteilt durch "
                "die Bestellmenge."
            ),
        },
        "average_inventory": {
            "quantity_id": "average_inventory",
            "name": "Durchschnittlicher Lagerbestand",
            "role": "derived",
            "scope": "per_product",
            "description": (
                "Durchschnittlicher Bestand unter der "
                "gewählten Q,R-Politik."
            ),
        },
        "expected_shortage_per_cycle": {
            "quantity_id": "expected_shortage_per_cycle",
            "name": "Erwartete Fehlmenge pro Bestellzyklus",
            "role": "derived",
            "scope": "per_product",
            "description": (
                "Erwartete ungedeckte Nachfrage während "
                "eines Bestellzyklus."
            ),
        },
        "annual_shortage_quantity": {
            "quantity_id": "annual_shortage_quantity",
            "name": "Erwartete jährliche Fehlmenge",
            "role": "derived",
            "scope": "per_product",
            "description": (
                "Erwartete Fehlmenge pro Zyklus multipliziert "
                "mit den Bestellungen pro Jahr."
            ),
        },
        "safety_stock": {
            "quantity_id": "safety_stock",
            "name": "Sicherheitsbestand",
            "role": "derived",
            "scope": "per_product",
            "description": (
                "Differenz zwischen Meldebestand und erwarteter "
                "Nachfrage während der Lieferzeit."
            ),
        },
    },
}


def calculate_expected_shortage(
    reorder_point: float,
    lead_time_demand_mean: float,
    lead_time_demand_std: float,
) -> float:
    """Berechnet die erwartete Fehlmenge eines Bestellzyklus."""

    if lead_time_demand_std == 0:
        return max(
            lead_time_demand_mean - reorder_point,
            0.0,
        )

    z_score = (
        reorder_point - lead_time_demand_mean
    ) / lead_time_demand_std

    density = math.exp(
        -0.5 * z_score**2
    ) / math.sqrt(2.0 * math.pi)

    tail_probability = (
        1.0 - _STANDARD_NORMAL.cdf(z_score)
    )

    return max(
        lead_time_demand_std
        * (
            density
            - z_score * tail_probability
        ),
        0.0,
    )


def evaluate_q_r_candidate(
    *,
    annual_demand: float,
    ordering_cost: float,
    holding_cost: float,
    shortage_cost: float,
    lead_time_demand_mean: float,
    lead_time_demand_std: float,
    order_quantity: float,
    reorder_point: float,
) -> QRPolicyEvaluation:
    """Bewertet einen konkreten Q,R-Kandidaten."""

    if order_quantity <= 0:
        raise ValueError(
            "Die Bestellmenge Q muss größer als null sein."
        )

    expected_shortage = calculate_expected_shortage(
        reorder_point,
        lead_time_demand_mean,
        lead_time_demand_std,
    )

    orders_per_year = annual_demand / order_quantity
    safety_stock = (
        reorder_point - lead_time_demand_mean
    )
    average_inventory = (
        order_quantity / 2.0 + safety_stock
    )
    annual_shortage_quantity = (
        expected_shortage * orders_per_year
    )

    annual_ordering_cost = (
        ordering_cost * orders_per_year
    )
    annual_holding_cost = (
        holding_cost * average_inventory
    )
    annual_shortage_cost = (
        shortage_cost * annual_shortage_quantity
    )

    return QRPolicyEvaluation(
        order_quantity=order_quantity,
        reorder_point=reorder_point,
        orders_per_year=orders_per_year,
        safety_stock=safety_stock,
        average_inventory=average_inventory,
        expected_shortage_per_cycle=expected_shortage,
        annual_shortage_quantity=annual_shortage_quantity,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        annual_shortage_cost=annual_shortage_cost,
        annual_relevant_cost=(
            annual_ordering_cost
            + annual_holding_cost
            + annual_shortage_cost
        ),
    )


def determine_reorder_point_candidates(
    *,
    annual_demand: float,
    holding_cost: float,
    shortage_cost: float,
    lead_time_demand_mean: float,
    lead_time_demand_std: float,
    order_quantity: int,
    minimum_reorder_point: int = 0,
    maximum_reorder_point: int | None = None,
) -> list[int]:
    """Ermittelt relevante ganzzahlige Meldebestände für ein Q."""

    tail_probability = (
        holding_cost
        * order_quantity
        / (shortage_cost * annual_demand)
    )

    if lead_time_demand_std == 0:
        if tail_probability < 1.0:
            continuous_reorder_point = (
                lead_time_demand_mean
            )
        else:
            continuous_reorder_point = float(
                minimum_reorder_point
            )
    elif tail_probability >= 1.0:
        continuous_reorder_point = float(
            minimum_reorder_point
        )
    else:
        service_probability = 1.0 - tail_probability
        continuous_reorder_point = (
            lead_time_demand_mean
            + lead_time_demand_std
            * _STANDARD_NORMAL.inv_cdf(
                service_probability
            )
        )

    continuous_reorder_point = max(
        continuous_reorder_point,
        minimum_reorder_point,
    )

    if maximum_reorder_point is not None:
        continuous_reorder_point = min(
            continuous_reorder_point,
            maximum_reorder_point,
        )

    candidates = {
        minimum_reorder_point,
        math.floor(continuous_reorder_point),
        math.ceil(continuous_reorder_point),
    }

    if maximum_reorder_point is not None:
        candidates.add(maximum_reorder_point)

    return sorted(
        candidate
        for candidate in candidates
        if candidate >= minimum_reorder_point
        and (
            maximum_reorder_point is None
            or candidate <= maximum_reorder_point
        )
    )


def find_optimal_q_r_policy(
    *,
    annual_demand: float,
    ordering_cost: float,
    holding_cost: float,
    shortage_cost: float,
    lead_time_demand_mean: float,
    lead_time_demand_std: float,
    minimum_order_quantity: int = 1,
    maximum_order_quantity: int | None = None,
    order_multiple: int | None = None,
    minimum_reorder_point: int = 0,
    maximum_reorder_point: int | None = None,
) -> QRPolicyEvaluation:
    """Findet die kostenminimale ganzzahlige Q,R-Politik."""

    if minimum_order_quantity < 1:
        raise ValueError(
            "Die Mindestbestellmenge muss mindestens 1 sein."
        )

    if (
        maximum_order_quantity is not None
        and maximum_order_quantity
        < minimum_order_quantity
    ):
        raise ValueError(
            "Die Mindestbestellmenge darf die "
            "Höchstbestellmenge nicht überschreiten."
        )

    minimum_allowed_order_quantity = (
        minimum_order_quantity
    )

    if order_multiple is not None:
        if order_multiple <= 0:
            raise ValueError(
                "Das Bestellvielfache muss größer als "
                "null sein."
            )

        minimum_allowed_order_quantity = (
            math.ceil(
                minimum_order_quantity / order_multiple
            )
            * order_multiple
        )

        if (
            maximum_order_quantity is not None
            and minimum_allowed_order_quantity
            > maximum_order_quantity
        ):
            raise ValueError(
                "Innerhalb der Bestellmengengrenzen "
                "existiert kein zulässiges "
                "Bestellvielfaches."
            )

    if minimum_reorder_point < 0:
        raise ValueError(
            "Der Mindestmeldebestand darf nicht negativ sein."
        )

    if (
        maximum_reorder_point is not None
        and maximum_reorder_point
        < minimum_reorder_point
    ):
        raise ValueError(
            "Der Mindestmeldebestand darf den "
            "Höchstmeldebestand nicht überschreiten."
        )

    unconstrained_eoq = math.sqrt(
        2.0
        * annual_demand
        * ordering_cost
        / holding_cost
    )

    if order_multiple is None:
        seed_order_quantity = max(
            minimum_order_quantity,
            math.floor(unconstrained_eoq + 0.5),
        )

        if maximum_order_quantity is not None:
            seed_order_quantity = min(
                seed_order_quantity,
                maximum_order_quantity,
            )
    else:
        seed_order_quantity = max(
            minimum_allowed_order_quantity,
            math.floor(
                unconstrained_eoq / order_multiple + 0.5
            )
            * order_multiple,
        )

        if maximum_order_quantity is not None:
            seed_order_quantity = min(
                seed_order_quantity,
                math.floor(
                    maximum_order_quantity
                    / order_multiple
                )
                * order_multiple,
            )

    seed_reorder_points = (
        determine_reorder_point_candidates(
            annual_demand=annual_demand,
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
            lead_time_demand_mean=(
                lead_time_demand_mean
            ),
            lead_time_demand_std=(
                lead_time_demand_std
            ),
            order_quantity=seed_order_quantity,
            minimum_reorder_point=(
                minimum_reorder_point
            ),
            maximum_reorder_point=(
                maximum_reorder_point
            ),
        )
    )

    seed_evaluations = [
        evaluate_q_r_candidate(
            annual_demand=annual_demand,
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
            lead_time_demand_mean=(
                lead_time_demand_mean
            ),
            lead_time_demand_std=(
                lead_time_demand_std
            ),
            order_quantity=seed_order_quantity,
            reorder_point=reorder_point,
        )
        for reorder_point in seed_reorder_points
    ]

    seed_cost = min(
        evaluation.annual_relevant_cost
        for evaluation in seed_evaluations
    )

    if maximum_order_quantity is not None:
        search_maximum = maximum_order_quantity
    else:
        search_maximum = max(
            seed_order_quantity,
            math.ceil(
                2.0
                * (
                    seed_cost / holding_cost
                    - minimum_reorder_point
                    + lead_time_demand_mean
                )
            ),
        )

    best_evaluation: QRPolicyEvaluation | None = None

    for order_quantity in range(
        minimum_order_quantity,
        search_maximum + 1,
    ):
        if (
            order_multiple is not None
            and order_quantity % order_multiple != 0
        ):
            continue

        reorder_points = determine_reorder_point_candidates(
            annual_demand=annual_demand,
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
            lead_time_demand_mean=(
                lead_time_demand_mean
            ),
            lead_time_demand_std=(
                lead_time_demand_std
            ),
            order_quantity=order_quantity,
            minimum_reorder_point=(
                minimum_reorder_point
            ),
            maximum_reorder_point=(
                maximum_reorder_point
            ),
        )

        for reorder_point in reorder_points:
            evaluation = evaluate_q_r_candidate(
                annual_demand=annual_demand,
                ordering_cost=ordering_cost,
                holding_cost=holding_cost,
                shortage_cost=shortage_cost,
                lead_time_demand_mean=(
                    lead_time_demand_mean
                ),
                lead_time_demand_std=(
                    lead_time_demand_std
                ),
                order_quantity=order_quantity,
                reorder_point=reorder_point,
            )

            if (
                best_evaluation is None
                or evaluation.annual_relevant_cost
                < best_evaluation.annual_relevant_cost
            ):
                best_evaluation = evaluation

    if best_evaluation is None:
        raise ValueError(
            "Es konnte keine zulässige Q,R-Politik "
            "ermittelt werden."
        )

    return best_evaluation


def get_q_r_bounds(
    specification: ModelSpecification,
    product_id: str,
) -> tuple[
    int,
    int | None,
    int,
    int | None,
]:
    """Liest die ganzzahligen Q- und R-Grenzen eines Produkts."""

    minimum_order_quantity = 1
    maximum_order_quantity: int | None = None
    minimum_reorder_point = 0
    maximum_reorder_point: int | None = None

    for constraint in specification["constraints"]:
        if constraint.get("product_id") != product_id:
            continue

        component_id = constraint["component_id"]
        parameters = constraint["parameters"]

        if component_id == "minimum_order_quantity":
            minimum_order_quantity = math.ceil(
                float(parameters["quantity"])
            )
        elif component_id == "maximum_order_quantity":
            maximum_order_quantity = math.floor(
                float(parameters["quantity"])
            )
        elif component_id == "minimum_reorder_point":
            minimum_reorder_point = math.ceil(
                float(parameters["reorder_point"])
            )
        elif component_id == "maximum_reorder_point":
            maximum_reorder_point = math.floor(
                float(parameters["reorder_point"])
            )

    return (
        minimum_order_quantity,
        maximum_order_quantity,
        minimum_reorder_point,
        maximum_reorder_point,
    )


def get_order_multiple(
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


def validate_q_r_forecast(
    forecast: DemandForecast,
) -> list[str]:
    """Prüft die für Q,R benötigte Forecast-Struktur."""

    try:
        prepare_continuous_review_forecast(
            forecast,
            lead_time_days=forecast["horizon_days"],
        )
    except ValueError as error:
        return [str(error)]

    return []


def validate_q_r_semantics(
    specification: ModelSpecification,
) -> list[str]:
    """Prüft die fachliche Zusammensetzung des Q,R-Modells."""

    errors: list[str] = []

    if specification["objective"]["sense"] != "minimize":
        errors.append(
            "Die Q,R-Politik muss die erwarteten "
            "jährlichen Kosten minimieren."
        )

    for product in specification["products"]:
        product_id = product["product_id"]
        lead_time_days = int(
            product["facts"]["lead_time_days"]
        )

        try:
            prepare_continuous_review_forecast(
                product["forecast"],
                lead_time_days=lead_time_days,
            )
        except ValueError as error:
            errors.append(
                f"Der Forecast für Produkt '{product_id}' "
                f"kann nicht auf die Lieferzeit übertragen "
                f"werden: {error}"
            )

        (
            minimum_order_quantity,
            maximum_order_quantity,
            minimum_reorder_point,
            maximum_reorder_point,
        ) = get_q_r_bounds(
            specification,
            product_id,
        )

        order_multiple = get_order_multiple(
            specification,
            product_id,
        )

        if (
            maximum_order_quantity is not None
            and minimum_order_quantity
            > maximum_order_quantity
        ):
            errors.append(
                f"Für Produkt '{product_id}' darf die "
                "Mindestbestellmenge die "
                "Höchstbestellmenge nicht überschreiten."
            )

        if (
            maximum_reorder_point is not None
            and minimum_reorder_point
            > maximum_reorder_point
        ):
            errors.append(
                f"Für Produkt '{product_id}' darf der "
                "Mindestmeldebestand den "
                "Höchstmeldebestand nicht überschreiten."
            )

        if (
            maximum_order_quantity is not None
            and order_multiple is not None
        ):
            smallest_allowed_multiple = (
                math.ceil(
                    minimum_order_quantity
                    / order_multiple
                )
                * order_multiple
            )

            if (
                smallest_allowed_multiple
                > maximum_order_quantity
            ):
                errors.append(
                    f"Für Produkt '{product_id}' existiert "
                    "innerhalb der Bestellmengengrenzen "
                    "kein zulässiges Bestellvielfaches."
                )

    return errors


def execute_continuous_review_q_r(
    specification: ModelSpecification,
) -> OptimizationExecutionResult:
    """Validiert und löst eine modulare Q,R-Spezifikation."""

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
            "Die Q,R-Spezifikation ist nicht vollständig: "
            + "; ".join(messages)
        )

    decisions: list[DecisionValue] = []
    metrics: list[MetricValue] = []
    representations: list[str] = []
    total_annual_cost = 0.0

    for product in specification["products"]:
        product_id = product["product_id"]
        facts = product["facts"]

        prepared_forecast = (
            prepare_continuous_review_forecast(
                product["forecast"],
                lead_time_days=int(
                    facts["lead_time_days"]
                ),
            )
        )

        (
            minimum_order_quantity,
            maximum_order_quantity,
            minimum_reorder_point,
            maximum_reorder_point,
        ) = get_q_r_bounds(
            specification,
            product_id,
        )

        order_multiple = get_order_multiple(
            specification,
            product_id,
        )

        result = find_optimal_q_r_policy(
            annual_demand=prepared_forecast[
                "annual_demand"
            ],
            ordering_cost=float(facts["ordering_cost"]),
            holding_cost=float(facts["holding_cost"]),
            shortage_cost=float(facts["shortage_cost"]),
            lead_time_demand_mean=prepared_forecast[
                "lead_time_demand_mean"
            ],
            lead_time_demand_std=prepared_forecast[
                "lead_time_demand_std"
            ],
            minimum_order_quantity=(
                minimum_order_quantity
            ),
            maximum_order_quantity=(
                maximum_order_quantity
            ),
            order_multiple=order_multiple,
            minimum_reorder_point=(
                minimum_reorder_point
            ),
            maximum_reorder_point=(
                maximum_reorder_point
            ),
        )

        total_annual_cost += result.annual_relevant_cost

        decisions.extend(
            [
                {
                    "quantity_id": "order_quantity",
                    "product_id": product_id,
                    "value": result.order_quantity,
                },
                {
                    "quantity_id": "reorder_point",
                    "product_id": product_id,
                    "value": result.reorder_point,
                },
            ]
        )

        product_metrics = {
            "annual_demand": prepared_forecast[
                "annual_demand"
            ],
            "lead_time_demand_mean": prepared_forecast[
                "lead_time_demand_mean"
            ],
            "lead_time_demand_std": prepared_forecast[
                "lead_time_demand_std"
            ],
            "orders_per_year": result.orders_per_year,
            "safety_stock": result.safety_stock,
            "average_inventory": result.average_inventory,
            "expected_shortage_per_cycle": (
                result.expected_shortage_per_cycle
            ),
            "annual_shortage_quantity": (
                result.annual_shortage_quantity
            ),
            "annual_ordering_cost": (
                result.annual_ordering_cost
            ),
            "annual_holding_cost": (
                result.annual_holding_cost
            ),
            "annual_shortage_cost": (
                result.annual_shortage_cost
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
                    (
                        "G(Q,R) = K*D/Q "
                        "+ h*(Q/2 + R - μ_L) "
                        "+ p*D*n(R)/Q"
                    ),
                    (
                        "Jahresbedarf D = "
                        f"{prepared_forecast['annual_demand']}"
                    ),
                    (
                        "Lieferzeitnachfrage μ_L = "
                        f"{prepared_forecast['lead_time_demand_mean']}"
                    ),
                    (
                        "Standardabweichung σ_L = "
                        f"{prepared_forecast['lead_time_demand_std']}"
                    ),
                    f"Bestellmenge Q = {result.order_quantity}",
                    f"Meldebestand R = {result.reorder_point}",
                    (
                        "Jährliche relevante Kosten = "
                        f"{result.annual_relevant_cost}"
                    ),
                ]
            )
        )

    return {
        "status": "optimal",
        "solution_method": "numerical",
        "objective_value": total_annual_cost,
        "decisions": decisions,
        "metrics": metrics,
        "technical_representation": "\n\n".join(
            representations
        ),
        "errors": [],
    }