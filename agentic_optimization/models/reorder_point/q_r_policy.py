"""Deterministische Q-R-Politik für kontinuierliche Bestandsüberwachung.

Das Modell verwendet die klassischen iterativen Optimalitätsbedingungen für
normalverteilte Nachfrage während der Lieferzeit. Es kommt ohne linearen
Solver aus und trennt Validierung, Spezifikation, Lösung und Darstellung.
"""

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

from agentic_optimization.models.types import (
    ConstraintDefinition,
    ModelInputField,
    ModelVariantDefinition,
    ResultMetric,
    StructuredConstraint,
    VariantResultPresentation,
)

REQUIRED_FIELDS = [
    "annual_demand",
    "fixed_order_cost",
    "holding_cost",
    "penalty_cost",
    "lead_time_demand_mean",
    "lead_time_demand_std",
]


INPUT_FIELDS: list[ModelInputField] = [
    {
        "name": "annual_demand",
        "label": "Jährliche Nachfrage",
        "type": "number",
        "required": True,
        "help": "Geschätzte Nachfrage pro Jahr (λ).",
        "placeholder": None,
        "min_value": 0.01,
        "step": 1.0,
        "unit": "Einheiten pro Jahr",
    },
    {
        "name": "fixed_order_cost",
        "label": "Fixe Bestellkosten",
        "type": "number",
        "required": True,
        "help": "Fixe Rüst- oder Bestellkosten je Bestellung (K).",
        "placeholder": None,
        "min_value": 0.0,
        "step": 1.0,
        "unit": "€ pro Bestellung",
    },
    {
        "name": "holding_cost",
        "label": "Jährliche Lagerhaltungskosten",
        "type": "number",
        "required": True,
        "help": "Lagerhaltungskosten je Einheit und Jahr (h).",
        "placeholder": None,
        "min_value": 0.01,
        "step": 0.1,
        "unit": "€ pro Einheit und Jahr",
    },
    {
        "name": "penalty_cost",
        "label": "Fehlmengenkosten",
        "type": "number",
        "required": True,
        "help": "Kosten je rückständiger Einheit (p).",
        "placeholder": None,
        "min_value": 0.01,
        "step": 0.1,
        "unit": "€ pro Einheit",
    },
    {
        "name": "lead_time_demand_mean",
        "label": "Mittlere Nachfrage während der Lieferzeit",
        "type": "number",
        "required": True,
        "help": "Erwartete Nachfrage während der Lieferzeit (μ).",
        "placeholder": None,
        "min_value": 0.0,
        "step": 1.0,
        "unit": "Einheiten",
    },
    {
        "name": "lead_time_demand_std",
        "label": "Standardabweichung der Lieferzeitnachfrage",
        "type": "number",
        "required": True,
        "help": "Standardabweichung der Nachfrage während der Lieferzeit (σ).",
        "placeholder": None,
        "min_value": 0.0,
        "step": 0.1,
        "unit": "Einheiten",
    },
]


def _constraint_definition(
    constraint_type: str,
    name: str,
    parameter_name: str,
    label: str,
    help_text: str,
) -> ConstraintDefinition:
    permits_zero = "reorder_point" in constraint_type

    return {
        "type": constraint_type,
        "name": name,
        "parameters": [
            {
                "name": parameter_name,
                "label": label,
                "type": "integer",
                "required": True,
                "help": help_text,
                "placeholder": None,
                "min_value": 0 if permits_zero else 1,
                "step": 1,
                "unit": "Einheiten",
            }
        ],
    }


SUPPORTED_CONSTRAINTS: dict[str, ConstraintDefinition] = {
    "minimum_order_quantity": _constraint_definition(
        "minimum_order_quantity",
        "Mindestbestellmenge",
        "quantity",
        "Mindestbestellmenge",
        "Untere Grenze für die Bestellmenge Q.",
    ),
    "maximum_order_quantity": _constraint_definition(
        "maximum_order_quantity",
        "Höchstbestellmenge",
        "quantity",
        "Höchstbestellmenge",
        "Obere Grenze für die Bestellmenge Q.",
    ),
    "minimum_reorder_point": _constraint_definition(
        "minimum_reorder_point",
        "Mindestmeldebestand",
        "reorder_point",
        "Mindestmeldebestand",
        "Untere Grenze für den Meldebestand R.",
    ),
    "maximum_reorder_point": _constraint_definition(
        "maximum_reorder_point",
        "Höchstmeldebestand",
        "reorder_point",
        "Höchstmeldebestand",
        "Obere Grenze für den Meldebestand R.",
    ),
    "storage_capacity": _constraint_definition(
        "storage_capacity",
        "Lagerkapazität (konservative Grenze)",
        "capacity",
        "Maximale Bestandsposition Q + R",
        (
            "Konservative Kapazitätsgrenze. Das Modell verlangt, "
            "dass die Summe aus Bestellmenge Q und Meldebestand R "
            "diesen Wert nicht überschreitet."
        ),
    ),
}


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_q_r_policy_inputs(inputs: dict[str, Any]) -> list[str]:
    """Prüft die Eingaben der Q-R-Politik auf Typ und Wertebereich."""

    rules = {
        "annual_demand": (
            "Die jährliche Nachfrage",
            lambda value: value > 0,
            "Die jährliche Nachfrage muss größer als 0 sein.",
        ),
        "fixed_order_cost": (
            "Die fixen Bestellkosten",
            lambda value: value >= 0,
            "Die fixen Bestellkosten dürfen nicht negativ sein.",
        ),
        "holding_cost": (
            "Die Lagerhaltungskosten",
            lambda value: value > 0,
            "Die Lagerhaltungskosten müssen größer als 0 sein.",
        ),
        "penalty_cost": (
            "Die Fehlmengenkosten",
            lambda value: value > 0,
            "Die Fehlmengenkosten müssen größer als 0 sein.",
        ),
        "lead_time_demand_mean": (
            "Die mittlere Lieferzeitnachfrage",
            lambda value: value >= 0,
            "Die mittlere Lieferzeitnachfrage darf nicht negativ sein.",
        ),
        "lead_time_demand_std": (
            "Die Standardabweichung der Lieferzeitnachfrage",
            lambda value: value >= 0,
            "Die Standardabweichung der Lieferzeitnachfrage darf nicht negativ sein.",
        ),
    }
    errors: list[str] = []

    for field_name, (label, range_check, range_error) in rules.items():
        value = inputs.get(field_name)
        if not _is_finite_number(value):
            errors.append(f"{label} muss eine endliche Zahl sein.")
        elif not range_check(value):
            errors.append(range_error)

    max_iterations = inputs.get("max_iterations", 100)
    if (
        not _is_finite_number(max_iterations)
        or not float(max_iterations).is_integer()
        or max_iterations <= 0
    ):
        errors.append("Die maximale Iterationszahl muss eine positive ganze Zahl sein.")

    tolerance = inputs.get("tolerance", 1e-6)
    if not _is_finite_number(tolerance) or tolerance <= 0:
        errors.append("Die Toleranz muss eine endliche Zahl größer als 0 sein.")

    return errors


def validate_q_r_policy_constraints(
    inputs: dict[str, Any],
    constraints: list[StructuredConstraint],
) -> list[str]:
    """Prüft Wertebereiche und Beziehungen der Q-R-Constraints."""

    del inputs
    errors: list[str] = []
    values: dict[str, float] = {}
    parameter_names = {
        "minimum_order_quantity": "quantity",
        "maximum_order_quantity": "quantity",
        "minimum_reorder_point": "reorder_point",
        "maximum_reorder_point": "reorder_point",
        "storage_capacity": "capacity",
    }
    labels = {
        "minimum_order_quantity": "Die Mindestbestellmenge",
        "maximum_order_quantity": "Die Höchstbestellmenge",
        "minimum_reorder_point": "Der Mindestmeldebestand",
        "maximum_reorder_point": "Der Höchstmeldebestand",
        "storage_capacity": "Die Lagerkapazität",
    }

    for constraint in constraints:
        constraint_type = constraint["type"]
        parameter_name = parameter_names.get(constraint_type)
        if parameter_name is None:
            continue
        value = constraint["parameters"].get(parameter_name)
        if value is None:
            continue
        if not _is_finite_number(value):
            errors.append(f"{labels[constraint_type]} muss eine endliche Zahl sein.")
            continue
        if not float(value).is_integer():
            errors.append(
                f"{labels[constraint_type]} muss in ganzen Einheiten angegeben werden."
            )
            continue
        values[constraint_type] = float(value)
        minimum = 0 if "reorder_point" in constraint_type else 0
        permits_zero = "reorder_point" in constraint_type
        if value < minimum or (value == 0 and not permits_zero):
            comparator = "darf nicht negativ sein" if permits_zero else "muss größer als 0 sein"
            errors.append(f"{labels[constraint_type]} {comparator}.")

    minimum_q = values.get("minimum_order_quantity")
    maximum_q = values.get("maximum_order_quantity")
    if minimum_q is not None and maximum_q is not None and minimum_q > maximum_q:
        errors.append(
            "Die Mindestbestellmenge darf die Höchstbestellmenge nicht überschreiten."
        )

    minimum_r = values.get("minimum_reorder_point")
    maximum_r = values.get("maximum_reorder_point")
    if minimum_r is not None and maximum_r is not None and minimum_r > maximum_r:
        errors.append(
            "Der Mindestmeldebestand darf den Höchstmeldebestand nicht überschreiten."
        )

    return errors


@dataclass(frozen=True)
class QRPolicySpecification:
    """Normalisierte, unveränderliche Eingabe für den Q-R-Solver."""

    annual_demand: float
    fixed_order_cost: float
    holding_cost: float
    penalty_cost: float
    lead_time_demand_mean: float
    lead_time_demand_std: float
    minimum_order_quantity: int | None = None
    maximum_order_quantity: int | None = None
    minimum_reorder_point: int | None = None
    maximum_reorder_point: int | None = None
    storage_capacity: int | None = None
    max_iterations: int = 100
    tolerance: float = 1e-6


@dataclass(frozen=True)
class QRPolicyResult:
    """Ergebnis der deterministischen Q-R-Iteration."""

    solver_status: str
    order_quantity: int | None
    reorder_point: int | None
    safety_stock: int | None
    expected_annual_cost: float | None
    iterations: int
    converged: bool
    model_representation: str


def create_q_r_policy_specification(
    inputs: dict[str, Any],
    constraints: list[StructuredConstraint],
) -> QRPolicySpecification:
    """Erstellt eine normalisierte Q-R-Spezifikation."""

    constraints_by_type: dict[str, StructuredConstraint] = {}
    for constraint in constraints:
        constraint_type = constraint["type"]
        if constraint_type in constraints_by_type:
            raise ValueError(
                f"Der Constraint-Typ '{constraint_type}' wurde mehrfach angegeben."
            )
        if constraint_type not in SUPPORTED_CONSTRAINTS:
            raise ValueError(
                f"Der Constraint-Typ '{constraint_type}' wird von dieser "
                "Modellvariante nicht unterstützt."
            )
        constraints_by_type[constraint_type] = constraint

    def optional_parameter(constraint_type: str, parameter_name: str) -> int | None:
        constraint = constraints_by_type.get(constraint_type)
        if constraint is None:
            return None
        parameters = constraint["parameters"]
        if parameter_name not in parameters:
            raise ValueError(
                f"Für den Constraint '{constraint_type}' fehlt der Parameter "
                f"'{parameter_name}'."
            )
        value = parameters[parameter_name]
        if not _is_finite_number(value):
            raise TypeError(
                f"Der Parameter '{parameter_name}' des Constraints "
                f"'{constraint_type}' muss eine endliche Zahl sein."
            )
        if not float(value).is_integer():
            raise ValueError(
                f"Der Parameter '{parameter_name}' des Constraints "
                f"'{constraint_type}' muss in ganzen Einheiten angegeben werden."
            )
        return int(value)

    return QRPolicySpecification(
        annual_demand=float(inputs["annual_demand"]),
        fixed_order_cost=float(inputs["fixed_order_cost"]),
        holding_cost=float(inputs["holding_cost"]),
        penalty_cost=float(inputs["penalty_cost"]),
        lead_time_demand_mean=float(inputs["lead_time_demand_mean"]),
        lead_time_demand_std=float(inputs["lead_time_demand_std"]),
        minimum_order_quantity=optional_parameter(
            "minimum_order_quantity", "quantity"
        ),
        maximum_order_quantity=optional_parameter(
            "maximum_order_quantity", "quantity"
        ),
        minimum_reorder_point=optional_parameter(
            "minimum_reorder_point", "reorder_point"
        ),
        maximum_reorder_point=optional_parameter(
            "maximum_reorder_point", "reorder_point"
        ),
        storage_capacity=optional_parameter("storage_capacity", "capacity"),
        max_iterations=int(inputs.get("max_iterations", 100)),
        tolerance=float(inputs.get("tolerance", 1e-6)),
    )


_STANDARD_NORMAL = NormalDist()


def _expected_shortage(reorder_point: float, mean: float, std: float) -> float:
    if std == 0:
        return max(mean - reorder_point, 0.0)
    z_score = (reorder_point - mean) / std
    density = math.exp(-0.5 * z_score**2) / math.sqrt(2 * math.pi)
    tail_probability = 1.0 - _STANDARD_NORMAL.cdf(z_score)
    return max(std * (density - z_score * tail_probability), 0.0)


def _apply_bounds(
    value: float,
    minimum: float | None,
    maximum: float | None,
) -> float:
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def _round_order_quantity(
    value: float,
    minimum: int | None,
    maximum: int | None,
) -> int:
    """Rundet Q kaufmännisch auf eine zulässige ganze Einheit."""

    bounded_value = _apply_bounds(value, minimum, maximum)
    rounded_value = max(1, math.floor(bounded_value + 0.5))

    if minimum is not None:
        rounded_value = max(rounded_value, minimum)
    if maximum is not None:
        rounded_value = min(rounded_value, maximum)

    return rounded_value


def _round_reorder_point(
    value: float,
    minimum: int | None,
    maximum: int | None,
) -> int:
    """Rundet R zur Absicherung der Nachfrage auf eine ganze Einheit auf."""

    bounded_value = _apply_bounds(value, minimum, maximum)
    rounded_value = max(0, math.ceil(bounded_value))

    if minimum is not None:
        rounded_value = max(rounded_value, minimum)
    if maximum is not None:
        rounded_value = min(rounded_value, maximum)

    return rounded_value


def _model_representation(specification: QRPolicySpecification) -> str:
    return (
        "Q-R-Politik (kontinuierliche Überprüfung)\n"
        "Annahme: Lieferzeitnachfrage X_L ~ Normal(μ, σ²).\n"
        "Start: Q₀ = sqrt(2 λ K / h).\n"
        "Iteration: P(X_L ≤ R) = 1 - Q/(p λ), "
        "R = μ + σ Φ⁻¹(P),\n"
        "n(R) = σ[φ(z) - z(1 - Φ(z))],\n"
        "Q_neu = sqrt(2 λ (K + p n(R)) / h).\n"
        "Kosten: G = h(Q/2 + R - μ) + Kλ/Q + pλn(R)/Q.\n"
        "Operative Rundung: Q auf die nächste ganze Einheit, "
        "R und der Sicherheitsbestand werden aufgerundet.\n"
        "Numerik: Servicewahrscheinlichkeit wird auf [1e-9, 1-1e-9] begrenzt.\n"
        f"Grenzen: min_Q={specification.minimum_order_quantity}, "
        f"max_Q={specification.maximum_order_quantity}, "
        f"min_R={specification.minimum_reorder_point}, "
        f"max_R={specification.maximum_reorder_point}, "
        f"Kapazität(Q+R)={specification.storage_capacity}."
    )


def solve_q_r_policy(specification: QRPolicySpecification) -> QRPolicyResult:
    """Löst die klassischen Q-R-Optimalitätsbedingungen iterativ."""

    representation = _model_representation(specification)
    order_quantity = math.sqrt(
        2
        * specification.annual_demand
        * specification.fixed_order_cost
        / specification.holding_cost
    )
    order_quantity = _round_order_quantity(
        order_quantity,
        specification.minimum_order_quantity,
        specification.maximum_order_quantity,
    )
    previous_reorder_point: int | None = None
    reorder_point = _round_reorder_point(
        specification.lead_time_demand_mean,
        specification.minimum_reorder_point,
        specification.maximum_reorder_point,
    )
    iterations = 0
    converged = False

    for iterations in range(1, specification.max_iterations + 1):
        service_probability = 1.0 - order_quantity / (
            specification.penalty_cost * specification.annual_demand
        )
        service_probability = min(max(service_probability, 1e-9), 1.0 - 1e-9)
        z_score = _STANDARD_NORMAL.inv_cdf(service_probability)
        reorder_point = (
            specification.lead_time_demand_mean
            + z_score * specification.lead_time_demand_std
        )
        reorder_point = _round_reorder_point(
            reorder_point,
            specification.minimum_reorder_point,
            specification.maximum_reorder_point,
        )
        shortage = _expected_shortage(
            reorder_point,
            specification.lead_time_demand_mean,
            specification.lead_time_demand_std,
        )
        new_order_quantity = math.sqrt(
            2
            * specification.annual_demand
            * (specification.fixed_order_cost + specification.penalty_cost * shortage)
            / specification.holding_cost
        )
        new_order_quantity = _round_order_quantity(
            new_order_quantity,
            specification.minimum_order_quantity,
            specification.maximum_order_quantity,
        )

        if (
            specification.storage_capacity is not None
            and new_order_quantity + reorder_point > specification.storage_capacity
        ):
            return QRPolicyResult(
                solver_status="Infeasible",
                order_quantity=None,
                reorder_point=None,
                safety_stock=None,
                expected_annual_cost=None,
                iterations=iterations,
                converged=False,
                model_representation=(
                    representation
                    + "\nNicht erfüllbar: Die berechnete Summe Q + R "
                    "überschreitet die Lagerkapazität."
                ),
            )

        q_change = abs(new_order_quantity - order_quantity)
        r_change = (
            math.inf
            if previous_reorder_point is None
            else abs(reorder_point - previous_reorder_point)
        )
        order_quantity = new_order_quantity
        if q_change < specification.tolerance and r_change < specification.tolerance:
            converged = True
            break
        previous_reorder_point = reorder_point

    shortage = _expected_shortage(
        reorder_point,
        specification.lead_time_demand_mean,
        specification.lead_time_demand_std,
    )
    expected_annual_cost = (
        specification.holding_cost
        * (
            order_quantity / 2
            + reorder_point
            - specification.lead_time_demand_mean
        )
        + specification.fixed_order_cost
        * specification.annual_demand
        / order_quantity
        + specification.penalty_cost
        * specification.annual_demand
        * shortage
        / order_quantity
    )

    return QRPolicyResult(
        solver_status="Optimal" if converged else "IterationLimit",
        order_quantity=order_quantity,
        reorder_point=reorder_point,
        safety_stock=math.ceil(
            reorder_point - specification.lead_time_demand_mean
        ),
        expected_annual_cost=expected_annual_cost,
        iterations=iterations,
        converged=converged,
        model_representation=representation,
    )


def optimize_q_r_policy(
    inputs: dict[str, Any],
    constraints: list[StructuredConstraint],
) -> dict[str, Any]:
    """Erstellt die Spezifikation, löst sie und serialisiert das Ergebnis."""

    specification = create_q_r_policy_specification(inputs, constraints)
    return asdict(solve_q_r_policy(specification))


def _format_result_number(value: float | None) -> str:
    if value is None:
        return "–"
    numeric_value = float(value)
    return str(int(numeric_value)) if numeric_value.is_integer() else f"{numeric_value:g}"


def _format_result_currency(value: float | None) -> str:
    if value is None:
        return "–"
    formatted = f"{float(value):,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{formatted} €"


def build_q_r_policy_result_presentation(
    optimization_result: dict[str, Any],
) -> VariantResultPresentation:
    """Formatiert die zentralen Q-R-Kennzahlen für die Ergebnis-UI."""

    metrics: list[ResultMetric] = [
        {
            "label": "Empfohlene Bestellmenge Q",
            "value": (
                f"{_format_result_number(optimization_result.get('order_quantity'))} "
                "Einheiten"
            ),
        },
        {
            "label": "Meldebestand R",
            "value": (
                f"{_format_result_number(optimization_result.get('reorder_point'))} "
                "Einheiten"
            ),
        },
        {
            "label": "Sicherheitsbestand",
            "value": (
                f"{_format_result_number(optimization_result.get('safety_stock'))} "
                "Einheiten"
            ),
        },
        {
            "label": "Erwartete jährliche Kosten",
            "value": _format_result_currency(
                optimization_result.get("expected_annual_cost")
            ),
        },
    ]
    technical_metrics: list[ResultMetric] = [
        {
            "label": "Iterationen",
            "value": _format_result_number(
                optimization_result.get("iterations")
            ),
        },
    ]
    return {
        "metrics": metrics,
        "technical_metrics": technical_metrics,
        "detail_title": None,
        "detail_rows": [],
    }


VARIANT_DEFINITION: ModelVariantDefinition = {
    "variant_id": "q_r_policy",
    "variant_name": "Q-R Policy",
    "selection_criteria": {
        "decision": "order_quantity",
        "objective": "minimize_cost",
        "demand_type": "uncertain",
        "product_scope": "single_product",
        "planning_horizon": "multiple_periods",
        "review_type": "continuous_review",
    },
    "required_fields": REQUIRED_FIELDS,
    "input_fields": INPUT_FIELDS,
    "supported_constraints": SUPPORTED_CONSTRAINTS,
    "input_validator": validate_q_r_policy_inputs,
    "constraint_validator": validate_q_r_policy_constraints,
    "optimizer": optimize_q_r_policy,
    "result_builder": build_q_r_policy_result_presentation,
}
