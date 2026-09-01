"""
Implementiert die Newsvendor-Variante für ein Produkt und eine Periode.

Die Datei enthält Registry-Metadaten, Validierung, Modellspezifikation,
deterministischen PuLP-Aufbau, Solverausführung und Ergebnisdarstellung.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any

import pulp

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
        "name": "selling_price",
        "label": "Verkaufspreis pro Einheit",
        "type": "number",
        "required": True,
        "help": (
            "Preis, zu dem eine Einheit verkauft wird."
        ),
        "placeholder": None,
        "min_value": 0.01,
        "step": 0.5,
        "unit": "€ pro Einheit",
    },
    {
        "name": "purchase_cost",
        "label": "Einkaufskosten pro Einheit",
        "type": "number",
        "required": True,
        "help": (
            "Kosten für die Beschaffung einer Einheit."
        ),
        "placeholder": None,
        "min_value": 0.01,
        "step": 0.5,
        "unit": "€ pro Einheit",
    },
    {
        "name": "salvage_value",
        "label": "Restwert pro unverkaufter Einheit",
        "type": "number",
        "required": True,
        "help": (
            "Wert einer Einheit, die nach der Periode "
            "nicht verkauft wurde. Der Restwert muss "
            "kleiner als die Einkaufskosten sein."
        ),
        "placeholder": None,
        "min_value": 0.0,
        "step": 0.5,
        "unit": "€ pro Einheit",
    },
    {
        "name": "demand_scenarios",
        "label": "Mögliche Nachfragewerte",
        "type": "number_list",
        "required": True,
        "help": (
            "Gib die möglichen Nachfragewerte durch Kommas "
            "getrennt ein."
        ),
        "placeholder": "z. B. 80, 100, 120, 140",
        "min_value": 0.0,
        "unit": "Einheiten",
    },
    {
        "name": "scenario_probabilities",
        "label": "Wahrscheinlichkeiten der Nachfragewerte",
        "type": "number_list",
        "required": True,
        "help": (
            "Gib zu jedem Nachfragewert eine Wahrscheinlichkeit "
            "zwischen 0 und 1 an. Die Summe muss 1 ergeben."
        ),
        "placeholder": "z. B. 0.2, 0.5, 0.2, 0.1",
        "min_value": 0.0,
        "max_value": 1.0,
        "unit": None,
    },
]


SUPPORTED_CONSTRAINTS: dict[str, ConstraintDefinition] = {
    "minimum_order_quantity": {
        "type": "minimum_order_quantity",
        "name": "Mindestbestellmenge",
        "parameters": [
            {
                "name": "quantity",
                "label": "Mindestbestellmenge",
                "type": "number",
                "required": True,
                "help": "Die mindestens zu bestellende Menge.",
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
                "help": "Die maximal zulässige Bestellmenge.",
                "placeholder": None,
                "min_value": 0.0,
                "step": 1.0,
                "unit": "Einheiten",
            }
        ],
    },
    "budget_limit": {
        "type": "budget_limit",
        "name": "Budgetgrenze",
        "parameters": [
            {
                "name": "budget",
                "label": "Verfügbares Budget",
                "type": "number",
                "required": True,
                "help": "Maximales Budget für die Bestellung.",
                "placeholder": None,
                "min_value": 0.0,
                "step": 100.0,
                "unit": "€",
            }
        ],
    },
    "storage_capacity": {
        "type": "storage_capacity",
        "name": "Lagerkapazität",
        "parameters": [
            {
                "name": "capacity",
                "label": "Lagerkapazität",
                "type": "number",
                "required": True,
                "help": "Maximale Anzahl von Produkteinheiten, die gelagert werden können.",
                "placeholder": None,
                "min_value": 0.0,
                "step": 1.0,
                "unit": "Einheiten",
            }
        ],
    },
}


def validate_single_product_single_period_inputs(
    inputs: dict[str, Any],
) -> list[str]:
    """
    Prüft Preise, Nachfrageszenarien und Szenariowahrscheinlichkeiten.

    Zusätzlich werden die für diese Newsvendor-Variante erforderlichen
    Beziehungen zwischen Verkaufspreis, Einkaufskosten und Restwert geprüft.
    """

    errors: list[str] = []

    selling_price = inputs.get("selling_price")
    purchase_cost = inputs.get("purchase_cost")
    salvage_value = inputs.get("salvage_value")
    demand_scenarios = inputs.get("demand_scenarios") or []
    scenario_probabilities = inputs.get("scenario_probabilities") or []

    if not isinstance(demand_scenarios, list):
        errors.append(
            "Nachfrageszenarien müssen als Liste angegeben werden."
        )
        demand_scenarios = []

    if not isinstance(scenario_probabilities, list):
        errors.append(
            "Wahrscheinlichkeiten müssen als Liste angegeben werden."
        )
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
            errors.append(
                f"{numeric_labels[field_name]} muss eine endliche Zahl sein."
            )

    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in demand_scenarios
    ):
        errors.append(
            "Nachfrageszenarien müssen endliche Zahlen sein."
        )

    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in scenario_probabilities
    ):
        errors.append(
            "Wahrscheinlichkeiten müssen endliche Zahlen sein."
        )

    if errors:
        return errors

    if selling_price is None or selling_price <= 0:
        errors.append(
            "Der Verkaufspreis muss größer als 0 sein."
        )

    if purchase_cost is None or purchase_cost <= 0:
        errors.append(
            "Die Einkaufskosten müssen größer als 0 sein."
        )

    if (
        selling_price is not None
        and purchase_cost is not None
        and selling_price <= purchase_cost
    ):
        errors.append(
            "Der Verkaufspreis muss größer als die Einkaufskosten sein."
        )

    if salvage_value is None:
        errors.append(
            "Der Restwert muss angegeben werden."
        )
    elif salvage_value < 0:
        errors.append(
            "Der Restwert darf nicht negativ sein."
        )

    if (
        salvage_value is not None
        and purchase_cost is not None
        and salvage_value >= purchase_cost
    ):
        errors.append(
            "Der Restwert muss kleiner als die "
            "Einkaufskosten sein."
        )

    if not demand_scenarios:
        errors.append(
            "Bitte geben Sie mindestens ein Nachfrageszenario ein."
        )

    if not scenario_probabilities:
        errors.append(
            "Bitte geben Sie mindestens eine Wahrscheinlichkeit ein."
        )

    if (
        demand_scenarios
        and scenario_probabilities
        and len(demand_scenarios) != len(scenario_probabilities)
    ):
        errors.append(
            "Die Anzahl der Nachfrageszenarien muss zur Anzahl "
            "der Wahrscheinlichkeiten passen."
        )

    if any(demand < 0 for demand in demand_scenarios):
        errors.append(
            "Nachfrageszenarien dürfen nicht negativ sein."
        )

    if any(
        probability < 0 or probability > 1
        for probability in scenario_probabilities
    ):
        errors.append(
            "Wahrscheinlichkeiten müssen zwischen 0 und 1 liegen."
        )

    if (
        scenario_probabilities
        and abs(sum(scenario_probabilities) - 1.0) > 0.001
    ):
        errors.append(
            "Die Wahrscheinlichkeiten müssen zusammen 1 ergeben."
        )

    return errors


def validate_single_product_single_period_constraints(
    inputs: dict[str, Any],
    constraints: list[StructuredConstraint],
) -> list[str]:
    """
    Prüft Wertebereiche und Beziehungen der unterstützten Constraints.

    Die generische Prüfung von Typen und Pflichtparametern erfolgt bereits
    im Validation Agent.
    """

    errors: list[str] = []

    minimum_quantity: float | None = None
    maximum_quantity: float | None = None

    for constraint in constraints:
        constraint_type = constraint["type"]
        parameters = constraint["parameters"]

        if constraint_type == "minimum_order_quantity":
            quantity = parameters.get("quantity")

            if quantity is None:
                continue

            if quantity <= 0:
                errors.append(
                    "Die Mindestbestellmenge muss größer als 0 sein."
                )
            else:
                minimum_quantity = float(quantity)

        elif constraint_type == "maximum_order_quantity":
            quantity = parameters.get("quantity")

            if quantity is None:
                continue

            if quantity < 0:
                errors.append(
                    "Die Höchstbestellmenge darf nicht negativ sein."
                )
            else:
                maximum_quantity = float(quantity)

        elif constraint_type == "budget_limit":
            budget = parameters.get("budget")

            if budget is not None and budget < 0:
                errors.append(
                    "Das Budget darf nicht negativ sein."
                )

        elif constraint_type == "storage_capacity":
            capacity = parameters.get("capacity")

            if capacity is not None and capacity < 0:
                errors.append(
                    "Die Lagerkapazität darf nicht negativ sein."
                )

    if (
        minimum_quantity is not None
        and maximum_quantity is not None
        and minimum_quantity > maximum_quantity
    ):
        errors.append(
            "Die Mindestbestellmenge darf die Höchstbestellmenge "
            "nicht überschreiten."
        )

    return errors


@dataclass(frozen=True)
class SingleProductSinglePeriodSpecification:
    """Normalisierte unveränderliche Eingabe für den PuLP-Builder."""
    selling_price: float
    purchase_cost: float
    salvage_value: float
    demand_scenarios: tuple[float, ...]
    scenario_probabilities: tuple[float, ...]

    minimum_order_quantity: float | None = None
    maximum_order_quantity: float | None = None
    budget_limit: float | None = None
    storage_capacity: float | None = None


@dataclass
class BuiltSingleProductSinglePeriodModel:
    """Bündelt das PuLP-Modell und die Variablen zur Ergebnisauswertung."""
    model: pulp.LpProblem
    order_quantity: pulp.LpVariable
    sold: dict[int, pulp.LpVariable]
    leftover: dict[int, pulp.LpVariable]

@dataclass(frozen=True)
class ScenarioResult:
    demand: float
    probability: float
    sold: float | None
    leftover: float | None

@dataclass(frozen=True)
class SingleProductSinglePeriodResult:
    solver_status: str
    order_quantity: float | None
    expected_profit: float | None
    scenario_results: tuple[ScenarioResult, ...]
    model_representation: str

def create_single_product_single_period_specification(
    inputs: dict[str, Any],
    constraints: list[StructuredConstraint],
) -> SingleProductSinglePeriodSpecification:
    """
    Erstellt aus bereits validierten Inputs und Constraints eine
    normalisierte, unveränderliche Modellspezifikation.
    """

    constraints_by_type: dict[str, StructuredConstraint] = {}

    for constraint in constraints:
        constraint_type = constraint["type"]

        if constraint_type in constraints_by_type:
            raise ValueError(
                f"Der Constraint-Typ '{constraint_type}' "
                "wurde mehrfach angegeben."
            )

        if constraint_type not in SUPPORTED_CONSTRAINTS:
            raise ValueError(
                f"Der Constraint-Typ '{constraint_type}' "
                "wird von dieser Modellvariante nicht unterstützt."
            )

        constraints_by_type[constraint_type] = constraint

    def get_optional_parameter(
        constraint_type: str,
        parameter_name: str,
    ) -> float | None:
        constraint = constraints_by_type.get(constraint_type)

        if constraint is None:
            return None

        parameters = constraint["parameters"]

        if parameter_name not in parameters:
            raise ValueError(
                f"Für den Constraint '{constraint_type}' fehlt "
                f"der Parameter '{parameter_name}'."
            )

        value = parameters[parameter_name]

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"Der Parameter '{parameter_name}' des Constraints "
                f"'{constraint_type}' muss eine Zahl sein."
            )

        return float(value)

    return SingleProductSinglePeriodSpecification(
        selling_price=float(inputs["selling_price"]),
        purchase_cost=float(inputs["purchase_cost"]),
        salvage_value=float(inputs["salvage_value"]),
        demand_scenarios=tuple(
            float(value)
            for value in inputs["demand_scenarios"]
        ),
        scenario_probabilities=tuple(
            float(value)
            for value in inputs["scenario_probabilities"]
        ),
        minimum_order_quantity=get_optional_parameter(
            "minimum_order_quantity",
            "quantity",
        ),
        maximum_order_quantity=get_optional_parameter(
            "maximum_order_quantity",
            "quantity",
        ),
        budget_limit=get_optional_parameter(
            "budget_limit",
            "budget",
        ),
        storage_capacity=get_optional_parameter(
            "storage_capacity",
            "capacity",
        ),
    )


def build_single_product_single_period_model(
    specification: SingleProductSinglePeriodSpecification,
) -> BuiltSingleProductSinglePeriodModel:
    """
    Erstellt das deterministische PuLP-Modell aus der Modellspezifikation.

    Optionale Constraints werden nur hinzugefügt, wenn ihr Wert in der
    Spezifikation gesetzt ist.
    """

    model = pulp.LpProblem(
        "Newsvendor_Single_Product_Single_Period",
        pulp.LpMaximize,
    )

    # Entscheidungsvariable und szenariospezifische Hilfsvariablen
    order_quantity = pulp.LpVariable(
        "order_quantity",
        lowBound=0,
    )

    scenario_indices = range(
        len(specification.demand_scenarios)
    )

    sold = {
        index: pulp.LpVariable(
            f"sold_{index}",
            lowBound=0,
        )
        for index in scenario_indices
    }

    leftover = {
        index: pulp.LpVariable(
            f"leftover_{index}",
            lowBound=0,
        )
        for index in scenario_indices
    }

    # Feste Constraints für Nachfrage und Bestandsbilanz
    for index, demand in enumerate(
        specification.demand_scenarios
    ):
        model += (
            sold[index] <= demand,
            f"demand_limit_{index}",
        )

        model += (
            sold[index] + leftover[index]
            == order_quantity,
            f"inventory_balance_{index}",
        )

    # Erwarteter Erlös und Zielfunktion
    expected_revenue = pulp.lpSum(
        specification.scenario_probabilities[index]
        * (
            specification.selling_price * sold[index]
            + specification.salvage_value * leftover[index]
        )
        for index in scenario_indices
    )

    model += (
        expected_revenue
        - specification.purchase_cost * order_quantity,
        "expected_profit",
    )

    # Vom Nutzer bestätigte optionale Constraints
    if specification.minimum_order_quantity is not None:
        model += (
            order_quantity
            >= specification.minimum_order_quantity,
            "minimum_order_quantity",
        )

    if specification.maximum_order_quantity is not None:
        model += (
            order_quantity
            <= specification.maximum_order_quantity,
            "maximum_order_quantity",
        )

    if specification.budget_limit is not None:
        model += (
            specification.purchase_cost * order_quantity
            <= specification.budget_limit,
            "budget_limit",
        )

    if specification.storage_capacity is not None:
        model += (
            order_quantity
            <= specification.storage_capacity,
            "storage_capacity",
        )

    return BuiltSingleProductSinglePeriodModel(
        model=model,
        order_quantity=order_quantity,
        sold=sold,
        leftover=leftover,
    )


def solve_single_product_single_period(
    specification: SingleProductSinglePeriodSpecification,
) -> SingleProductSinglePeriodResult:
    """
    Baut und löst das PuLP-Modell.

    Bei optimaler Lösung werden Bestellmenge, erwarteter Gewinn und
    Szenarioergebnisse ausgelesen. Bei anderen Solverstatus bleiben die
    fachlichen Ergebniswerte leer.
    """

    built_model = build_single_product_single_period_model(
        specification
    )

    model_representation = str(built_model.model)

    built_model.model.solve(
        pulp.PULP_CBC_CMD(msg=False)
    )

    solver_status = pulp.LpStatus[
        built_model.model.status
    ]

    if solver_status != "Optimal":
        return SingleProductSinglePeriodResult(
            solver_status=solver_status,
            order_quantity=None,
            expected_profit=None,
            scenario_results=(),
            model_representation=model_representation,
        )

    order_quantity_value = (
        built_model.order_quantity.value()
    )

    objective_value = pulp.value(
        built_model.model.objective
    )

    scenario_results = tuple(
        ScenarioResult(
            demand=specification.demand_scenarios[index],
            probability=(
                specification.scenario_probabilities[index]
            ),
            sold=built_model.sold[index].value(),
            leftover=built_model.leftover[index].value(),
        )
        for index in range(
            len(specification.demand_scenarios)
        )
    )

    return SingleProductSinglePeriodResult(
        solver_status=solver_status,
        order_quantity=order_quantity_value,
        expected_profit=objective_value,
        scenario_results=scenario_results,
        model_representation=model_representation,
    )


def optimize_single_product_single_period(
    inputs: dict[str, Any],
    constraints: list[StructuredConstraint],
) -> dict[str, Any]:
    """
    Führt Spezifikation, Modellaufbau und Solverausführung zusammen.

    Das Ergebnis wird als Dictionary für den zentralen Result-Node zurückgegeben.
    """

    specification = (
        create_single_product_single_period_specification(
            inputs,
            constraints,
        )
    )

    result = solve_single_product_single_period(
        specification
    )

    return asdict(result)


def _format_result_number(
    value: float | None,
) -> str:
    if value is None:
        return "–"

    numeric_value = float(value)

    if numeric_value.is_integer():
        return str(int(numeric_value))

    return f"{numeric_value:g}"


def _format_result_currency(
    value: float | None,
) -> str:
    if value is None:
        return "–"

    formatted = f"{float(value):,.2f}"

    formatted = (
        formatted
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )

    return f"{formatted} €"


def _format_result_percentage(
    value: float | None,
) -> str:
    if value is None:
        return "–"

    return f"{float(value) * 100:g} %"


def build_single_product_single_period_result_presentation(
    optimization_result: dict[str, Any],
) -> VariantResultPresentation:
    """
    Bereitet die fachlichen Ergebniskennzahlen und
    Szenariodetails dieser Modellvariante für die UI auf.
    """

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
            "label": "Erwarteter Gewinn",
            "value": _format_result_currency(
                optimization_result.get(
                    "expected_profit"
                )
            ),
        },
    ]

    detail_rows = [
        {
            "Nachfrage": _format_result_number(
                scenario.get("demand")
            ),
            "Wahrscheinlichkeit": (
                _format_result_percentage(
                    scenario.get("probability")
                )
            ),
            "Verkauft": _format_result_number(
                scenario.get("sold")
            ),
            "Restbestand": _format_result_number(
                scenario.get("leftover")
            ),
        }
        for scenario in optimization_result.get(
            "scenario_results",
            [],
        )
        if isinstance(scenario, dict)
    ]

    return {
        "metrics": metrics,
        "technical_metrics": [],
        "detail_title": (
            "Ergebnis nach Nachfrageszenario"
        ),
        "detail_rows": detail_rows,
    }


VARIANT_DEFINITION: ModelVariantDefinition = {
    "variant_id": "single_product_single_period",
    "variant_name": "Single Product, Single Period",
    "selection_criteria": {
        "decision": "order_quantity",
        "objective": "maximize_profit",
        "demand_type": "uncertain",
        "product_scope": "single_product",
        "planning_horizon": "single_period",
    },
    "input_fields": INPUT_FIELDS,
    "supported_constraints": SUPPORTED_CONSTRAINTS,
    "input_validator": validate_single_product_single_period_inputs,
    "constraint_validator": validate_single_product_single_period_constraints,
    "optimizer": optimize_single_product_single_period,
    "result_builder": build_single_product_single_period_result_presentation,
}
