import math
from typing import Any

from agentic_optimization.models.types import (
    ConstraintDefinition,
    ModelInputField,
    ModelVariantDefinition,
    StructuredConstraint,
)

REQUIRED_FIELDS = [
    "selling_price",
    "purchase_cost",
    "salvage_value",
    "demand_scenarios",
    "scenario_probabilities",
]


INPUT_FIELDS: list[ModelInputField] = [
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
        "help": "Kommagetrennt, z. B. 80, 100, 120",
    },
    {
        "name": "scenario_probabilities",
        "label": "Wahrscheinlichkeiten der Szenarien",
        "type": "number_list",
        "required": True,
        "help": "Kommagetrennt, z. B. 0.2, 0.5, 0.3",
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
                "help": (
                    "Maximale Anzahl von Produkteinheiten, "
                    "die gelagert werden können."
                ),
            }
        ],
    },
}


def validate_single_product_single_period_inputs(
    inputs: dict[str, Any],
) -> list[str]:
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
        and salvage_value > purchase_cost
    ):
        errors.append(
            "Der Restwert darf nicht größer als die Einkaufskosten sein."
        )

    if not demand_scenarios:
        errors.append(
            "Bitte gib mindestens ein Nachfrageszenario ein."
        )

    if not scenario_probabilities:
        errors.append(
            "Bitte gib mindestens eine Wahrscheinlichkeit ein."
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


def _format_number(value: float) -> str:
    """
    Formatiert validierte Zahlen deterministisch für den erzeugten Python-Code.
    """

    if isinstance(value, bool):
        raise TypeError("Boolesche Werte sind keine gültigen Zahlen.")

    if isinstance(value, int):
        return str(value)

    if not isinstance(value, float):
        raise TypeError(
            f"Erwartete int oder float, erhalten: {type(value).__name__}"
        )

    return repr(value)


def _format_number_list(values: list[float]) -> str:
    return "[" + ", ".join(_format_number(value) for value in values) + "]"


def validate_single_product_single_period_constraints(
    inputs: dict[str, Any],
    constraints: list[StructuredConstraint],
) -> list[str]:
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


def build_single_product_single_period_code(
    inputs: dict[str, Any],
    constraints: list[StructuredConstraint],
) -> str:
    """
    Erzeugt deterministisch PuLP-Code für ein Newsvendor-Modell
    mit einem Produkt und einer Periode.

    Die Funktion erwartet vollständig validierte Inputs und Constraints.
    Sie verwendet kein LLM und erfindet keine Werte.
    """

    selling_price = inputs["selling_price"]
    purchase_cost = inputs["purchase_cost"]
    salvage_value = inputs["salvage_value"]
    demand_scenarios = inputs["demand_scenarios"]
    scenario_probabilities = inputs["scenario_probabilities"]

    constraints_by_type: dict[str, StructuredConstraint] = {}

    for constraint in constraints:
        constraint_type = constraint["type"]

        if constraint_type in constraints_by_type:
            raise ValueError(
                f"Constraint-Typ '{constraint_type}' wurde mehrfach angegeben."
            )

        constraints_by_type[constraint_type] = constraint

    unsupported_types = (
        set(constraints_by_type) - set(SUPPORTED_CONSTRAINTS)
    )

    if unsupported_types:
        unsupported = ", ".join(sorted(unsupported_types))
        raise ValueError(
            f"Nicht unterstützte Constraint-Typen: {unsupported}"
        )

    lines = [
        "import pulp",
        "",
        "# Eingabedaten",
        f"selling_price = {_format_number(selling_price)}",
        f"purchase_cost = {_format_number(purchase_cost)}",
        f"salvage_value = {_format_number(salvage_value)}",
        (
            "demand_scenarios = "
            f"{_format_number_list(demand_scenarios)}"
        ),
        (
            "scenario_probabilities = "
            f"{_format_number_list(scenario_probabilities)}"
        ),
        "",
        "# Modell erstellen",
        (
            'model = pulp.LpProblem('
            '"Newsvendor_Single_Product_Single_Period", '
            "pulp.LpMaximize"
            ")"
        ),
        "",
        "# Entscheidungsvariable",
        (
            'order_quantity = pulp.LpVariable('
            '"order_quantity", lowBound=0)'
        ),
        "",
        "# Szenarioabhängige Hilfsvariablen",
        "sold = {",
        '    i: pulp.LpVariable(f"sold_{i}", lowBound=0)',
        "    for i in range(len(demand_scenarios))",
        "}",
        "",
        "leftover = {",
        '    i: pulp.LpVariable(f"leftover_{i}", lowBound=0)',
        "    for i in range(len(demand_scenarios))",
        "}",
        "",
        "# Feste Modellbedingungen",
        "for i, demand in enumerate(demand_scenarios):",
        (
            "    model += sold[i] <= demand, "
            'f"demand_limit_{i}"'
        ),
        "    model += (",
        "        sold[i] + leftover[i] == order_quantity,",
        '        f"inventory_balance_{i}",',
        "    )",
        "",
    ]

    for constraint_type in SUPPORTED_CONSTRAINTS:
        constraint = constraints_by_type.get(constraint_type)

        if constraint is None:
            continue

        parameters = constraint["parameters"]

        if constraint_type == "minimum_order_quantity":
            quantity = _format_number(parameters["quantity"])

            lines.extend(
                [
                    "# Optionale Mindestbestellmenge",
                    (
                        f"model += order_quantity >= {quantity}, "
                        '"minimum_order_quantity"'
                    ),
                    "",
                ]
            )

        elif constraint_type == "maximum_order_quantity":
            quantity = _format_number(parameters["quantity"])

            lines.extend(
                [
                    "# Optionale Höchstbestellmenge",
                    (
                        f"model += order_quantity <= {quantity}, "
                        '"maximum_order_quantity"'
                    ),
                    "",
                ]
            )

        elif constraint_type == "budget_limit":
            budget = _format_number(parameters["budget"])

            lines.extend(
                [
                    "# Optionale Budgetgrenze",
                    "model += (",
                    (
                        "    purchase_cost * order_quantity "
                        f"<= {budget},"
                    ),
                    '    "budget_limit",',
                    ")",
                    "",
                ]
            )

        elif constraint_type == "storage_capacity":
            capacity = _format_number(parameters["capacity"])

            lines.extend(
                [
                    "# Optionale Lagerkapazität in Produkteinheiten",
                    (
                        f"model += order_quantity <= {capacity}, "
                        '"storage_capacity"'
                    ),
                    "",
                ]
            )

        else:
            raise ValueError(
                f"Nicht unterstützter Constraint-Typ: {constraint_type}"
            )

    lines.extend(
        [
            "# Erwartete Erlöse",
            "expected_revenue = pulp.lpSum(",
            "    scenario_probabilities[i]",
            "    * (",
            "        selling_price * sold[i]",
            "        + salvage_value * leftover[i]",
            "    )",
            "    for i in range(len(demand_scenarios))",
            ")",
            "",
            "# Zielfunktion: Erwarteter Gewinn",
            (
                "model += expected_revenue "
                "- purchase_cost * order_quantity"
            ),
            "",
            "# Modell lösen",
            "model.solve()",
            "",
            'print("Status:", pulp.LpStatus[model.status])',
            (
                'print("Optimale Bestellmenge:", '
                "order_quantity.value())"
            ),
            (
                'print("Erwarteter Gewinn:", '
                "pulp.value(model.objective))"
            ),
        ]
    )

    return "\n".join(lines) + "\n"


VARIANT_DEFINITION: ModelVariantDefinition = {
    "variant_id": "single_product_single_period",
    "variant_name": "Single Product, Single Period",
    "required_fields": REQUIRED_FIELDS,
    "input_fields": INPUT_FIELDS,
    "supported_constraints": SUPPORTED_CONSTRAINTS,
    "input_validator": validate_single_product_single_period_inputs,
    "constraint_validator": (
        validate_single_product_single_period_constraints
    ),
    "code_builder": build_single_product_single_period_code,
}