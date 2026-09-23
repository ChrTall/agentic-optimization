"""
Validierung modularer Modellspezifikationen.
"""

import math
from collections import Counter
from typing import Literal, NotRequired, TypedDict

from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
    OBJECTIVE_COMPONENTS,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.types import (
    ConstraintParameterDefinition,
    ModelInputField,
)


MissingInputSource = Literal[
    "forecast",
    "product_fact",
    "global_fact",
    "constraint_parameter",
]


FieldDefinition = (
    ModelInputField
    | ConstraintParameterDefinition
)

class MissingInput(TypedDict):
    """Beschreibt eine noch benötigte Eingabe."""

    source: MissingInputSource
    field: ModelInputField | ConstraintParameterDefinition
    product_id: NotRequired[str]
    component_id: NotRequired[str]
    constraint_index: NotRequired[int]


class SpecificationValidationResult(TypedDict):
    """Ergebnis der Prüfung einer Modellspezifikation."""

    is_valid: bool
    errors: list[str]
    missing_inputs: list[MissingInput]


def _validate_specification_structure(
    specification: object,
) -> list[str]:
    """Prüft die Containerstruktur einer Modellspezifikation."""

    if not isinstance(specification, dict):
        return [
            "Die Modellspezifikation muss als Objekt vorliegen."
        ]

    errors: list[str] = []

    core_id = specification.get("core_id")

    if (
        core_id is not None
        and (
            not isinstance(core_id, str)
            or not core_id.strip()
        )
    ):
        errors.append(
            "Die ID des Grundmodells muss eine "
            "nicht leere Zeichenkette sein."
        )

    products = specification.get("products", [])

    if not isinstance(products, list):
        errors.append(
            "Die Produkte müssen als Liste vorliegen."
        )
    else:
        for index, product in enumerate(products):
            if not isinstance(product, dict):
                errors.append(
                    f"Das Produkt an Position {index} "
                    "muss als Objekt vorliegen."
                )
                continue

            product_id = product.get("product_id")

            product_reference = (
                f"'{product_id}'"
                if (
                    isinstance(product_id, str)
                    and product_id.strip()
                )
                else f"an Position {index}"
            )

            if (
                product_id is not None
                and not isinstance(product_id, str)
            ):
                errors.append(
                    f"Die Produkt-ID an Position {index} "
                    "muss eine Zeichenkette sein."
                )

            forecast = product.get("forecast", {})

            if not isinstance(forecast, dict):
                errors.append(
                    f"Die Nachfrageprognose für Produkt "
                    f"{product_reference} muss als Objekt vorliegen."
                )

            facts = product.get("facts", {})

            if not isinstance(facts, dict):
                errors.append(
                    f"Die Produktangaben des Produkts "
                    f"{product_reference} müssen als Objekt "
                    "vorliegen."
                )

    global_facts = specification.get(
        "global_facts",
        {},
    )

    if not isinstance(global_facts, dict):
        errors.append(
            "Die globalen Angaben müssen als Objekt vorliegen."
        )

    objective = specification.get("objective", {})

    if not isinstance(objective, dict):
        errors.append(
            "Die Zielfunktion muss als Objekt vorliegen."
        )
    else:
        objective_sense = objective.get("sense")

        if (
            objective_sense is not None
            and not isinstance(objective_sense, str)
        ):
            errors.append(
                "Die Zielrichtung muss als Zeichenkette vorliegen."
            )

        component_ids = objective.get(
            "component_ids",
            [],
        )

        if not isinstance(component_ids, list):
            errors.append(
                "Die Zielfunktionsbausteine müssen "
                "als Liste vorliegen."
            )
        else:
            for index, component_id in enumerate(
                component_ids
            ):
                if (
                    not isinstance(component_id, str)
                    or not component_id.strip()
                ):
                    errors.append(
                        f"Die ID des Zielfunktionsbausteins "
                        f"an Position {index} muss eine "
                        "nicht leere Zeichenkette sein."
                    )

    constraints = specification.get(
        "constraints",
        [],
    )

    if not isinstance(constraints, list):
        errors.append(
            "Die Constraints müssen als Liste vorliegen."
        )
    else:
        for index, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                errors.append(
                    f"Der Constraint an Position {index} "
                    "muss als Objekt vorliegen."
                )
                continue

            component_id = constraint.get("component_id")

            if (
                component_id is not None
                and (
                    not isinstance(component_id, str)
                    or not component_id.strip()
                )
            ):
                errors.append(
                    f"Die ID des Constraints an Position "
                    f"{index} muss eine nicht leere "
                    "Zeichenkette sein."
                )

            parameters = constraint.get(
                "parameters",
                {},
            )

            if not isinstance(parameters, dict):
                errors.append(
                    f"Die Parameter des Constraints an "
                    f"Position {index} müssen als Objekt "
                    "vorliegen."
                )

            product_id = constraint.get("product_id")

            if (
                product_id is not None
                and not isinstance(product_id, str)
            ):
                errors.append(
                    f"Die Produkt-ID des Constraints an "
                    f"Position {index} muss eine "
                    "Zeichenkette sein."
                )

    return errors


def _has_value(values: dict, name: str) -> bool:
    """Prüft, ob ein verwendbarer Wert vorhanden ist."""

    if name not in values:
        return False

    value = values[name]

    if value is None:
        return False

    if isinstance(value, str) and not value.strip():
        return False

    return not (isinstance(value, (list, dict)) and not value)


def _to_finite_number(
    value: object,
) -> float | None:
    """Konvertiert einen numerischen Wert in einen endlichen Float."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        return None

    try:
        numeric_value = float(value)
    except (OverflowError, ValueError):
        return None

    if not math.isfinite(numeric_value):
        return None

    return numeric_value


def _validate_field_value(
    field: FieldDefinition,
    value: object,
    location: str,
) -> list[str]:
    """Prüft Typ und Wertebereich eines vorhandenen Feldwerts."""

    errors: list[str] = []
    field_type = field["type"]
    label = field["label"]

    is_number = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )

    finite_number = _to_finite_number(value)

    if field_type == "number":
        if not is_number:
            return [
                (f"Das Feld '{label}' {location} "
                "muss eine Zahl sein.")
            ]

        if finite_number is None:
            return [
                (f"Das Feld '{label}' {location} "
                "muss eine endliche Zahl sein.")
            ]

    elif field_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return [
                (f"Das Feld '{label}' {location} "
                "muss eine ganze Zahl sein.")
            ]

    elif field_type == "forecast_quantiles":
        if not isinstance(value, list):
            return [
                (f"Das Feld '{label}' {location} "
                "muss eine Liste von Quantilen sein.")
            ]

        return errors

    else:
        return [
            (f"Der Feldtyp '{field_type}' für das Feld "
            f"'{label}' wird nicht unterstützt.")
        ]

    minimum = field.get("min_value")

    if minimum is not None and value < minimum:
        errors.append(
            f"Das Feld '{label}' {location} "
            f"muss mindestens {minimum} betragen."
        )

    maximum = field.get("max_value")

    if maximum is not None and value > maximum:
        errors.append(
            f"Das Feld '{label}' {location} "
            f"darf höchstens {maximum} betragen."
        )

    return errors


def _deduplicate_missing_inputs(
    missing_inputs: list[MissingInput],
) -> list[MissingInput]:
    """Entfernt mehrfach angeforderte Eingabefelder."""

    unique_inputs: list[MissingInput] = []
    seen_keys: set[
        tuple[str, str, str | None, int | None]
    ] = set()

    for missing_input in missing_inputs:
        constraint_index = (
            missing_input.get("constraint_index")
            if missing_input["source"]
            == "constraint_parameter"
            else None
        )

        key = (
            missing_input["source"],
            missing_input["field"]["name"],
            missing_input.get("product_id"),
            constraint_index,
        )
        if key in seen_keys:
            continue

        seen_keys.add(key)
        unique_inputs.append(missing_input)

    return unique_inputs


def validate_model_specification(
    specification: ModelSpecification,
) -> SpecificationValidationResult:
    """Prüft Vollständigkeit und Kompatibilität der Spezifikation."""

    structure_errors = _validate_specification_structure(
        specification
    )

    if structure_errors:
        return {
            "is_valid": False,
            "errors": structure_errors,
            "missing_inputs": [],
        }
    errors: list[str] = []
    missing_inputs: list[MissingInput] = []

    core_id = specification.get("core_id")

    registered_core = CORE_REGISTRY.get(core_id)

    if registered_core is None:
        errors.append(
            f"Das Grundmodell '{core_id}' wird nicht unterstützt."
        )

        return {
            "is_valid": False,
            "errors": errors,
            "missing_inputs": missing_inputs,
        }

    core_definition = registered_core["definition"]

    products = specification.get("products", [])

    if len(products) < core_definition["minimum_products"]:
        errors.append(
            f"Das Grundmodell '{core_id}' benötigt "
            "mindestens ein Produkt."
        )

    product_ids = [
        product.get("product_id")
        for product in products
        if product.get("product_id")
    ]

    if len(product_ids) != len(set(product_ids)):
        errors.append(
            "Jede Produkt-ID darf nur einmal vorkommen."
        )

    for product in products:
        product_id = product.get("product_id")

        if not product_id:
            errors.append(
                "Für jedes Produkt muss eine Produkt-ID angegeben werden."
            )
            continue

        forecast = product.get("forecast", {})

        forecast_error_count = len(errors)

        for field in core_definition["required_forecast_fields"]:
            field_name = field["name"]

            if not _has_value(forecast, field_name):
                missing_inputs.append(
                    {
                        "source": "forecast",
                        "field": field,
                        "product_id": product_id,
                    }
                )
                continue

            errors.extend(
                _validate_field_value(
                    field,
                    forecast[field_name],
                    f"für Produkt '{product_id}'",
                )
            )

        forecast_complete = all(
            _has_value(forecast, field["name"])
            for field in core_definition["required_forecast_fields"]
        )

        if (
            forecast_complete
            and len(errors) == forecast_error_count
        ):
            forecast_errors = registered_core[
                "forecast_validator"
            ](forecast)

            errors.extend(
                f"Die Nachfrageprognose für Produkt "
                f"'{product_id}' ist ungültig: {error}"
                for error in forecast_errors
            )

        facts = product.get("facts", {})

        for field in core_definition[
            "required_product_facts"
        ]:
            field_name = field["name"]

            if not _has_value(facts, field_name):
                missing_inputs.append(
                    {
                        "source": "product_fact",
                        "field": field,
                        "product_id": product_id,
                    }
                )
                continue

            errors.extend(
                _validate_field_value(
                    field,
                    facts[field_name],
                    f"für Produkt '{product_id}'",
                )
            )

    global_facts = specification.get(
        "global_facts",
        {},
    )

    for field in core_definition[
        "required_global_facts"
    ]:
        field_name = field["name"]

        if not _has_value(global_facts, field_name):
            missing_inputs.append(
                {
                    "source": "global_fact",
                    "field": field,
                }
            )
            continue

        errors.extend(
            _validate_field_value(
                field,
                global_facts[field_name],
                "in den globalen Angaben",
            )
        )

    provided_quantities = set(
        core_definition["provided_quantities"]
    )

    objective = specification.get("objective", {})
    component_ids = objective.get("component_ids", [])

    component_counts = Counter(component_ids)

    for component_id, count in component_counts.items():
        if count > 1:
            errors.append(
                f"Der Zielfunktionsbaustein '{component_id}' "
                "darf nur einmal ausgewählt werden."
            )

    if objective.get("sense") not in {"maximize", "minimize"}:
        errors.append(
            "Die Zielrichtung muss 'maximize' oder 'minimize' sein."
        )

    if not component_ids:
        errors.append(
            "Es wurde kein Zielfunktionsbaustein ausgewählt."
        )

    selected_component_ids = set(component_ids)

    for required_component_id in core_definition[
        "required_objective_component_ids"
    ]:
        if required_component_id not in selected_component_ids:
            errors.append(
                f"Das Grundmodell '{core_id}' benötigt "
                "den Zielfunktionsbaustein "
                f"'{required_component_id}'."
            )

    for component_id in component_ids:
        definition = OBJECTIVE_COMPONENTS.get(component_id)

        if definition is None:
            errors.append(
                f"Der Zielfunktionsbaustein '{component_id}' "
                "ist nicht registriert."
            )
            continue

        if core_id not in definition["compatible_core_ids"]:
            errors.append(
                f"Der Zielfunktionsbaustein '{component_id}' "
                f"ist nicht mit '{core_id}' kompatibel."
            )

        missing_quantities = (
            set(definition["required_model_quantities"])
            - provided_quantities
        )

        for quantity_id in sorted(missing_quantities):
            errors.append(
                f"Der Zielfunktionsbaustein '{component_id}' "
                f"benötigt die Modellgröße '{quantity_id}'."
            )

        for product in products:
            product_id = product.get("product_id")

            if not product_id:
                continue

            facts = product.get("facts", {})

            for field in definition["required_product_facts"]:
                field_name = field["name"]

                if not _has_value(facts, field_name):
                    missing_inputs.append(
                        {
                            "source": "product_fact",
                            "field": field,
                            "product_id": product_id,
                            "component_id": component_id,
                        }
                    )
                    continue

                errors.extend(
                    _validate_field_value(
                        field,
                        facts[field_name],
                        f"für Produkt '{product_id}'",
                    )
                )

        global_facts = specification.get("global_facts", {})

        for field in definition["required_global_facts"]:
            field_name = field["name"]

            if not _has_value(global_facts, field_name):
                missing_inputs.append(
                    {
                        "source": "global_fact",
                        "field": field,
                        "component_id": component_id,
                    }
                )
                continue

            errors.extend(
                _validate_field_value(
                    field,
                    global_facts[field_name],
                    "in den globalen Angaben",
                )
            )

    valid_product_ids = set(product_ids)

    seen_constraint_keys: set[
        tuple[str | None, str | None]
    ] = set()

    for index, constraint in enumerate(
        specification.get("constraints", [])
    ):
        component_id = constraint.get("component_id")
        definition = CONSTRAINT_COMPONENTS.get(component_id)

        if definition is None:
            errors.append(
                f"Der Constraint-Baustein '{component_id}' "
                "ist nicht registriert."
            )
            continue

        if core_id not in definition["compatible_core_ids"]:
            errors.append(
                f"Der Constraint-Baustein '{component_id}' "
                f"ist nicht mit '{core_id}' kompatibel."
            )

        missing_quantities = (
            set(definition["required_model_quantities"])
            - provided_quantities
        )

        for quantity_id in sorted(missing_quantities):
            errors.append(
                f"Der Constraint-Baustein '{component_id}' "
                f"benötigt die Modellgröße '{quantity_id}'."
            )

        product_id = constraint.get("product_id")

        constraint_key = (
            component_id,
            (
                product_id
                if definition["scope"] == "per_product"
                else None
            ),
        )

        if constraint_key in seen_constraint_keys:
            errors.append(
                f"Der Constraint '{component_id}' wurde "
                "für denselben Geltungsbereich mehrfach angegeben."
            )
        else:
            seen_constraint_keys.add(constraint_key)

        if definition["scope"] == "per_product":
            if not product_id:
                errors.append(
                    f"Der Constraint '{component_id}' muss "
                    "einem Produkt zugeordnet sein."
                )
            elif product_id not in valid_product_ids:
                errors.append(
                    f"Der Constraint '{component_id}' verweist "
                    f"auf das unbekannte Produkt '{product_id}'."
                )
        elif product_id is not None:
            errors.append(
                f"Der globale Constraint '{component_id}' "
                "darf keinem einzelnen Produkt zugeordnet sein."
            )

        if (
            definition["scope"] == "per_product"
            and product_id in valid_product_ids
        ):
            relevant_products = [
                product
                for product in products
                if product["product_id"] == product_id
            ]
        elif definition["scope"] == "global":
            relevant_products = products
        else:
            relevant_products = []

        for product in relevant_products:
            relevant_product_id = product["product_id"]
            facts = product.get("facts", {})

            for field in definition["required_product_facts"]:
                field_name = field["name"]

                if not _has_value(facts, field_name):
                    missing_inputs.append(
                        {
                            "source": "product_fact",
                            "field": field,
                            "product_id": relevant_product_id,
                            "component_id": component_id,
                            "constraint_index": index,
                        }
                    )
                    continue

                errors.extend(
                    _validate_field_value(
                        field,
                        facts[field_name],
                        f"für Produkt '{relevant_product_id}'",
                    )
                )

        global_facts = specification.get(
            "global_facts",
            {},
        )

        for field in definition["required_global_facts"]:
            field_name = field["name"]

            if not _has_value(global_facts, field_name):
                missing_inputs.append(
                    {
                        "source": "global_fact",
                        "field": field,
                        "component_id": component_id,
                        "constraint_index": index,
                    }
                )
                continue

            errors.extend(
                _validate_field_value(
                    field,
                    global_facts[field_name],
                    "in den globalen Angaben",
                )
            )

        parameters = constraint.get("parameters", {})

        for field in definition["parameters"]:
            field_name = field["name"]

            if not _has_value(parameters, field_name):
                missing_inputs.append(
                    {
                        "source": "constraint_parameter",
                        "field": field,
                        "component_id": component_id,
                        "constraint_index": index,
                    }
                )
                continue

            errors.extend(
                _validate_field_value(
                    field,
                    parameters[field_name],
                    f"im Constraint '{component_id}'",
                )
            )

    order_bounds_by_product: dict[
        str,
        dict[str, float],
    ] = {}

    for constraint in specification.get("constraints", []):
        component_id = constraint.get("component_id")
        product_id = constraint.get("product_id")
        parameters = constraint.get("parameters", {})

        if not product_id or not isinstance(parameters, dict):
            continue

        if component_id in {
            "minimum_order_quantity",
            "maximum_order_quantity",
        }:
            parameter_value = parameters.get("quantity")
        elif component_id == "order_multiple":
            parameter_value = parameters.get("multiple")
        else:
            continue

        numeric_value = _to_finite_number(
            parameter_value
        )

        if numeric_value is None:
            continue

        order_bounds_by_product.setdefault(
            product_id,
            {},
        )[component_id] = numeric_value

    for product_id, bounds in order_bounds_by_product.items():
        minimum = bounds.get("minimum_order_quantity")
        maximum = bounds.get("maximum_order_quantity")
        multiple = bounds.get("order_multiple")

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            errors.append(
                f"Für das Produkt '{product_id}' darf die "
                "Mindestbestellmenge die Höchstbestellmenge "
                "nicht überschreiten."
            )

        if (
            minimum is not None
            and maximum is not None
            and multiple is not None
            and multiple > 0
        ):
            smallest_allowed_multiple = (
                math.ceil(minimum / multiple)
                * multiple
            )

            if smallest_allowed_multiple > maximum:
                errors.append(
                    f"Für das Produkt '{product_id}' existiert "
                    "zwischen Mindest- und Höchstbestellmenge "
                    "kein zulässiges Bestellvielfaches."
                )

    missing_inputs = _deduplicate_missing_inputs(
        missing_inputs
    )
    errors = list(dict.fromkeys(errors))

    if not errors and not missing_inputs:
        semantic_errors = registered_core[
            "semantic_validator"
        ](specification)

        errors.extend(semantic_errors)
        errors = list(dict.fromkeys(errors))

    return {
        "is_valid": not errors and not missing_inputs,
        "errors": errors,
        "missing_inputs": missing_inputs,
    }