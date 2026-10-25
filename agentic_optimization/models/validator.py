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
from agentic_optimization.i18n import t


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
            t("Die Modellspezifikation muss als Objekt vorliegen.")
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
            t("Die ID des Grundmodells muss eine nicht leere Zeichenkette sein.")
        )

    products = specification.get("products", [])

    if not isinstance(products, list):
        errors.append(
            t("Die Produkte müssen als Liste vorliegen.")
        )
    else:
        for index, product in enumerate(products):
            if not isinstance(product, dict):
                errors.append(
                    t("Das Produkt an Position {index} muss als Objekt vorliegen.", index=index)
                )
                continue

            product_id = product.get("product_id")

            product_reference = (
                f"'{product_id}'"
                if (
                    isinstance(product_id, str)
                    and product_id.strip()
                )
                else t("an Position {index}", index=index)
            )

            if (
                product_id is not None
                and not isinstance(product_id, str)
            ):
                errors.append(
                    t("Die Produkt-ID an Position {index} muss eine Zeichenkette sein.", index=index)
                )

            forecast = product.get("forecast", {})

            if not isinstance(forecast, dict):
                errors.append(
                    t("Die Nachfrageprognose für Produkt {product_reference} muss als Objekt vorliegen.", product_reference=product_reference)
                )

            facts = product.get("facts", {})

            if not isinstance(facts, dict):
                errors.append(
                    t("Die Produktangaben des Produkts {product_reference} müssen als Objekt vorliegen.", product_reference=product_reference)
                )

    global_facts = specification.get(
        "global_facts",
        {},
    )

    if not isinstance(global_facts, dict):
        errors.append(
            t("Die globalen Angaben müssen als Objekt vorliegen.")
        )

    objective = specification.get("objective", {})

    if not isinstance(objective, dict):
        errors.append(
            t("Die Zielfunktion muss als Objekt vorliegen.")
        )
    else:
        objective_sense = objective.get("sense")

        if (
            objective_sense is not None
            and not isinstance(objective_sense, str)
        ):
            errors.append(
                t("Die Zielrichtung muss als Zeichenkette vorliegen.")
            )

        component_ids = objective.get(
            "component_ids",
            [],
        )

        if not isinstance(component_ids, list):
            errors.append(
                t("Die Zielfunktionsbausteine müssen als Liste vorliegen.")
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
                        t("Die ID des Zielfunktionsbausteins an Position {index} muss eine nicht leere Zeichenkette sein.", index=index)
                    )

    constraints = specification.get(
        "constraints",
        [],
    )

    if not isinstance(constraints, list):
        errors.append(
            t("Die Constraints müssen als Liste vorliegen.")
        )
    else:
        for index, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                errors.append(
                    t("Der Constraint an Position {index} muss als Objekt vorliegen.", index=index)
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
                    t("Die ID des Constraints an Position {index} muss eine nicht leere Zeichenkette sein.", index=index)
                )

            parameters = constraint.get(
                "parameters",
                {},
            )

            if not isinstance(parameters, dict):
                errors.append(
                    t("Die Parameter des Constraints an Position {index} müssen als Objekt vorliegen.", index=index)
                )

            product_id = constraint.get("product_id")

            if (
                product_id is not None
                and not isinstance(product_id, str)
            ):
                errors.append(
                    t("Die Produkt-ID des Constraints an Position {index} muss eine Zeichenkette sein.", index=index)
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
    label = t(field["label"])

    is_number = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )

    finite_number = _to_finite_number(value)

    if field_type == "number":
        if not is_number:
            return [
                (t("Das Feld '{label}' {location} muss eine Zahl sein.", label=label, location=location))
            ]

        if finite_number is None:
            return [
                (t("Das Feld '{label}' {location} muss eine endliche Zahl sein.", label=label, location=location))
            ]

    elif field_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return [
                (t("Das Feld '{label}' {location} muss eine ganze Zahl sein.", label=label, location=location))
            ]

    elif field_type == "forecast_quantiles":
        if not isinstance(value, list):
            return [
                (t("Das Feld '{label}' {location} muss eine Liste von Quantilen sein.", label=label, location=location))
            ]

        return errors

    else:
        return [
            (t("Der Feldtyp '{field_type}' für das Feld '{label}' wird nicht unterstützt.", field_type=field_type, label=label))
        ]

    minimum = field.get("min_value")

    if minimum is not None and value < minimum:
        errors.append(
            t("Das Feld '{label}' {location} muss mindestens {minimum} betragen.", label=label, location=location, minimum=minimum)
        )

    maximum = field.get("max_value")

    if maximum is not None and value > maximum:
        errors.append(
            t("Das Feld '{label}' {location} darf höchstens {maximum} betragen.", label=label, location=location, maximum=maximum)
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
            t("Das Grundmodell '{core_id}' wird nicht unterstützt.", core_id=core_id)
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
            t("Das Grundmodell '{core_id}' benötigt mindestens ein Produkt.", core_id=core_id)
        )

    product_ids = [
        product.get("product_id")
        for product in products
        if product.get("product_id")
    ]

    if len(product_ids) != len(set(product_ids)):
        errors.append(
            t("Jede Produkt-ID darf nur einmal vorkommen.")
        )

    for product in products:
        product_id = product.get("product_id")

        if not product_id:
            errors.append(
                t("Für jedes Produkt muss eine Produkt-ID angegeben werden.")
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
                    t("für Produkt '{product_id}'", product_id=product_id),
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
                t("Die Nachfrageprognose für Produkt '{product_id}' ist ungültig: {error}", product_id=product_id, error=error)
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
                    t("für Produkt '{product_id}'", product_id=product_id),
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
                t("in den globalen Angaben"),
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
                t("Der Zielfunktionsbaustein '{component_id}' darf nur einmal ausgewählt werden.", component_id=component_id)
            )

    if objective.get("sense") not in {"maximize", "minimize"}:
        errors.append(
            t("Die Zielrichtung muss 'maximize' oder 'minimize' sein.")
        )

    if not component_ids:
        errors.append(
            t("Es wurde kein Zielfunktionsbaustein ausgewählt.")
        )

    selected_component_ids = set(component_ids)

    for required_component_id in core_definition[
        "required_objective_component_ids"
    ]:
        if required_component_id not in selected_component_ids:
            errors.append(
                t("Das Grundmodell '{core_id}' benötigt den Zielfunktionsbaustein '{required_component_id}'.", core_id=core_id, required_component_id=required_component_id)
            )

    for component_id in component_ids:
        definition = OBJECTIVE_COMPONENTS.get(component_id)

        if definition is None:
            errors.append(
                t("Der Zielfunktionsbaustein '{component_id}' ist nicht registriert.", component_id=component_id)
            )
            continue

        if core_id not in definition["compatible_core_ids"]:
            errors.append(
                t("Der Zielfunktionsbaustein '{component_id}' ist nicht mit '{core_id}' kompatibel.", component_id=component_id, core_id=core_id)
            )

        missing_quantities = (
            set(definition["required_model_quantities"])
            - provided_quantities
        )

        for quantity_id in sorted(missing_quantities):
            errors.append(
                t("Der Zielfunktionsbaustein '{component_id}' benötigt die Modellgröße '{quantity_id}'.", component_id=component_id, quantity_id=quantity_id)
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
                        t("für Produkt '{product_id}'", product_id=product_id),
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
                    t("in den globalen Angaben"),
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
                t("Der Constraint-Baustein '{component_id}' ist nicht registriert.", component_id=component_id)
            )
            continue

        if core_id not in definition["compatible_core_ids"]:
            errors.append(
                t("Der Constraint-Baustein '{component_id}' ist nicht mit '{core_id}' kompatibel.", component_id=component_id, core_id=core_id)
            )

        missing_quantities = (
            set(definition["required_model_quantities"])
            - provided_quantities
        )

        for quantity_id in sorted(missing_quantities):
            errors.append(
                t("Der Constraint-Baustein '{component_id}' benötigt die Modellgröße '{quantity_id}'.", component_id=component_id, quantity_id=quantity_id)
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
                t("Der Constraint '{component_id}' wurde für denselben Geltungsbereich mehrfach angegeben.", component_id=component_id)
            )
        else:
            seen_constraint_keys.add(constraint_key)

        if definition["scope"] == "per_product":
            if not product_id:
                errors.append(
                    t("Der Constraint '{component_id}' muss einem Produkt zugeordnet sein.", component_id=component_id)
                )
            elif product_id not in valid_product_ids:
                errors.append(
                    t("Der Constraint '{component_id}' verweist auf das unbekannte Produkt '{product_id}'.", component_id=component_id, product_id=product_id)
                )
        elif product_id is not None:
            errors.append(
                t("Der globale Constraint '{component_id}' darf keinem einzelnen Produkt zugeordnet sein.", component_id=component_id)
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
                        t("für Produkt '{relevant_product_id}'", relevant_product_id=relevant_product_id),
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
                    t("in den globalen Angaben"),
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
                    t("im Constraint '{component_id}'", component_id=component_id),
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
                t("Für das Produkt '{product_id}' darf die Mindestbestellmenge die Höchstbestellmenge nicht überschreiten.", product_id=product_id)
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
                    t("Für das Produkt '{product_id}' existiert zwischen Mindest- und Höchstbestellmenge kein zulässiges Bestellvielfaches.", product_id=product_id)
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