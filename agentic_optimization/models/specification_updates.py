"""
Aktualisierung modularer Modellspezifikationen.
"""

from copy import deepcopy
from typing import Any

from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.validator import (
    MissingInput,
)


def apply_modular_input_value(
    specification: ModelSpecification,
    missing_input: MissingInput,
    value: Any,
) -> ModelSpecification:
    """Übernimmt einen ergänzten Wert in eine Kopie der Spezifikation."""

    updated_specification = deepcopy(specification)

    source = missing_input["source"]
    field_name = missing_input["field"]["name"]

    if source in {"forecast", "product_fact"}:
        product_id = missing_input.get("product_id")

        if not product_id:
            raise ValueError(
                "Für die Produkteingabe fehlt die Produkt-ID."
            )

        product = next(
            (
                product
                for product
                in updated_specification["products"]
                if product["product_id"] == product_id
            ),
            None,
        )

        if product is None:
            raise ValueError(
                f"Das Produkt '{product_id}' ist nicht vorhanden."
            )

        if source == "forecast":
            product["forecast"][field_name] = value
        else:
            product["facts"][field_name] = value

        return updated_specification

    if source == "global_fact":
        updated_specification["global_facts"][
            field_name
        ] = value

        return updated_specification

    if source == "constraint_parameter":
        constraint_index = missing_input.get(
            "constraint_index"
        )

        if constraint_index is None:
            raise ValueError(
                "Für den Constraint-Parameter fehlt der Index."
            )

        try:
            constraint = updated_specification[
                "constraints"
            ][constraint_index]
        except IndexError as error:
            raise ValueError(
                "Der zugehörige Constraint ist nicht vorhanden."
            ) from error

        constraint["parameters"][field_name] = value

        return updated_specification

    raise ValueError(
        f"Die Eingabequelle '{source}' wird nicht unterstützt."
    )