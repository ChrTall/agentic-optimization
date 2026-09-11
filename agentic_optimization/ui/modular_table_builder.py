"""
Erzeugt Tabellenzeilen für viele Produkte.
"""

from typing import Any, Literal, TypedDict

from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.validator import (
    validate_model_specification,
)


ProductStatus = Literal[
    "complete",
    "incomplete",
]


class ProductTableRow(TypedDict):
    """Aufbereitete Tabellenzeile eines Produkts."""

    product_id: str
    values: dict[str, Any]
    forecast_complete: bool
    missing_field_names: list[str]
    status: ProductStatus


def build_product_table_rows(
    specification: ModelSpecification,
) -> list[ProductTableRow]:
    """Erzeugt editierbare Produktzeilen aus der Spezifikation."""

    validation_result = validate_model_specification(
        specification
    )

    missing_inputs_by_product: dict[
        str,
        list,
    ] = {}

    for missing_input in validation_result["missing_inputs"]:
        product_id = missing_input.get("product_id")

        if product_id:
            missing_inputs_by_product.setdefault(
                product_id,
                [],
            ).append(missing_input)

    rows: list[ProductTableRow] = []

    for product in specification["products"]:
        product_id = product["product_id"]
        forecast = product.get("forecast", {})
        facts = product.get("facts", {})

        product_missing_inputs = (
            missing_inputs_by_product.get(
                product_id,
                [],
            )
        )

        missing_field_names = [
            missing_input["field"]["name"]
            for missing_input in product_missing_inputs
        ]

        forecast_complete = not any(
            missing_input["source"] == "forecast"
            for missing_input in product_missing_inputs
        )

        values: dict[str, Any] = {
            "expected_value": forecast.get(
                "expected_value"
            ),
            "horizon_days": forecast.get(
                "horizon_days"
            ),
            **facts,
        }

        for missing_input in product_missing_inputs:
            field = missing_input["field"]

            if field["type"] != "forecast_quantiles":
                values.setdefault(
                    field["name"],
                    None,
                )

        rows.append(
            {
                "product_id": product_id,
                "values": values,
                "forecast_complete": forecast_complete,
                "missing_field_names": missing_field_names,
                "status": (
                    "complete"
                    if not product_missing_inputs
                    else "incomplete"
                ),
            }
        )

    return rows