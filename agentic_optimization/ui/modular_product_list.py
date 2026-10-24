"""
Erfassung einer Produktliste für die modulare Eingabetabelle.
"""


from typing import Any, TypedDict
from agentic_optimization.models.product_names import clean_product_name
from agentic_optimization.models.specification import (
    ProductSpecification,
)
from agentic_optimization.ui.field_renderer import (
    parse_input_value,
)
from agentic_optimization.ui.modular_input_builder import (
    DynamicProductTableSchema,
)


class DynamicProductTableResult(TypedDict):
    """Geprüfte Produktdaten aus der dynamischen Tabelle."""

    products: list[ProductSpecification]
    errors: list[str]


def _is_empty_table_value(value: Any) -> bool:
    """Erkennt leere Zellen der dynamischen Produkttabelle."""

    return (
        value is None
        or value == ""
        or value == []
    )


def parse_dynamic_product_table(
    rows: list[dict[str, Any]],
    schema: DynamicProductTableSchema,
    *,
    maximum_products: int = 500,
    generate_product_ids: bool = False,
) -> DynamicProductTableResult:
    """Erzeugt Produktobjekte aus vollständigen Tabellenzeilen."""

    active_rows = [
        row
        for row in rows
        if any(
            not _is_empty_table_value(value)
            for value in row.values()
        )
    ]

    if not active_rows:
        return {
            "products": [],
            "errors": [
                "Erfassen Sie mindestens ein Produkt."
            ],
        }

    if len(active_rows) > maximum_products:
        return {
            "products": [],
            "errors": [
                "Es können höchstens "
                f"{maximum_products} Produkte gleichzeitig "
                "erfasst werden."
            ],
        }

    errors: list[str] = []
    products: list[ProductSpecification] = []
    normalized_product_ids: set[str] = set()

    for row_index, row in enumerate(
        active_rows,
        start=1,
    ):
        raw_product_id = row.get("product_id")
        product_id = (
            str(raw_product_id).strip()
            if not _is_empty_table_value(raw_product_id)
            else ""
        )

        if not product_id and generate_product_ids:
            reserved = {str(item.get("product_id", "")).strip().casefold() for item in active_rows}
            index = row_index
            while f"product_{index}" in reserved or f"product_{index}" in normalized_product_ids:
                index += 1
            product_id = f"product_{index}"

        if not product_id:
            errors.append(
                f"In Zeile {row_index} fehlt die "
                "Produktkennung."
            )
            continue

        normalized_product_id = product_id.casefold()

        if normalized_product_id in normalized_product_ids:
            errors.append(
                "Produktkennungen dürfen nicht mehrfach "
                f"vorkommen: {product_id}"
            )
            continue

        normalized_product_ids.add(
            normalized_product_id
        )

        forecast: dict[str, Any] = {}
        facts: dict[str, Any] = {}

        for fields, target in (
            (
                schema["forecast_fields"],
                forecast,
            ),
            (
                schema["fact_fields"],
                facts,
            ),
        ):
            for field in fields:
                field_name = field["name"]
                raw_value = row.get(field_name)

                try:
                    parsed_value = parse_input_value(
                        field,
                        raw_value,
                    )
                except (TypeError, ValueError) as error:
                    errors.append(
                        f"{product_id}: {error}"
                    )
                    continue

                if (
                    field["required"]
                    and _is_empty_table_value(
                        parsed_value
                    )
                ):
                    errors.append(
                        f"{product_id}: Bitte geben Sie "
                        f"einen Wert für "
                        f"„{field['label']}“ ein."
                    )
                    continue

                if _is_empty_table_value(parsed_value):
                    continue

                target[field_name] = parsed_value

        products.append(
            {
                "product_id": product_id,
                "forecast": forecast,
                "facts": facts,
            }
        )
        if row.get("display_name"):
            name = clean_product_name(row["display_name"])
            if name is None:
                errors.append(f"Zeile {row_index}: Der Produktname darf höchstens 120 Zeichen enthalten.")
            else:
                products[-1]["display_name"] = name

    if errors:
        return {
            "products": [],
            "errors": errors,
        }

    return {
        "products": products,
        "errors": [],
    }
