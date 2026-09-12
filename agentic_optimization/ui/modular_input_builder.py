"""
Bereitet fehlende modulare Modelleingaben für die UI auf.
"""

from typing import Literal, TypedDict

from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.specification_updates import (
    apply_modular_input_value,
)
from agentic_optimization.models.validator import (
    MissingInput,
    validate_model_specification,
)

ProductDisplayMode = Literal[
    "cards",
    "table",
]


class ProductInputGroup(TypedDict):
    """Fehlende Forecast- und Faktendaten eines Produkts."""

    product_id: str
    forecast_inputs: list[MissingInput]
    fact_inputs: list[MissingInput]


class ConstraintInputGroup(TypedDict):
    """Fehlende Parameter eines ausgewählten Constraints."""

    constraint_index: int
    component_id: str
    product_id: str | None
    inputs: list[MissingInput]


class ModularInputLayout(TypedDict):
    """Vollständige UI-Struktur für fehlende Modelleingaben."""

    display_mode: ProductDisplayMode
    product_groups: list[ProductInputGroup]
    global_inputs: list[MissingInput]
    constraint_groups: list[ConstraintInputGroup]
    validation_errors: list[str]


def build_modular_input_layout(
    specification: ModelSpecification,
    *,
    card_limit: int = 3,
) -> ModularInputLayout:
    """Gruppiert fehlende Eingaben nach UI-Bereich."""

    validation_result = validate_model_specification(
        specification
    )

    display_mode: ProductDisplayMode = (
        "cards"
        if len(specification["products"]) <= card_limit
        else "table"
    )

    product_groups_by_id: dict[
        str,
        ProductInputGroup,
    ] = {}

    global_inputs: list[MissingInput] = []

    constraint_groups_by_index: dict[
        int,
        ConstraintInputGroup,
    ] = {}

    for missing_input in validation_result["missing_inputs"]:
        source = missing_input["source"]

        if source in {"forecast", "product_fact"}:
            product_id = missing_input.get("product_id")

            if not product_id:
                continue

            product_group = product_groups_by_id.setdefault(
                product_id,
                {
                    "product_id": product_id,
                    "forecast_inputs": [],
                    "fact_inputs": [],
                },
            )

            target = (
                product_group["forecast_inputs"]
                if source == "forecast"
                else product_group["fact_inputs"]
            )

            target.append(missing_input)
            continue

        if source == "global_fact":
            global_inputs.append(missing_input)
            continue

        if source == "constraint_parameter":
            constraint_index = missing_input.get(
                "constraint_index"
            )

            if constraint_index is None:
                continue

            constraint = specification["constraints"][
                constraint_index
            ]

            constraint_group = (
                constraint_groups_by_index.setdefault(
                    constraint_index,
                    {
                        "constraint_index": constraint_index,
                        "component_id": constraint[
                            "component_id"
                        ],
                        "product_id": constraint.get(
                            "product_id"
                        ),
                        "inputs": [],
                    },
                )
            )

            constraint_group["inputs"].append(
                missing_input
            )

    product_groups = [
        product_groups_by_id[product["product_id"]]
        for product in specification["products"]
        if product["product_id"] in product_groups_by_id
    ]

    constraint_groups = [
        constraint_groups_by_index[index]
        for index in sorted(constraint_groups_by_index)
    ]

    return {
        "display_mode": display_mode,
        "product_groups": product_groups,
        "global_inputs": global_inputs,
        "constraint_groups": constraint_groups,
        "validation_errors": validation_result["errors"],
    }