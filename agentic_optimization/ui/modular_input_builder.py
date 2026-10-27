"""
Bereitet fehlende modulare Modelleingaben für die UI auf.
"""

from copy import deepcopy
from typing import Any, Literal, TypedDict

from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ModelSpecification,
)
from agentic_optimization.models.specification_updates import (
    apply_modular_input_value,
)
from agentic_optimization.models.validator import (
    MissingInput,
    validate_model_specification,
)
from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
    OBJECTIVE_COMPONENTS,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)
from agentic_optimization.models.types import (
    ModelInputField,
)
from agentic_optimization.i18n import t

ProductDisplayMode = Literal[
    "cards",
    "table",
]


class DynamicProductTableSchema(TypedDict):
    """Spalten einer Produkttabelle ohne vorhandene Produktzeilen."""

    forecast_fields: list[ModelInputField]
    fact_fields: list[ModelInputField]


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


def input_key(missing_input: MissingInput) -> tuple:
    """Fachlicher Schlüssel einer Eingabe: Quelle, Produkt/Baustein, Feld."""

    return (
        missing_input["source"],
        missing_input.get("product_id"),
        missing_input.get("constraint_index"),
        missing_input["field"]["name"],
    )


def _append_unique_fields(
    target: list[ModelInputField],
    fields: list[ModelInputField],
) -> None:
    """Ergänzt Eingabefelder ohne doppelte Feldnamen."""

    existing_names = {
        field["name"]
        for field in target
    }

    for field in fields:
        if field["name"] in existing_names:
            continue

        target.append(deepcopy(field))
        existing_names.add(field["name"])


def build_dynamic_product_table_schema(
    core_id: str,
    proposal: ComponentSelectionProposal,
) -> DynamicProductTableSchema:
    """
    Bestimmt die Produktspalten aus Grundmodell und Bausteinen.

    Dafür müssen noch keine konkreten Produkte existieren.
    """

    registered_core = CORE_REGISTRY.get(core_id)

    if registered_core is None:
        raise ValueError(
            t("Das Grundmodell '{core_id}' ist nicht registriert.", core_id=core_id)
        )

    if proposal["core_id"] != core_id:
        raise ValueError(
            t("Grundmodell und Bausteinvorschlag stimmen nicht überein.")
        )

    core_definition = registered_core["definition"]

    forecast_fields = deepcopy(
        core_definition["required_forecast_fields"]
    )
    fact_fields = deepcopy(
        core_definition["required_product_facts"]
    )

    for selected_component in proposal[
        "objective_components"
    ]:
        component_id = selected_component["component_id"]
        definition = OBJECTIVE_COMPONENTS[component_id]

        _append_unique_fields(
            fact_fields,
            definition["required_product_facts"],
        )

    for selected_constraint in proposal["constraints"]:
        component_id = selected_constraint["component_id"]
        definition = CONSTRAINT_COMPONENTS[component_id]

        _append_unique_fields(
            fact_fields,
            definition["required_product_facts"],
        )

    return {
        "forecast_fields": forecast_fields,
        "fact_fields": fact_fields,
    }


def _get_product_group_inputs(
    product_group: ProductInputGroup,
) -> list[MissingInput]:
    """Liefert alle offenen Produktfelder in Tabellenreihenfolge."""

    return [
        *product_group["forecast_inputs"],
        *product_group["fact_inputs"],
    ]


def can_render_product_input_table(
    layout: ModularInputLayout,
) -> bool:
    """
    Prüft, ob alle Produkte dieselben offenen Tabellenfelder besitzen.
    """

    product_groups = layout["product_groups"]

    if (
        layout["display_mode"] != "table"
        or not product_groups
    ):
        return False

    signatures = [
        [
            (
                missing_input["source"],
                missing_input["field"]["name"],
            )
            for missing_input
            in _get_product_group_inputs(product_group)
        ]
        for product_group in product_groups
    ]

    first_signature = signatures[0]

    if not first_signature:
        return False

    if any(
        signature != first_signature
        for signature in signatures[1:]
    ):
        return False

    field_names = [
        field_name
        for _, field_name in first_signature
    ]

    return len(field_names) == len(set(field_names))


def build_product_input_table_rows(
    layout: ModularInputLayout,
    specification: ModelSpecification | None = None,
) -> list[dict[str, Any]]:
    """Erzeugt eine editierbare Zeile je Produkt.

    Mit ``specification`` werden vorhandene Werte vorbelegt.
    """

    if not can_render_product_input_table(layout):
        raise ValueError(
            t("Die offenen Produktfelder können nicht sicher in einer gemeinsamen Tabelle dargestellt werden.")
        )

    first_group = layout["product_groups"][0]
    field_names = [
        missing_input["field"]["name"]
        for missing_input
        in _get_product_group_inputs(first_group)
    ]

    products_by_id = {
        product["product_id"]: product
        for product in (specification or {}).get("products", [])
    }

    def existing(product_id: str, missing_input: MissingInput) -> Any:
        product = products_by_id.get(product_id) or {}
        container = (
            product.get("forecast") or {}
            if missing_input["source"] == "forecast"
            else product.get("facts") or {}
        )
        value = container.get(missing_input["field"]["name"])

        if missing_input["field"]["type"] == "forecast_quantiles" and value:
            return "; ".join(
                f"{entry['level']:g}: {entry['value']:g}"
                for entry in value
                if isinstance(entry, dict)
            )

        if isinstance(value, list):
            return "; ".join(f"{entry:g}" for entry in value)

        return value

    return [
        {
            "product_id": product_group["product_id"],
            **{
                missing_input["field"]["name"]: existing(
                    product_group["product_id"], missing_input
                )
                for missing_input in _get_product_group_inputs(product_group)
            },
        }
        for product_group in layout["product_groups"]
    ]


def build_modular_input_layout(
    specification: ModelSpecification,
    *,
    card_limit: int = 3,
    include_complete: bool = False,
) -> ModularInputLayout:
    """Gruppiert Eingaben nach UI-Bereich.

    Standardmäßig nur fehlende Eingaben (für Chat und Prüfung). Mit
    ``include_complete`` alle Eingabefelder des Modells in fester
    Reihenfolge – auch bereits ausgefüllte. So bleiben Felder in
    Formular 3 sichtbar und behalten ihre Position.
    """

    validation_result = validate_model_specification(
        specification
    )

    display_mode: ProductDisplayMode = (
        "cards"
        if validation_result["errors"]
        or len(specification["products"]) <= card_limit
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

    validator_missing_inputs = list(validation_result["missing_inputs"])
    editable_inputs = [] if include_complete else list(
        validator_missing_inputs
    )

    if include_complete or validation_result["errors"]:
        registered_core = CORE_REGISTRY.get(
            specification["core_id"]
        )

        if registered_core is not None:
            core_definition = registered_core["definition"]
            forecast_fields = core_definition[
                "required_forecast_fields"
            ]
            fact_fields = list(
                core_definition["required_product_facts"]
            )
            global_fields = list(
                core_definition["required_global_facts"]
            )

            for component_id in specification["objective"][
                "component_ids"
            ]:
                component = OBJECTIVE_COMPONENTS.get(
                    component_id
                )
                if component is None:
                    continue

                _append_unique_fields(
                    fact_fields,
                    component["required_product_facts"],
                )
                _append_unique_fields(
                    global_fields,
                    component["required_global_facts"],
                )

            for constraint in specification["constraints"]:
                component = CONSTRAINT_COMPONENTS.get(
                    constraint["component_id"]
                )
                if component is None:
                    continue

                _append_unique_fields(
                    fact_fields,
                    component["required_product_facts"],
                )
                _append_unique_fields(
                    global_fields,
                    component["required_global_facts"],
                )

            def add_if_new(candidate: MissingInput) -> None:
                if not any(
                    input_key(item) == input_key(candidate)
                    for item in editable_inputs
                ):
                    editable_inputs.append(candidate)

            for product in specification["products"]:
                product_id = product["product_id"]

                for field in forecast_fields:
                    add_if_new({
                        "source": "forecast",
                        "field": field,
                        "product_id": product_id,
                    })

                for field in fact_fields:
                    add_if_new({
                        "source": "product_fact",
                        "field": field,
                        "product_id": product_id,
                    })

            for field in global_fields:
                add_if_new({
                    "source": "global_fact",
                    "field": field,
                })

            for index, constraint in enumerate(
                specification["constraints"]
            ):
                component_id = constraint["component_id"]
                component = CONSTRAINT_COMPONENTS.get(
                    component_id
                )
                if component is None:
                    continue

                for field in component["parameters"]:
                    add_if_new({
                        "source": "constraint_parameter",
                        "field": field,
                        "component_id": component_id,
                        "constraint_index": index,
                    })

    if include_complete:
        known_keys = {input_key(item) for item in editable_inputs}

        for missing_input in validator_missing_inputs:
            if input_key(missing_input) not in known_keys:
                editable_inputs.append(missing_input)

    for missing_input in editable_inputs:
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