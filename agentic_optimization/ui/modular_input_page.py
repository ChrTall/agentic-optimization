"""
Streamlit-Darstellung für fehlende modulare Modelleingaben.
"""

import streamlit as st

from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.validator import (
    MissingInput,
)
from agentic_optimization.state import (
    ModularInputSubmission,
)
from agentic_optimization.ui.field_renderer import (
    render_input_field,
)
from agentic_optimization.ui.modular_form_parser import (
    RenderedModularInput,
    parse_modular_input_submissions,
)
from agentic_optimization.ui.modular_input_builder import (
    ModularInputLayout,
    build_modular_input_layout,
    build_product_input_table_rows,
    can_render_product_input_table,
)


def _render_missing_input(
    missing_input: MissingInput,
    *,
    key_prefix: str,
) -> RenderedModularInput:
    """Rendert genau eine fehlende Eingabe."""

    raw_value = render_input_field(
        missing_input["field"],
        None,
        key_prefix=key_prefix,
    )

    return {
        "missing_input": missing_input,
        "raw_value": raw_value,
    }


def _render_product_input_table(
    layout: ModularInputLayout,
) -> list[RenderedModularInput]:
    """Rendert gemeinsame offene Felder als editierbare Tabelle."""

    table_rows = build_product_input_table_rows(
        layout
    )
    first_group = layout["product_groups"][0]
    first_group_inputs = [
        *first_group["forecast_inputs"],
        *first_group["fact_inputs"],
    ]

    field_names = [
        missing_input["field"]["name"]
        for missing_input in first_group_inputs
    ]

    column_config = {
        "product_id": st.column_config.TextColumn(
            "Produkt",
            disabled=True,
        )
    }

    for missing_input in first_group_inputs:
        field = missing_input["field"]
        field_name = field["name"]

        if field["type"] in {
            "number",
            "integer",
        }:
            column_config[field_name] = (
                st.column_config.NumberColumn(
                    field["label"],
                    help=field.get("help"),
                    min_value=field.get("min_value"),
                    max_value=field.get("max_value"),
                    step=field.get("step"),
                )
            )
        else:
            help_text = field.get("help")

            if field["type"] == "forecast_quantiles":
                help_text = (
                    f"{help_text or ''} "
                    "Format: 0.1: 70; 0.5: 100; "
                    "0.9: 140"
                ).strip()

            column_config[field_name] = (
                st.column_config.TextColumn(
                    field["label"],
                    help=help_text,
                )
            )

    edited_data = st.data_editor(
        table_rows,
        column_config=column_config,
        column_order=[
            "product_id",
            *field_names,
        ],
        disabled=["product_id"],
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        height=min(
            38 * len(table_rows) + 40,
            600,
        ),
        key="modular_product_input_table",
    )

    if isinstance(edited_data, list):
        edited_rows = edited_data
    elif hasattr(edited_data, "to_dict"):
        edited_rows = edited_data.to_dict(
            orient="records"
        )
    else:
        raise TypeError(
            "Die bearbeitete Produkttabelle besitzt "
            "ein unbekanntes Datenformat."
        )

    rows_by_product = {
        row["product_id"]: row
        for row in edited_rows
    }
    rendered_inputs: list[RenderedModularInput] = []

    for product_group in layout["product_groups"]:
        product_id = product_group["product_id"]
        row = rows_by_product.get(product_id)

        if row is None:
            raise ValueError(
                f"Die Tabellenzeile für '{product_id}' fehlt."
            )

        for missing_input in [
            *product_group["forecast_inputs"],
            *product_group["fact_inputs"],
        ]:
            field_name = missing_input["field"]["name"]

            rendered_inputs.append(
                {
                    "missing_input": missing_input,
                    "raw_value": row.get(field_name),
                }
            )

    return rendered_inputs


def render_modular_input_page(
    specification: ModelSpecification,
) -> list[ModularInputSubmission] | None:
    """
    Rendert Formular 3.

    None bedeutet, dass das Formular noch nicht abgesendet wurde.
    """

    layout = build_modular_input_layout(
        specification
    )

    for error in layout["validation_errors"]:
        st.error(error)

    use_product_table = can_render_product_input_table(
        layout
    )

    if (
        layout["display_mode"] == "table"
        and not use_product_table
    ):
        st.info(
            "Die Produkte besitzen unterschiedliche offene "
            "Felder und werden deshalb einzeln dargestellt."
        )
    if not (
        layout["product_groups"]
        or layout["global_inputs"]
        or layout["constraint_groups"]
    ):
        st.success(
            "Alle benötigten Modelleingaben sind vollständig."
        )
        return None

    rendered_inputs: list[
        RenderedModularInput
    ] = []

    # Die editierbare Tabelle liegt bewusst außerhalb des
    # Formulars. Dadurch übernimmt Streamlit auch den zuletzt
    # bearbeiteten Tabellenwert vor dem Absenden zuverlässig.
    if use_product_table:
        st.markdown("### Produktdaten")
        st.caption(
            "Eine Zeile entspricht einem Produkt. "
            "Die Produktkennung kann nicht verändert "
            "werden."
        )
        rendered_inputs.extend(
            _render_product_input_table(layout)
        )

    with st.form("modular_model_input_form"):
        if not use_product_table:
            for product_group in layout[
                "product_groups"
            ]:
                product_id = product_group["product_id"]

                with st.expander(
                    f"Produkt: {product_id}",
                    expanded=True,
                ):
                    if product_group["forecast_inputs"]:
                        st.markdown(
                            "#### Nachfrageprognose"
                        )

                        for missing_input in product_group[
                            "forecast_inputs"
                        ]:
                            rendered_inputs.append(
                                _render_missing_input(
                                    missing_input,
                                    key_prefix=(
                                        "modular_forecast_"
                                        f"{product_id}"
                                    ),
                                )
                            )

                    if product_group["fact_inputs"]:
                        st.markdown(
                            "#### Wirtschaftliche Angaben"
                        )

                        for missing_input in product_group[
                            "fact_inputs"
                        ]:
                            rendered_inputs.append(
                                _render_missing_input(
                                    missing_input,
                                    key_prefix=(
                                        "modular_fact_"
                                        f"{product_id}"
                                    ),
                                )
                            )

        if layout["global_inputs"]:
            st.markdown("### Modellübergreifende Angaben")

            for missing_input in layout["global_inputs"]:
                rendered_inputs.append(
                    _render_missing_input(
                        missing_input,
                        key_prefix="modular_global",
                    )
                )

        if layout["constraint_groups"]:
            st.markdown(
                "### Betriebliche Rahmenbedingungen"
            )

            for constraint_group in layout[
                "constraint_groups"
            ]:
                component_id = constraint_group[
                    "component_id"
                ]
                constraint_index = constraint_group[
                    "constraint_index"
                ]

                st.markdown(f"**{component_id}**")

                for missing_input in constraint_group[
                    "inputs"
                ]:
                    rendered_inputs.append(
                        _render_missing_input(
                            missing_input,
                            key_prefix=(
                                "modular_constraint_"
                                f"{constraint_index}"
                            ),
                        )
                    )

        submitted = st.form_submit_button(
            "Angaben prüfen",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return None

    return parse_modular_input_submissions(
        rendered_inputs
    )
