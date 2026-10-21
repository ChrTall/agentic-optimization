"""
Streamlit-Seite zur Erfassung mehrerer Produktkennungen.
"""
from typing import Any

import streamlit as st

from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ProductSpecification,
)
from agentic_optimization.ui.modular_product_list import (
    parse_dynamic_product_table,
)
from agentic_optimization.ui.modular_input_builder import (
    build_dynamic_product_table_schema,
)


def _build_dynamic_column_config(
    schema,
) -> dict[str, Any]:
    """Erzeugt die Streamlit-Spaltendefinitionen."""

    column_config: dict[str, Any] = {
        "product_id": st.column_config.TextColumn(
            "Produkt",
            required=True,
            help=(
                "Eindeutige Artikelnummer oder "
                "Produktbezeichnung."
            ),
        )
    }

    for field in [
        *schema["forecast_fields"],
        *schema["fact_fields"],
    ]:
        field_name = field["name"]
        label = field["label"]
        unit = field.get("unit")

        if unit:
            label = f"{label} ({unit})"

        if field["type"] in {
            "number",
            "integer",
        }:
            column_config[field_name] = (
                st.column_config.NumberColumn(
                    label,
                    required=field["required"],
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
                    "Format: 0.1:70; 0.5:100; 0.9:140"
                ).strip()

            column_config[field_name] = (
                st.column_config.TextColumn(
                    label,
                    required=field["required"],
                    help=help_text,
                )
            )

    return column_config


def render_dynamic_product_table_page(
    *,
    core_id: str,
    proposal: ComponentSelectionProposal,
    prefilled_rows: list[dict[str, Any]] | None = None,
    editor_version: int = 0,
) -> list[ProductSpecification] | None:
    """
    Erfasst Produktkennungen und Produktwerte in einer Tabelle.

    prefilled_rows enthält optional bereits (z. B. im Chat)
    erfasste Zeilen. Ändern sie sich, muss editor_version erhöht
    werden, damit der Tabelleneditor neu aufgebaut wird.

    None bedeutet, dass die Tabelle noch nicht erfolgreich
    abgesendet wurde.
    """

    schema = build_dynamic_product_table_schema(
        core_id,
        proposal,
    )
    field_names = [
        field["name"]
        for field in [
            *schema["forecast_fields"],
            *schema["fact_fields"],
        ]
    ]
    empty_row = {
        "product_id": "",
        **{
            field_name: None
            for field_name in field_names
        },
    }
    initial_rows = [
        {
            **empty_row,
            **{
                key: value
                for key, value in row.items()
                if key in empty_row
            },
        }
        for row in (prefilled_rows or [])
    ] or [empty_row]

    st.subheader("Angaben zur Berechnung")
    st.write(
        "Eine Zeile entspricht einem Produkt. Sie können "
        "weitere Zeilen hinzufügen oder vorhandene Zeilen "
        "löschen."
    )
    st.caption(
        "Vollständig leere Zeilen werden ignoriert. Zum "
        "Löschen markieren Sie die Zeile links und verwenden "
        "Sie das Löschsymbol oder die Entf-Taste."
    )

    edited_data = st.data_editor(
        initial_rows,
        column_config=_build_dynamic_column_config(
            schema
        ),
        column_order=[
            "product_id",
            *field_names,
        ],
        hide_index=False,
        width="stretch",
        num_rows="dynamic",
        key=(
            "dynamic_modular_product_table"
            if not editor_version
            else f"dynamic_modular_product_table_{editor_version}"
        ),
    )

    submitted = st.button(
        "Angaben prüfen",
        type="primary",
        width="stretch",
        key="submit_dynamic_modular_product_table",
    )

    if not submitted:
        return None

    if isinstance(edited_data, list):
        edited_rows = edited_data
    elif hasattr(edited_data, "to_dict"):
        edited_rows = edited_data.to_dict(
            orient="records"
        )
    else:
        st.error(
            "Die Produkttabelle konnte nicht verarbeitet "
            "werden."
        )
        return None

    result = parse_dynamic_product_table(
        edited_rows,
        schema,
    )

    if result["errors"]:
        for error in result["errors"]:
            st.error(error)

        return None

    return result["products"]