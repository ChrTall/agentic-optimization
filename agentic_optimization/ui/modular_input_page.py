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
    build_modular_input_layout,
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

    if layout["display_mode"] == "table":
        st.info(
            "Für viele Produkte ist eine tabellarische "
            "Bearbeitung vorgesehen. Bis zur vollständigen "
            "Tabellenanbindung werden die Produkte einzeln "
            "gruppiert dargestellt."
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

    with st.form("modular_model_input_form"):
        for product_group in layout["product_groups"]:
            product_id = product_group["product_id"]

            with st.expander(
                f"Produkt: {product_id}",
                expanded=True,
            ):
                if product_group["forecast_inputs"]:
                    st.markdown("#### Nachfrageprognose")

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
                    st.markdown("#### Wirtschaftliche Angaben")

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
