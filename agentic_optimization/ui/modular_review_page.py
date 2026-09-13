"""
Streamlit-Darstellung für die Prüfung modularer Modelle.
"""

from typing import Any

import streamlit as st

from agentic_optimization.models.component_catalog import (
    build_component_catalog,
)
from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ModelSpecification,
)
from agentic_optimization.ui.modular_review_builder import (
    build_modular_review_data,
)


def _format_value(value: Any) -> str:
    """Formatiert einen Wert für die Review-Seite."""

    if isinstance(value, list):
        quantiles = []

        for item in value:
            if (
                isinstance(item, dict)
                and "level" in item
                and "value" in item
            ):
                quantiles.append(
                    f"Q{float(item['level']) * 100:g}: "
                    f"{item['value']}"
                )

        if quantiles:
            return ", ".join(quantiles)

        return ", ".join(str(item) for item in value)

    return str(value)


def _build_field_labels(
    core_id: str,
) -> dict[str, str]:
    """Lädt Feldbezeichnungen aus dem Bausteinkatalog."""

    catalog = build_component_catalog(core_id)
    fields = [
        *catalog["core"]["required_forecast_fields"],
        *catalog["core"]["required_product_facts"],
        *catalog["core"]["required_global_facts"],
    ]

    for component in catalog["objective_components"]:
        fields.extend(component["required_product_facts"])
        fields.extend(component["required_global_facts"])

    for constraint in catalog["constraint_components"]:
        fields.extend(constraint["required_product_facts"])
        fields.extend(constraint["required_global_facts"])

    return {
        field["name"]: field["label"]
        for field in fields
    }


def _build_constraint_parameter_labels(
    core_id: str,
) -> dict[str, dict[str, str]]:
    """Lädt Parameterlabels getrennt nach Constraint-Baustein."""

    catalog = build_component_catalog(core_id)

    return {
        constraint["component_id"]: {
            field["name"]: field["label"]
            for field in constraint["parameters"]
        }
        for constraint in catalog["constraint_components"]
    }


def render_modular_review_page(
    proposal: ComponentSelectionProposal,
    specification: ModelSpecification,
) -> bool:
    """
    Rendert Formular 4.

    True bedeutet, dass der Nutzer das Modell bestätigt hat.
    """

    review = build_modular_review_data(
        proposal,
        specification,
    )
    field_labels = _build_field_labels(
        review["core_id"]
    )

    constraint_parameter_labels = (
        _build_constraint_parameter_labels(
            review["core_id"]
        )
    )

    if review["validation_errors"]:
        for error in review["validation_errors"]:
            st.error(error)

    if review["missing_inputs"]:
        st.warning(
            "Das Modell enthält noch fehlende Eingaben."
        )

    with st.container(border=True):
        st.markdown("### 1. Grundmodell")
        st.markdown(f"**{review['core_name']}**")

        objective_label = (
            "Maximierung"
            if review["objective_sense"] == "maximize"
            else "Minimierung"
        )
        st.write(f"Zielrichtung: {objective_label}")

    with st.container(border=True):
        st.markdown("### 2. Eingabedaten")

        for product in review["products"]:
            with st.expander(
                f"Produkt: {product['product_id']}",
                expanded=True,
            ):
                st.markdown("#### Nachfrageprognose")

                for field_name, value in product[
                    "forecast"
                ].items():
                    label = field_labels.get(
                        field_name,
                        field_name,
                    )
                    st.markdown(
                        f"**{label}:** {_format_value(value)}"
                    )

                st.markdown("#### Wirtschaftliche Angaben")

                for field_name, value in product[
                    "facts"
                ].items():
                    label = field_labels.get(
                        field_name,
                        field_name,
                    )
                    st.markdown(
                        f"**{label}:** {_format_value(value)}"
                    )

        if review["global_facts"]:
            st.markdown("#### Modellübergreifende Angaben")

            for field_name, value in review[
                "global_facts"
            ].items():
                label = field_labels.get(
                    field_name,
                    field_name,
                )
                st.markdown(
                    f"**{label}:** {_format_value(value)}"
                )

    with st.container(border=True):
        st.markdown("### 3. Zusammensetzung des Modells")

        st.markdown("#### Zielfunktionsbausteine")

        for component in review[
            "objective_components"
        ]:
            effect = (
                "wirkt positiv"
                if component["effect"] == "benefit"
                else "wirkt als Kosten"
            )

            st.markdown(
                f"**{component['name']}** – {effect}"
            )
            st.caption(component["reason"])

        st.markdown("#### Betriebliche Rahmenbedingungen")

        if review["constraints"]:
            for constraint in review["constraints"]:
                st.markdown(f"**{constraint['name']}**")

                if constraint["product_id"]:
                    st.write(
                        "Produkt: "
                        f"{constraint['product_id']}"
                    )

                for parameter_name, value in constraint[
                    "parameters"
                ].items():
                    labels = constraint_parameter_labels.get(
                        constraint["component_id"],
                        {},
                    )
                    label = labels.get(
                        parameter_name,
                        parameter_name,
                    )
                    st.write(
                        f"{label}: {_format_value(value)}"
                    )

                st.caption(constraint["reason"])
        else:
            st.info(
                "Es wurden keine zusätzlichen betrieblichen "
                "Rahmenbedingungen ausgewählt."
            )

    if not review["ready_for_confirmation"]:
        st.error(
            "Das Modell kann noch nicht bestätigt werden."
        )
        return False

    confirmed = st.checkbox(
        "Ich habe Grundmodell, Eingabedaten und "
        "Modellbausteine geprüft.",
        key="confirm_modular_review",
    )

    return st.button(
        "Optimierung starten",
        type="primary",
        use_container_width=True,
        disabled=not confirmed,
        key="start_modular_optimization",
    )