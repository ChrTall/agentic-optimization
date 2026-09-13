"""
Streamlit-Darstellung modularer Optimierungsergebnisse.
"""

import streamlit as st

from agentic_optimization.models.execution import (
    OptimizationExecutionResult,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.ui.modular_result_builder import (
    build_modular_result_data,
)


def _format_result_value(value: int | float) -> str:
    """Formatiert Zahlen mit deutschem Dezimaltrennzeichen."""

    formatted = f"{float(value):,.2f}"

    return (
        formatted
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )


def render_modular_result_page(
    specification: ModelSpecification,
    result: OptimizationExecutionResult,
) -> None:
    """Rendert das Ergebnis eines modularen Modells."""

    result_data = build_modular_result_data(
        specification,
        result,
    )

    st.subheader("Optimierungsergebnis")

    if result_data["successful"]:
        st.success(result_data["status_label"])
    elif result_data["errors"]:
        st.error(result_data["status_label"])
    else:
        st.warning(result_data["status_label"])

    for error in result_data["errors"]:
        st.error(error)

    st.markdown(
        f"**Grundmodell:** {result_data['core_name']}"
    )
    st.markdown(
        "**Berechnungsmethode:** "
        f"{result_data['solution_method_label']}"
    )

    if result_data["objective_value"] is not None:
        st.metric(
            "Zielfunktionswert",
            _format_result_value(
                result_data["objective_value"]
            ),
        )

    if result_data["decisions"]:
        st.markdown("### Empfohlene Entscheidungen")

        for decision in result_data["decisions"]:
            product_label = (
                f" – {decision['product_id']}"
                if decision["product_id"]
                else ""
            )

            st.metric(
                f"{decision['label']}{product_label}",
                _format_result_value(
                    decision["value"]
                ),
            )

    if result_data["metrics"]:
        st.markdown("### Weitere Kennzahlen")

        metric_rows = [
            {
                "Produkt": (
                    metric["product_id"] or "Gesamtmodell"
                ),
                "Kennzahl": metric["label"],
                "Wert": _format_result_value(
                    metric["value"]
                ),
            }
            for metric in result_data["metrics"]
        ]

        st.dataframe(
            metric_rows,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Technische Details"):
        st.markdown(
            f"**Status:** {result_data['status']}"
        )
        st.markdown(
            "**Lösungsmethode:** "
            f"{result_data['solution_method']}"
        )

        if result_data["technical_representation"]:
            st.code(
                result_data["technical_representation"],
                language="text",
            )
        else:
            st.info(
                "Für diese Berechnung ist keine technische "
                "Darstellung verfügbar."
            )
