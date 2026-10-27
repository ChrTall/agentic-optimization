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
    ModularResultValue,
    build_modular_result_data,
)
from agentic_optimization.i18n import t


def _format_result_value(value: int | float) -> str:
    """Formatiert Zahlen mit deutschem Dezimaltrennzeichen."""

    formatted = f"{float(value):,.2f}"

    return (
        formatted
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )


def _build_result_overview_rows(
    values: list[ModularResultValue],
) -> list[dict[str, str]]:
    """Gruppiert Ergebniswerte zu einer Zeile je Produkt."""

    rows_by_product: dict[
        str,
        dict[str, str],
    ] = {}

    for item in values:
        product_label = (
            item["product_id"]
            if item["product_id"] is not None
            else t("Gesamtmodell")
        )

        row = rows_by_product.setdefault(
            product_label,
            {
                "Produkt": product_label,
            },
        )
        row[item["label"]] = _format_result_value(
            item["value"]
        )

    return list(rows_by_product.values())


def render_modular_result_page(
    specification: ModelSpecification,
    result: OptimizationExecutionResult,
    *,
    explanation: str | None = None,
) -> bool:
    """
    Rendert das Ergebnis eines modularen Modells.

    True bedeutet, dass eine Erklärung angefordert wurde.
    """

    result_data = build_modular_result_data(
        specification,
        result,
    )

    st.subheader(t("Optimierungsergebnis"))

    if result_data["successful"]:
        st.success(result_data["status_label"])
    elif result_data["errors"]:
        st.error(result_data["status_label"])
    else:
        st.warning(result_data["status_label"])

    for error in result_data["errors"]:
        st.error(error)

    for warning in result_data["warnings"]:
        st.warning(warning)

    st.markdown(
        t("**Grundmodell:** {core_name}", core_name=result_data['core_name'])
    )
    st.markdown(
        t("**Berechnungsmethode:** {solution_method_label}", solution_method_label=result_data['solution_method_label'])
    )

    if result_data["objective_value"] is not None:
        st.metric(
            result_data["objective_label"],
            _format_result_value(
                result_data["objective_value"]
            )
            + " €",
        )

    if result_data["decisions"]:
        st.markdown(t("### Empfohlene Entscheidungen"))

        decision_product_ids = {
            decision["product_id"]
            for decision in result_data["decisions"]
            if decision["product_id"] is not None
        }

        if len(decision_product_ids) > 1:
            st.dataframe(
                _build_result_overview_rows(
                    result_data["decisions"]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            for decision in result_data["decisions"]:
                product_label = (
                    f" – {decision['product_id']}"
                    if decision["product_id"]
                    else ""
                )

                value = decision["value"]
                st.metric(
                    f"{decision['label']}{product_label}",
                    # Mengen in ganzen Einheiten ohne „,00“ anzeigen.
                    f"{int(value):,}".replace(",", ".")
                    if float(value).is_integer()
                    else _format_result_value(value),
                )

    if result_data["excluded_requirements"]:
        st.markdown(t("### Geltungsbereich der Empfehlung"))

        st.warning(
            t("Die folgenden Anforderungen wurden bei der Berechnung nicht berücksichtigt. Die Empfehlung gilt daher nur, wenn diese Vorgaben außer Acht gelassen werden können.")
        )

        for requirement in result_data[
            "excluded_requirements"
        ]:
            st.markdown(
                t("**Nicht berücksichtigt:** {description}", description=requirement['description'])
            )
            st.write(requirement["reason"])

    if result_data["metrics"]:
        st.markdown(t("### Weitere Kennzahlen"))

        metric_product_ids = {
            metric["product_id"]
            for metric in result_data["metrics"]
            if metric["product_id"] is not None
        }

        if len(metric_product_ids) > 1:
            metric_rows = _build_result_overview_rows(
                result_data["metrics"]
            )
        else:
            metric_rows = [
                {
                    "Produkt": (
                        metric["product_id"]
                        or t("Gesamtmodell")
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
            width="stretch",
            hide_index=True,
        )

    explanation_requested = False

    if result_data["successful"]:
        st.markdown(t("### Ergebnis verstehen"))

        if (
            isinstance(explanation, str)
            and explanation.strip()
        ):
            st.info(explanation.strip())
        else:
            st.write(
                t("Lassen Sie sich die Empfehlung und die wichtigsten Kennzahlen in verständlicher Form erklären.")
            )
            explanation_requested = st.button(
                t("Optimierungsergebnis erklären"),
                width="stretch",
                key="explain_modular_optimization_result",
            )

    with st.expander(t("Technische Details")):
        st.markdown(
            t("**Status:** {status}", status=result_data['status'])
        )
        st.markdown(
            t("**Lösungsmethode:** {solution_method}", solution_method=result_data['solution_method'])
        )

        if result_data["technical_representation"]:
            st.code(
                result_data["technical_representation"],
                language="text",
            )
        else:
            st.info(
                t("Für diese Berechnung ist keine technische Darstellung verfügbar.")
            )

    return bool(explanation_requested)