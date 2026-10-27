"""
Streamlit-Darstellung für die Prüfung modularer Modelle.
"""

from typing import Any

import streamlit as st

from agentic_optimization.models.component_catalog import (
    build_component_catalog,
)
from agentic_optimization.models.core_applicability import (
    CONDITION_LABELS,
    CORE_ASSUMPTION_CONFIRMATION_PREFIX,
)
from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ModelSpecification,
)
from agentic_optimization.ui.modular_review_builder import (
    ReviewConstraint,
    ReviewProduct,
    build_default_review_explanation,
    build_modular_review_data,
    build_modular_selection_issue_data,
)
from agentic_optimization.i18n import t


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


def _build_product_overview_rows(
    products: list[ReviewProduct],
    field_labels: dict[str, str],
) -> list[dict[str, str]]:
    """Erzeugt eine kompakte Tabellenzeile je Produkt."""

    rows: list[dict[str, str]] = []

    for product in products:
        row = {
            "Produkt": product["product_id"],
        }

        for field_name, value in product[
            "forecast"
        ].items():
            row[
                field_labels.get(
                    field_name,
                    field_name,
                )
            ] = _format_value(value)

        for field_name, value in product[
            "facts"
        ].items():
            row[
                field_labels.get(
                    field_name,
                    field_name,
                )
            ] = _format_value(value)

        rows.append(row)

    return rows


def _build_constraint_overview_rows(
    constraints: list[ReviewConstraint],
    parameter_labels: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Erzeugt eine kompakte Zeile je Nebenbedingung."""

    rows: list[dict[str, str]] = []

    for constraint in constraints:
        row = {
            "Nebenbedingung": constraint["name"],
            "Produkt": (
                constraint["product_id"]
                or t("Gesamtmodell")
            ),
        }
        labels = parameter_labels.get(
            constraint["component_id"],
            {},
        )

        for parameter_name, value in constraint[
            "parameters"
        ].items():
            row[
                labels.get(
                    parameter_name,
                    parameter_name,
                )
            ] = _format_value(value)

        rows.append(row)

    return rows


def _render_product_details(
    product: ReviewProduct,
    field_labels: dict[str, str],
) -> None:
    """Zeigt Forecast und wirtschaftliche Angaben eines Produkts."""

    forecast_column, facts_column = st.columns(2)

    with forecast_column:
        st.markdown(t("#### Nachfrageprognose"))

        if product["forecast"]:
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
        else:
            st.caption(
                t("Keine Nachfrageangaben vorhanden.")
            )

    with facts_column:
        st.markdown(t("#### Wirtschaftliche Angaben"))

        if product["facts"]:
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
        else:
            st.caption(
                t("Keine wirtschaftlichen Angaben vorhanden.")
            )


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
        field["name"]: t(field["label"])
        for field in fields
    }


def _build_constraint_parameter_labels(
    core_id: str,
) -> dict[str, dict[str, str]]:
    """Lädt Parameterlabels getrennt nach Constraint-Baustein."""

    catalog = build_component_catalog(core_id)

    return {
        constraint["component_id"]: {
            field["name"]: t(field["label"])
            for field in constraint["parameters"]
        }
        for constraint in catalog["constraint_components"]
    }


def render_modular_selection_issues(
    proposal: ComponentSelectionProposal,
) -> None:
    """Zeigt ungeklärte oder nicht unterstützte Anforderungen."""

    issue_data = build_modular_selection_issue_data(
        proposal
    )

    if (
        issue_data["selection_status"]
        == "needs_clarification"
    ):
        st.warning(
            t("Für die Modellzusammenstellung werden noch zusätzliche Angaben benötigt.")
        )
    elif (
        issue_data["selection_status"]
        == "unsupported"
    ):
        st.error(
            t("Mindestens eine Anforderung kann mit den verfügbaren Modellbausteinen derzeit nicht abgebildet werden.")
        )
    else:
        return

    st.caption(
        t("Erkanntes Grundmodell: {core_name}", core_name=issue_data['core_name'])
    )

    for requirement in issue_data[
        "unresolved_requirements"
    ]:
        st.markdown(
            f"**{requirement['description']}**"
        )
        st.write(requirement["reason"])
        st.caption(
            t("Quelle: ")
            + ", ".join(requirement["sources"])
        )


def render_modular_exclusion_confirmation(
    proposal: ComponentSelectionProposal,
) -> bool:
    """
    Fragt nach einer ausdrücklichen Bestätigung, bevor
    nicht unterstützte Anforderungen ausgeschlossen werden.
    """

    issue_data = build_modular_selection_issue_data(
        proposal
    )

    if issue_data["selection_status"] != "unsupported":
        return False

    st.warning(
        t("Sie können die Berechnung ohne die oben genannten Anforderungen fortsetzen. Diese Vorgaben werden dann nicht in der Empfehlung berücksichtigt und später deutlich als Ausschlüsse ausgewiesen.")
    )

    confirmed = st.checkbox(
        t("Ich bestätige, dass die genannten Anforderungen für diese Berechnung nicht berücksichtigt werden sollen."),
        key="confirm_modular_requirement_exclusions",
    )

    continue_without_requirements = st.button(
        t("Ohne diese Anforderungen fortfahren"),
        width="stretch",
        disabled=not confirmed,
        key="continue_without_unsupported_requirements",
    )

    return bool(
        confirmed
        and continue_without_requirements
    )


def render_modular_review_page(
    proposal: ComponentSelectionProposal,
    specification: ModelSpecification,
    *,
    explanation: str | None = None,
    pending_core_assumptions: list[str] | None = None,
    core_assumption_evidence: list[dict[str, str]] | None = None,
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

    explanation_text = (
        explanation.strip()
        if isinstance(explanation, str)
        and explanation.strip()
        else build_default_review_explanation(
            review,
            pending_core_assumptions=[
                t(CONDITION_LABELS[condition_id])
                for condition_id in (
                    pending_core_assumptions or []
                )
            ],
        )
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
            t("Das Modell enthält noch fehlende Eingaben.")
        )

    with st.container(border=True):
        st.markdown(t("### 1. Was wird berechnet?"))

        st.markdown(f"**{review['core_name']}**")
        st.write(review["core_description"])

        st.markdown(t("#### Kurz erklärt"))
        if pending_core_assumptions:
            st.warning(
                t("Die Modellwahl ist vorläufig: Nicht alle Voraussetzungen des Grundmodells sind bereits bestätigt.")
            )
        st.info(explanation_text)

    with st.container(border=True):
        st.markdown(t("### 2. Berechnungsgrundlage"))

        products = review["products"]

        if len(products) == 1:
            product = products[0]

            with st.expander(
                t("Produkt: {product_id}", product_id=product['product_id']),
                expanded=True,
            ):
                _render_product_details(
                    product,
                    field_labels,
                )
        else:
            st.caption(
                t("{value0} Produkte werden in dieser Berechnung berücksichtigt.", value0=len(products))
            )

            st.dataframe(
                _build_product_overview_rows(
                    products,
                    field_labels,
                ),
                width="stretch",
                hide_index=True,
            )

            product_ids = [
                product["product_id"]
                for product in products
            ]

            selected_product_id = st.selectbox(
                t("Produktdetails anzeigen"),
                product_ids,
                key="modular_review_product_details",
            )

            selected_product = next(
                product
                for product in products
                if product["product_id"]
                == selected_product_id
            )

            with st.expander(
                t("Details: {selected_product_id}", selected_product_id=selected_product_id),
                expanded=True,
            ):
                _render_product_details(
                    selected_product,
                    field_labels,
                )

        if review["global_facts"]:
            st.markdown(t("#### Modellübergreifende Angaben"))

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
        st.markdown(
            t("### 3. Berücksichtigte Modellbausteine")
        )

        core_column, objective_column, constraint_column = (
            st.columns(3)
        )

        with core_column:
            st.markdown(t("**Grundmodell**"))
            st.write(review["core_name"])

        with objective_column:
            st.markdown(t("**Optimierungsziel**"))
            st.write(review["objective_summary"])

        with constraint_column:
            st.markdown(t("**Nebenbedingungen**"))

            if review["constraints"]:
                for constraint in review["constraints"]:
                    st.markdown(
                        f"- {constraint['name']}"
                    )
            else:
                st.write(t("Keine zusätzlichen Vorgaben"))

        st.markdown(
            t("#### Betriebliche Rahmenbedingungen")
        )

        if review["constraints"]:
            constraint_product_ids = {
                constraint["product_id"]
                for constraint in review["constraints"]
                if constraint["product_id"] is not None
            }

            if len(constraint_product_ids) > 1:
                st.dataframe(
                    _build_constraint_overview_rows(
                        review["constraints"],
                        constraint_parameter_labels,
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                for constraint in review["constraints"]:
                    with st.container(border=True):
                        st.markdown(
                            f"**{constraint['name']}**"
                        )

                        if constraint["product_id"]:
                            st.write(
                                t("Produkt: {product_id}", product_id=constraint['product_id'])
                            )

                        for (
                            parameter_name,
                            value,
                        ) in constraint[
                            "parameters"
                        ].items():
                            labels = (
                                constraint_parameter_labels.get(
                                    constraint["component_id"],
                                    {},
                                )
                            )
                            label = labels.get(
                                parameter_name,
                                parameter_name,
                            )
                            st.write(
                                f"{label}: "
                                f"{_format_value(value)}"
                            )
        else:
            st.info(
                t("Es wurden keine zusätzlichen betrieblichen Rahmenbedingungen berücksichtigt.")
            )

        with st.expander(
            t("Warum wurden diese Bausteine ausgewählt?")
        ):
            st.markdown(
                t("**Kosten- und Erlösbausteine**")
            )

            for component in review[
                "objective_components"
            ]:
                effect = (
                    t("wirkt positiv")
                    if component["effect"] == "benefit"
                    else t("wirkt als Kosten")
                )

                st.markdown(
                    f"**{component['name']}** – {effect}"
                )
                st.caption(component["reason"])

            if review["constraints"]:
                st.markdown(t("**Nebenbedingungen**"))

                for constraint in review["constraints"]:
                    st.markdown(
                        f"**{constraint['name']}**"
                    )
                    st.caption(constraint["reason"])

    with st.container(border=True):
        st.markdown(
            t("### 4. Modellannahmen und Hinweise")
        )
        st.write(
            t("Die Berechnung basiert auf folgenden fachlichen Annahmen:")
        )

        for assumption in review["assumptions"]:
            st.markdown(f"- {assumption}")

        if review["excluded_requirements"]:
            st.warning(
                t("Die folgenden Anforderungen werden in dieser Berechnung nicht berücksichtigt. Die Ergebnisse gelten daher nur ohne diese Vorgaben.")
            )

            for requirement in review[
                "excluded_requirements"
            ]:
                st.markdown(
                    t("**Nicht berücksichtigt:** {description}", description=requirement['description'])
                )
                st.caption(requirement["reason"])

    if not review["ready_for_confirmation"]:
        st.error(
            t("Das Modell kann noch nicht bestätigt werden.")
        )
        return False

    assumptions_confirmed = _render_core_assumption_confirmations(
        pending_core_assumptions or [],
        core_assumption_evidence or [],
    )

    confirmed = st.checkbox(
        t("Ich habe Grundmodell, Eingabedaten und Modellbausteine geprüft."),
        key="confirm_modular_review",
    )

    return st.button(
        t("Optimierung starten"),
        type="primary",
        width="stretch",
        disabled=not (confirmed and assumptions_confirmed),
        key="start_modular_optimization",
    )


def _render_core_assumption_confirmations(
    pending_core_assumptions: list[str],
    core_assumption_evidence: list[dict[str, str]],
) -> bool:
    """Zeigt LLM-Textbelege getrennt von der Nutzerbestätigung."""

    supported_evidence = [
        item for item in core_assumption_evidence
        if item.get("status") == "supported"
        and item.get("condition_id") in pending_core_assumptions
        and item.get("condition_id") in CONDITION_LABELS
        and isinstance(item.get("excerpt"), str)
        and item["excerpt"].strip()
    ]
    if supported_evidence:
        st.markdown(t("#### Aus Ihrer Beschreibung abgeleitet"))
        st.caption(
            t("Diese Einschätzungen stammen vom Sprachmodell. Bitte prüfen Sie, ob der jeweilige Textbeleg die Voraussetzung tatsächlich stützt.")
        )
        for item in supported_evidence:
            st.markdown(
                f"**{t(CONDITION_LABELS[item['condition_id']])}**"
            )
            st.write(t("Ihre Angabe: „{excerpt}“", excerpt=item['excerpt']))
    contrary_evidence = [
        item for item in core_assumption_evidence
        if item.get("status") == "contradicted"
        and item.get("condition_id") in pending_core_assumptions
        and item.get("condition_id") in CONDITION_LABELS
        and isinstance(item.get("excerpt"), str)
        and item["excerpt"].strip()
    ]
    if contrary_evidence:
        st.warning(
            t("Möglicher Widerspruch zu einer Modellannahme: Bitte prüfen Sie die folgenden Textstellen besonders sorgfältig. Bestätigen Sie die Annahme nur, wenn sie für Ihre Situation tatsächlich zutrifft.")
        )
        for item in contrary_evidence:
            st.markdown(f"**{t(CONDITION_LABELS[item['condition_id']])}**")
            st.write(t("Ihre Angabe: „{excerpt}“", excerpt=item['excerpt']))
    assumptions_confirmed = True
    if pending_core_assumptions:
        st.warning(
            t("Bitte bestätigen Sie jede dieser Voraussetzungen einzeln. Eine automatische Einschätzung ist noch keine Bestätigung:")
        )
        for condition_id in pending_core_assumptions:
            condition_confirmed = st.checkbox(
                t(CONDITION_LABELS[condition_id]),
                key=(
                    CORE_ASSUMPTION_CONFIRMATION_PREFIX
                    + condition_id
                ),
            )
            assumptions_confirmed = (
                assumptions_confirmed and condition_confirmed
            )

    return assumptions_confirmed
