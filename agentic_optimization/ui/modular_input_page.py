"""
Streamlit-Darstellung von Formular 3 („Angaben zur Berechnung“).

Alle Eingabefelder des Modells bleiben sichtbar und behalten ihre Position,
auch nachdem sie ausgefüllt wurden. Produktnamen sind änderbar, und
zusätzliche kompatible Bausteine lassen sich in einem aufklappbaren Bereich
an- und abwählen.
"""

import hashlib
from typing import Any, Literal, TypedDict

import streamlit as st

from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.specification_updates import (
    list_optional_components,
)
from agentic_optimization.models.validator import (
    MissingInput,
    validate_model_specification,
)
from agentic_optimization.state import (
    ModularInputSubmission,
)
from agentic_optimization.ui.field_renderer import (
    parse_input_value,
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
    input_key,
)
from agentic_optimization.i18n import t


class ModularInputPageResult(TypedDict):
    """Was der Nutzer in Formular 3 ausgelöst hat."""

    action: Literal["submit", "continue", "edit"]
    submissions: list[ModularInputSubmission]
    renames: list[tuple[str, str]]
    active_component_ids: list[str] | None


def _existing_input_value(
    specification: ModelSpecification,
    input_definition: MissingInput,
):
    source = input_definition["source"]
    field_name = input_definition["field"]["name"]

    if source in {"forecast", "product_fact"}:
        product = next(
            (
                product
                for product in specification["products"]
                if product["product_id"]
                == input_definition["product_id"]
            ),
            None,
        )
        if product is None:
            return None

        container = (
            product.get("forecast") or {}
            if source == "forecast"
            else product.get("facts") or {}
        )
        return container.get(field_name)

    if source == "global_fact":
        return specification["global_facts"].get(field_name)

    if source == "constraint_parameter":
        index = input_definition["constraint_index"]
        return specification["constraints"][index][
            "parameters"
        ].get(field_name)

    return None


def _value_digest(value: Any) -> str:
    """Kurzer Fingerabdruck eines Werts für den Widget-Schlüssel.

    Ändert sich der Wert von außen (z. B. über den Chat), entsteht ein
    neues Widget mit dem neuen Wert statt eines veralteten Feldinhalts.
    """

    return hashlib.sha1(repr(value).encode("utf-8")).hexdigest()[:8]


def _render_input(
    specification: ModelSpecification,
    missing_input: MissingInput,
    *,
    key_prefix: str,
    missing_keys: set[tuple],
) -> RenderedModularInput:
    """Rendert genau ein Eingabefeld – ausgefüllt oder offen."""

    existing_value = _existing_input_value(
        specification,
        missing_input,
    )

    try:
        parse_input_value(
            missing_input["field"],
            existing_value,
        )
    except (TypeError, ValueError):
        existing_value = None

    field = missing_input["field"]

    if input_key(missing_input) in missing_keys and field["required"]:
        field = {**field, "label": t("{label} · fehlt noch", label=t(field["label"]))}

    raw_value = render_input_field(
        field,
        existing_value,
        key_prefix=f"{key_prefix}_{_value_digest(existing_value)}",
    )

    return {
        "missing_input": missing_input,
        "raw_value": raw_value,
    }


def _render_product_input_table(
    layout: ModularInputLayout,
    specification: ModelSpecification,
) -> list[RenderedModularInput]:
    """Rendert gemeinsame Produktfelder als editierbare Tabelle."""

    table_rows = build_product_input_table_rows(layout, specification)
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
            t("Produkt"),
            disabled=True,
        )
    }

    for missing_input in first_group_inputs:
        field = missing_input["field"]
        field_name = field["name"]

        if field["type"] in {"number", "integer"}:
            column_config[field_name] = st.column_config.NumberColumn(
                t(field["label"]),
                help=t(field.get("help")),
                min_value=field.get("min_value"),
                max_value=field.get("max_value"),
                step=field.get("step"),
            )
        else:
            help_text = t(field.get("help"))

            if field["type"] == "forecast_quantiles":
                help_text = (
                    f"{help_text or ''} "
                    "Format: 0.1: 70; 0.5: 100; 0.9: 140"
                ).strip()

            column_config[field_name] = st.column_config.TextColumn(
                t(field["label"]),
                help=help_text,
            )

    edited_data = st.data_editor(
        table_rows,
        column_config=column_config,
        column_order=["product_id", *field_names],
        disabled=["product_id"],
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        height=min(38 * len(table_rows) + 40, 600),
        key="modular_product_input_table",
    )

    if isinstance(edited_data, list):
        edited_rows = edited_data
    elif hasattr(edited_data, "to_dict"):
        edited_rows = edited_data.to_dict(orient="records")
    else:
        raise TypeError(
            t("Die bearbeitete Produkttabelle besitzt ein unbekanntes Datenformat.")
        )

    rows_by_product = {row["product_id"]: row for row in edited_rows}
    rendered_inputs: list[RenderedModularInput] = []

    for product_group in layout["product_groups"]:
        product_id = product_group["product_id"]
        row = rows_by_product.get(product_id)

        if row is None:
            raise ValueError(
                t("Die Tabellenzeile für '{product_id}' fehlt.", product_id=product_id)
            )

        for missing_input in [
            *product_group["forecast_inputs"],
            *product_group["fact_inputs"],
        ]:
            rendered_inputs.append(
                {
                    "missing_input": missing_input,
                    "raw_value": row.get(missing_input["field"]["name"]),
                }
            )

    return rendered_inputs


def _render_component_selection(
    specification: ModelSpecification,
) -> list[str] | None:
    """Aufklappbarer Bereich für zusätzliche Bausteine.

    Liefert die neue Auswahl, sobald der Nutzer etwas an- oder abwählt.
    """

    components = list_optional_components(specification)
    optional = [c for c in components if not c["locked"]]

    if not optional:
        return None

    active_count = sum(c["active"] for c in optional)

    with st.expander(
        t("➕ Zusätzliche Bausteine ({active_count} von {value3} aktiv)", active_count=active_count, value3=len(optional))
    ):
        st.caption(
            t("Optional: Ergänzen Sie weitere Kosten, Erlöse oder betriebliche Grenzen. Neue Bausteine erscheinen oben als Eingabefelder.")
        )
        selected: list[str] = []

        for kind, heading in (
            ("objective", t("Kosten und Erlöse")),
            ("constraint", t("Betriebliche Grenzen")),
        ):
            group = [c for c in components if c["kind"] == kind]

            if not group:
                continue

            st.markdown(f"**{heading}**")

            for component in group:
                checked = st.checkbox(
                    t(component["name"]),
                    value=component["active"],
                    disabled=component["locked"],
                    help=(
                        t("Wird vom Rechenmodell immer benötigt.")
                        if component["locked"]
                        else None
                    ),
                    key=(
                        "modular_component_toggle_"
                        f"{component['component_id']}_"
                        f"{int(component['active'])}"
                    ),
                )

                if checked or component["locked"]:
                    selected.append(component["component_id"])

    current = sorted(c["component_id"] for c in components if c["active"])

    if sorted(selected) != current:
        return selected

    return None


def render_modular_input_page(
    specification: ModelSpecification,
) -> ModularInputPageResult | None:
    """
    Rendert Formular 3.

    None bedeutet, dass der Nutzer nichts ausgelöst hat.
    """

    layout = build_modular_input_layout(
        specification,
        include_complete=True,
    )
    validation = validate_model_specification(specification)
    missing_keys = {
        input_key(missing_input)
        for missing_input in validation["missing_inputs"]
    }
    required_missing = sum(
        missing_input["field"]["required"]
        for missing_input in validation["missing_inputs"]
    )

    for error in layout["validation_errors"]:
        st.error(error)

    if validation["is_valid"]:
        st.success(
            t("Alle Pflichtangaben sind vollständig. Sie können Werte noch ändern, zusätzliche Bausteine ergänzen oder zur Überprüfung weitergehen.")
        )
    elif required_missing:
        st.info(
            t("Noch **{required_missing}** ", required_missing=required_missing)
            + (t("Angabe") if required_missing == 1 else t("Angaben"))
            + t(" offen. Tragen Sie sie hier ein oder nennen Sie sie im Chat – ausgefüllte Felder bleiben sichtbar und änderbar.")
        )

    use_product_table = can_render_product_input_table(layout)
    rendered_inputs: list[RenderedModularInput] = []
    renames: list[tuple[str, str]] = []

    if use_product_table:
        st.markdown(t("### Produktdaten"))
        st.caption(t("Eine Zeile entspricht einem Produkt."))
        rendered_inputs.extend(
            _render_product_input_table(layout, specification)
        )

    with st.container():
        if not use_product_table:
            for product_index, product_group in enumerate(
                layout["product_groups"]
            ):
                product_id = product_group["product_id"]

                with st.expander(t("Produkt: {product_id}", product_id=product_id), expanded=True):
                    new_name = st.text_input(
                        t("Produktname"),
                        value=product_id,
                        key=(
                            f"modular_product_name_{product_index}_"
                            f"{_value_digest(product_id)}"
                        ),
                    )

                    if new_name.strip() and new_name.strip() != product_id:
                        renames.append((product_id, new_name.strip()))

                    if product_group["forecast_inputs"]:
                        st.markdown(t("#### Nachfrageprognose"))

                        for missing_input in product_group["forecast_inputs"]:
                            rendered_inputs.append(
                                _render_input(
                                    specification,
                                    missing_input,
                                    key_prefix=(
                                        f"modular_forecast_{product_index}"
                                    ),
                                    missing_keys=missing_keys,
                                )
                            )

                    if product_group["fact_inputs"]:
                        st.markdown(t("#### Wirtschaftliche Angaben"))

                        for missing_input in product_group["fact_inputs"]:
                            rendered_inputs.append(
                                _render_input(
                                    specification,
                                    missing_input,
                                    key_prefix=(
                                        f"modular_fact_{product_index}"
                                    ),
                                    missing_keys=missing_keys,
                                )
                            )

        if layout["global_inputs"]:
            st.markdown(t("### Modellübergreifende Angaben"))

            for missing_input in layout["global_inputs"]:
                rendered_inputs.append(
                    _render_input(
                        specification,
                        missing_input,
                        key_prefix="modular_global",
                        missing_keys=missing_keys,
                    )
                )

        if layout["constraint_groups"]:
            st.markdown(t("### Betriebliche Rahmenbedingungen"))

            for constraint_group in layout["constraint_groups"]:
                component_id = constraint_group["component_id"]
                product_id = constraint_group.get("product_id")
                component_name = t(CONSTRAINT_COMPONENTS[component_id]["name"])
                product_suffix = (
                    f" – {product_id}"
                    if product_id and len(specification["products"]) > 1
                    else ""
                )

                st.markdown(f"**{component_name}{product_suffix}**")

                for missing_input in constraint_group["inputs"]:
                    rendered_inputs.append(
                        _render_input(
                            specification,
                            missing_input,
                            # Stabil über Baustein und Produkt statt über
                            # die Position: Beim Abwählen eines Bausteins
                            # verschieben sich die Indizes.
                            key_prefix=(
                                f"modular_constraint_{component_id}_"
                                f"{_value_digest(product_id)}"
                            ),
                            missing_keys=missing_keys,
                        )
                    )

        active_component_ids = _render_component_selection(specification)

        if active_component_ids is not None:
            return {
                "action": "edit",
                "submissions": [],
                "renames": renames,
                "active_component_ids": active_component_ids,
            }

        # Eingaben werden sofort übernommen: Sobald ein Feld geändert
        # wurde, geht der Wert an den Workflow, und der Chat stellt von
        # selbst die nächste Frage.
        changed, input_errors = _changed_submissions(
            specification, rendered_inputs
        )

        for error in input_errors:
            st.warning(error)

        st.caption(
            t("Eingaben werden sofort übernommen – der Assistent fragt danach automatisch die nächste offene Angabe ab.")
        )

        continue_clicked = False

        if validation["is_valid"] and not changed:
            continue_clicked = st.button(
                t("Weiter zur Überprüfung"),
                type="primary",
                width="stretch",
                key="continue_to_modular_review",
            )

    if changed or renames:
        return {
            "action": "submit",
            "submissions": changed,
            "renames": renames,
            "active_component_ids": None,
        }

    if continue_clicked:
        return {
            "action": "continue",
            "submissions": [],
            "renames": [],
            "active_component_ids": None,
        }

    return None


def _changed_submissions(
    specification: ModelSpecification,
    rendered_inputs: list[RenderedModularInput],
) -> tuple[list[ModularInputSubmission], list[str]]:
    """Ermittelt Felder, deren Wert sich gegenüber dem Modell geändert hat.

    Ungültige Eingaben (z. B. halb getippte Quantile) werden nicht
    übernommen, sondern als Hinweis zurückgegeben.
    """

    changed: list[ModularInputSubmission] = []
    errors: list[str] = []

    for rendered in rendered_inputs:
        missing_input = rendered["missing_input"]
        field = missing_input["field"]

        try:
            submissions = parse_modular_input_submissions(
                [rendered],
                require_all=False,
            )
        except (TypeError, ValueError) as error:
            errors.append(f"{t(field['label'])}: {error}")
            continue

        if not submissions:
            continue

        try:
            existing = parse_input_value(
                field,
                _existing_input_value(specification, missing_input),
            )
        except (TypeError, ValueError):
            existing = None

        if submissions[0]["value"] != existing:
            changed.extend(submissions)

    return changed, errors
