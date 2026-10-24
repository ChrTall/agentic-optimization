"""
Streamlit-Darstellung für fehlende modulare Modelleingaben.
"""

from copy import deepcopy
from functools import partial
import math

import streamlit as st

from agentic_optimization.ui.modular_components import (
    add_component, component_options, component_is_active, input_address,
    remove_component, removable_objectives, update_model_draft,
)

from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
    OBJECTIVE_COMPONENTS,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.specification_updates import apply_modular_input_value
from agentic_optimization.models.product_names import clean_product_name, product_name
from agentic_optimization.models.validator import (
    MissingInput,
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
)


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
            product.get("forecast", {})
            if source == "forecast"
            else product.get("facts", {})
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


def _draft_key(specification, item):
    return input_address(specification, item)


def _draft_value(specification, item, widget_key=None):
    key = _draft_key(specification, item)
    values = st.session_state.setdefault("modular_input_drafts", {})
    known = st.session_state.setdefault("modular_input_known_values", {})
    existing = _existing_input_value(specification, item)
    if key not in known or known[key] != existing:
        values[key] = deepcopy(existing)
        known[key] = deepcopy(existing)
        if widget_key:
            st.session_state.pop(widget_key, None)
    return values.get(key)


def _sync_form_values(changes):
    """Widget-Callbacks laufen vor dem Chat: nur formal geprüfte Werte übernehmen."""
    graph = st.session_state.get("graph_state") or {}
    specification = graph.get("modular_model", {}).get("specification")
    if specification is None:
        return
    updated = specification
    for item, raw in changes:
        address = input_address(specification, item)
        try:
            value = parse_input_value(item["field"], raw)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("Bitte geben Sie eine endliche Zahl ein.")
            if item["field"]["type"] == "number_list" and any(not math.isfinite(v) for v in value):
                raise ValueError("Bitte geben Sie endliche Zahlen ein.")
        except (TypeError, ValueError):
            # Eine ungültige Korrektur darf keinen früheren gültigen Wert
            # als vermeintlich ausgefüllte Angabe im Chat zurücklassen.
            value = None
        updated = apply_modular_input_value(updated, item, value)
        st.session_state.setdefault("modular_input_drafts", {})[address] = raw
        st.session_state.setdefault("modular_input_known_values", {})[address] = deepcopy(value)
    if updated != specification:
        st.session_state["graph_state"] = update_model_draft(graph, updated)


def _sync_widget_input(item, widget_key):
    _sync_form_values([(item, st.session_state.get(widget_key))])


def _sync_product_name(product_id, key):
    graph = st.session_state["graph_state"]
    specification = deepcopy(graph["modular_model"]["specification"])
    product = next(p for p in specification["products"] if p["product_id"] == product_id)
    name = clean_product_name(st.session_state.get(key))
    if name:
        product["display_name"] = name
    else:
        product.pop("display_name", None)
    st.session_state["graph_state"] = update_model_draft(graph, specification)


def _render_product_names(specification):
    with st.expander("Produktnamen prüfen oder bearbeiten", expanded=False):
        st.caption("Die Namen werden in allen weiteren Schritten angezeigt. "
                   "Bitte prüfen Sie automatisch übernommene Bezeichnungen.")
        known = st.session_state.setdefault("modular_product_name_known", {})
        for index, product in enumerate(specification["products"], 1):
            product_id = product["product_id"]
            key = f"modular_product_name_{product_id}"
            value = product.get("display_name")
            if product_id in known and known[product_id] != value:
                st.session_state.pop(key, None)
            known[product_id] = value
            st.text_input(f"Produktname {index}", value=value or product_name(specification, product_id),
                          key=key, max_chars=120,
                          on_change=partial(_sync_product_name, product_id, key))


def _sync_product_table(layout):
    """Übernimmt abgeschlossene Zelländerungen aus dem Tabelleneditor."""
    rows = st.session_state.get("modular_table_base", [])
    edits = st.session_state.get("modular_product_input_table", {}).get("edited_rows", {})
    changes = []
    groups = {group["product_id"]: group for group in layout["product_groups"]}
    for row_index, fields in edits.items():
        row = rows[int(row_index)]
        group = groups[row["product_id"]]
        for item in [*group["forecast_inputs"], *group["fact_inputs"]]:
            name = item["field"]["name"]
            if name in fields:
                changes.append((item, fields[name]))
    _sync_form_values(changes)


def _render_missing_input(
    specification: ModelSpecification,
    missing_input: MissingInput,
    *,
    key_prefix: str,
) -> RenderedModularInput:
    """Rendert genau eine fehlende Eingabe."""

    existing_value = _draft_value(
        specification, missing_input,
        f"{key_prefix}_{missing_input['field']['name']}",
    )

    try:
        existing_value = parse_input_value(
            missing_input["field"],
            existing_value,
        )
    except (TypeError, ValueError):
        existing_value = None

    raw_value = render_input_field(
        missing_input["field"],
        existing_value,
        key_prefix=key_prefix,
        on_change=partial(_sync_widget_input, missing_input,
                          f"{key_prefix}_{missing_input['field']['name']}"),
    )
    st.session_state["modular_input_drafts"][_draft_key(specification, missing_input)] = raw_value
    try:
        parse_input_value(missing_input["field"], raw_value)
    except (TypeError, ValueError) as error:
        st.error(str(error))

    return {
        "missing_input": missing_input,
        "raw_value": raw_value,
    }


def _render_product_input_table(
    layout: ModularInputLayout,
    specification: ModelSpecification | None = None,
) -> list[RenderedModularInput]:
    """Rendert gemeinsame offene Felder als editierbare Tabelle."""

    table_rows = build_product_input_table_rows(
        layout, specification
    )
    if specification is not None:
        for group, row in zip(layout["product_groups"], table_rows):
            row["product_name"] = product_name(specification, group["product_id"])
            for item in [*group["forecast_inputs"], *group["fact_inputs"]]:
                value = _draft_value(specification, item)
                if item["field"]["type"] == "forecast_quantiles" and isinstance(value, list):
                    value = "; ".join(f"{q['level']}: {q['value']}" for q in value)
                elif item["field"]["type"] == "number_list" and isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                row[item["field"]["name"]] = value
        if st.session_state.get("modular_table_base") != table_rows:
            st.session_state.pop("modular_product_input_table", None)
            st.session_state["modular_table_base"] = deepcopy(table_rows)
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
        "product_id": None if specification is not None else st.column_config.TextColumn(
            "Produkt",
            disabled=True,
        )
    }
    if specification is not None:
        column_config["product_name"] = st.column_config.TextColumn("Produkt", disabled=True)

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
            "product_name" if specification is not None else "product_id",
            *field_names,
        ],
        disabled=["product_id", "product_name"] if specification is not None else ["product_id"],
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        height=min(
            38 * len(table_rows) + 40,
            600,
        ),
        key="modular_product_input_table",
        on_change=partial(_sync_product_table, layout),
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
            if specification is not None:
                st.session_state["modular_input_drafts"][_draft_key(specification, missing_input)] = row.get(field_name)
                try:
                    parse_input_value(missing_input["field"], row.get(field_name))
                except (TypeError, ValueError) as error:
                    st.error(f"Produkt {product_id}: {error}")

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
        specification, include_completed=True
    )
    _render_product_names(specification)

    for error in layout["validation_errors"]:
        st.error(error)

    additional_layout = deepcopy(layout)
    base_specification = deepcopy(specification)
    for component_id in removable_objectives(specification):
        base_specification["objective"]["component_ids"].remove(component_id)
    base_specification["constraints"] = []
    base_layout = build_modular_input_layout(base_specification, include_completed=True)
    base_addresses = {input_address(base_specification, item) for item in _layout_inputs(base_layout)}
    for group in layout["product_groups"]:
        for name in ("forecast_inputs", "fact_inputs"):
            group[name] = [item for item in group[name]
                           if input_address(specification, item) in base_addresses]
    layout["global_inputs"] = [item for item in layout["global_inputs"]
                               if input_address(specification, item) in base_addresses]
    layout["constraint_groups"] = []

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
            _render_product_input_table(layout, specification)
        )

    # Die Eingabefelder werden bewusst nicht in einem Streamlit-
    # Formular gebündelt. Dadurch ist der aktuelle Widgetwert bereits
    # im Session State angekommen, bevor der Prüfbutton ausgewertet
    # wird. Andernfalls konnte insbesondere bei Zahlenfeldern der
    # zuletzt eingegebene Wert erst beim zweiten Absenden vorliegen.
    with st.container():
        if not use_product_table:
            for product_group in layout[
                "product_groups"
            ]:
                product_id = product_group["product_id"]

                with st.expander(
                    f"Produkt: {product_name(specification, product_id)}",
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
                                    specification,
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
                                    specification,
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
                        specification,
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

                component_name = CONSTRAINT_COMPONENTS[
                    component_id
                ]["name"]

                product_id = constraint_group["product_id"]
                st.markdown(f"**{component_name}**" + (f" – {product_name(specification, product_id)}" if product_id else ""))

                for missing_input in constraint_group[
                    "inputs"
                ]:
                    rendered_inputs.append(
                        _render_missing_input(
                            specification,
                            missing_input,
                            key_prefix=(
                                "modular_constraint_"
                                f"{component_id}_{product_id}"
                            ),
                        )
                    )

        rendered_inputs.extend(_render_additional_components(
            specification, additional_layout, base_addresses,
        ))
        confirmed = st.checkbox(
            "Ich habe die zusätzlichen Möglichkeiten geprüft. Alle für meinen "
            "Fall relevanten Angaben sind berücksichtigt.",
            key="modular_additions_confirmed",
        )
        submitted = st.button(
            "Weiter zur Überprüfung",
            type="primary",
            width="stretch",
            key="submit_modular_model_inputs",
            disabled=not confirmed,
        )

    if not submitted or not confirmed:
        return None

    return parse_modular_input_submissions(
        rendered_inputs
    )


def _layout_inputs(layout):
    return [*[item for group in layout["product_groups"]
              for item in [*group["forecast_inputs"], *group["fact_inputs"]]],
            *layout["global_inputs"],
            *[item for group in layout["constraint_groups"] for item in group["inputs"]]]


def _render_additional_components(specification, layout, base_addresses):
    rendered = []
    seen = set(base_addresses)
    known = st.session_state.setdefault("modular_component_known_selection", {})
    options = component_options(specification, include_active=True)
    with st.expander("Weitere Angaben berücksichtigen", expanded=False):
        st.caption("Haken Sie die passenden Angaben an. Die Eingabefelder erscheinen "
                   "direkt darunter. Zum Entfernen nehmen Sie den Haken wieder heraus.")
        if not options:
            st.info("Für dieses Modell sind keine zusätzlichen Angaben verfügbar.")
        for option in options:
            active = component_is_active(specification, option)
            key = "modular_component_" + option["option_id"]
            if key not in st.session_state or known.get(option["option_id"]) != active:
                st.session_state[key] = active
                known[option["option_id"]] = active
            if option["product_id"] is not None:
                st.caption(f"Produkt: {product_name(specification, option['product_id'])}")
            label = (OBJECTIVE_COMPONENTS if option["kind"] == "objective" else CONSTRAINT_COMPONENTS)[option["component_id"]]["name"]
            selected = st.checkbox(label, key=key, help=option["help"] or None)
            if selected != active:
                updated = (add_component(specification, option["option_id"]) if selected
                           else remove_component(specification, option))
                st.session_state["graph_state"] = update_model_draft(st.session_state["graph_state"], updated)
                st.session_state["modular_additions_confirmed"] = False
                st.rerun()
                return rendered
            if not active:
                continue
            registry = OBJECTIVE_COMPONENTS if option["kind"] == "objective" else CONSTRAINT_COMPONENTS
            definition = registry[option["component_id"]]
            product_fields = {f["name"] for f in definition["required_product_facts"]}
            global_fields = {f["name"] for f in definition["required_global_facts"]}
            for item in _layout_inputs(layout):
                address = input_address(specification, item)
                if address in seen:
                    continue
                source = item["source"]
                belongs = (source == "product_fact" and item["field"]["name"] in product_fields
                           or source == "global_fact" and item["field"]["name"] in global_fields)
                if source == "constraint_parameter" and option["kind"] == "constraint":
                    constraint = specification["constraints"][item["constraint_index"]]
                    belongs = (constraint["component_id"] == option["component_id"]
                               and constraint.get("product_id") == option["product_id"])
                if not belongs:
                    continue
                seen.add(address)
                if source == "product_fact":
                    if len(specification["products"]) > 1:
                        st.caption(f"Produkt: {product_name(specification, item['product_id'])}")
                    prefix = f"modular_fact_{item['product_id']}"
                elif source == "global_fact":
                    prefix = "modular_global"
                else:
                    prefix = f"modular_constraint_{option['component_id']}_{option['product_id']}"
                rendered.append(_render_missing_input(specification, item, key_prefix=prefix))
    return rendered
