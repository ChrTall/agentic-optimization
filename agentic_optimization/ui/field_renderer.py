"""
Rendert Registry-definierte Eingabefelder und konvertiert ihre Widgetwerte.
"""

from typing import Any, TypeAlias

import streamlit as st

from agentic_optimization.models.types import (
    ConstraintParameterDefinition,
    ModelInputField,
)

FieldDefinition: TypeAlias = (
    ModelInputField | ConstraintParameterDefinition
)


def _build_field_label(
    field_definition: FieldDefinition,
) -> str:
    label = field_definition["label"]
    unit = field_definition.get("unit")

    if unit:
        return f"{label} ({unit})"

    return label


def _format_existing_number_list(
    existing_value: Any,
) -> str:
    if not isinstance(existing_value, list):
        return ""

    return ", ".join(
        str(value)
        for value in existing_value
    )


def render_input_field(
    field_definition: FieldDefinition,
    existing_value: Any,
    *,
    key_prefix: str,
    disabled: bool = False,
) -> Any:
    """
    Rendert ein Eingabefeld anhand seiner Registrydefinition.

    Der Rückgabewert ist zunächst der rohe Widgetwert. Typkonvertierung
    und generische Grenzprüfung erfolgen mit parse_input_value().
    """

    field_name = field_definition["name"]
    field_type = field_definition["type"]

    label = _build_field_label(field_definition)
    widget_key = f"{key_prefix}_{field_name}"

    help_text = field_definition.get("help")
    placeholder = field_definition.get("placeholder")

    if field_type in {"number", "integer"}:
        is_integer_field = field_type == "integer"

        initial_value = (
            int(existing_value)
            if (
                is_integer_field
                and isinstance(existing_value, (int, float))
                and not isinstance(existing_value, bool)
                and float(existing_value).is_integer()
            )
            else float(existing_value)
            if isinstance(existing_value, (int, float))
            and not isinstance(existing_value, bool)
            else None
        )

        # Nur beim ersten Rendern initialisieren, damit Streamlit-Reruns
        # spätere Nutzereingaben nicht überschreiben.
        if widget_key not in st.session_state:
            st.session_state[
                widget_key
            ] = initial_value

        number_input_arguments: dict[str, Any] = {
            "label": label,
            "value": None,
            "help": help_text,
            "key": widget_key,
            "disabled": disabled,
        }

        min_value = field_definition.get("min_value")
        max_value = field_definition.get("max_value")
        step = field_definition.get("step")

        if min_value is not None:
            number_input_arguments["min_value"] = (
                int(min_value)
                if is_integer_field
                else float(min_value)
            )

        if max_value is not None:
            number_input_arguments["max_value"] = (
                int(max_value)
                if is_integer_field
                else float(max_value)
            )

        if step is not None:
            number_input_arguments["step"] = (
                int(step)
                if is_integer_field
                else float(step)
            )

        if placeholder:
            number_input_arguments["placeholder"] = placeholder

        return st.number_input(**number_input_arguments)

    if field_type == "number_list":
        return st.text_input(
            label,
            value=_format_existing_number_list(
                existing_value
            ),
            help=help_text,
            placeholder=placeholder,
            key=widget_key,
            disabled=disabled,
        )

    raise ValueError(
        f"Der Feldtyp '{field_type}' wird noch nicht unterstützt."
    )


def _validate_numeric_bounds(
    *,
    field_definition: FieldDefinition,
    values: list[float],
) -> None:
    """Prüft Werte gegen min_value und max_value der Field-Definition."""

    min_value = field_definition.get("min_value")
    max_value = field_definition.get("max_value")
    label = field_definition["label"]

    if min_value is not None:
        for value in values:
            if value < min_value:
                raise ValueError(
                    f"'{label}' darf keine Werte kleiner als "
                    f"{min_value} enthalten."
                )

    if max_value is not None:
        for value in values:
            if value > max_value:
                raise ValueError(
                    f"'{label}' darf keine Werte größer als "
                    f"{max_value} enthalten."
                )


def parse_input_value(
    field_definition: FieldDefinition,
    raw_value: Any,
) -> int | float | list[float] | None:
    """
    Konvertiert einen Widgetwert in den internen Modellwert.

    Leere Werte werden als None beziehungsweise leere Liste zurückgegeben.
    Die Prüfung erforderlicher Felder erfolgt anschließend im Validation Agent.
    """

    field_type = field_definition["type"]

    if field_type in {"number", "integer"}:
        if raw_value in (None, ""):
            return None

        if (
            not isinstance(raw_value, (int, float))
            or isinstance(raw_value, bool)
        ):
            raise TypeError(
                f"'{field_definition['label']}' muss eine Zahl sein."
            )

        parsed_value = float(raw_value)

        _validate_numeric_bounds(
            field_definition=field_definition,
            values=[parsed_value],
        )

        if field_type == "integer":
            if not parsed_value.is_integer():
                raise ValueError(
                    f"'{field_definition['label']}' muss in "
                    "ganzen Einheiten angegeben werden."
                )

            return int(parsed_value)

        return parsed_value

    if field_type == "number_list":
        if raw_value in (None, "", []):
            return []

        if isinstance(raw_value, list):
            raw_items = raw_value
        elif isinstance(raw_value, str):
            raw_items = [
                item.strip()
                for item in raw_value.split(",")
                if item.strip()
            ]
        else:
            raise TypeError(
                f"'{field_definition['label']}' muss eine "
                "kommagetrennte Zahlenliste sein."
            )

        parsed_values: list[float] = []

        for raw_item in raw_items:
            if (
                not isinstance(raw_item, (int, float, str))
                or isinstance(raw_item, bool)
            ):
                raise TypeError(
                    f"'{field_definition['label']}' enthält "
                    "einen Wert mit ungültigem Datentyp."
                )

            try:
                parsed_values.append(float(raw_item))
            except ValueError as error:
                raise ValueError(
                    f"'{field_definition['label']}' muss "
                    "ausschließlich Zahlen enthalten."
                ) from error

        _validate_numeric_bounds(
            field_definition=field_definition,
            values=parsed_values,
        )

        return parsed_values

    raise ValueError(
        f"Der Feldtyp '{field_type}' wird noch nicht unterstützt."
    )
