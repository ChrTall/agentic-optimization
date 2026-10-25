"""
Rendert Registry-definierte Eingabefelder und konvertiert ihre Widgetwerte.
"""
import math

from typing import Any, TypeAlias

import streamlit as st

from agentic_optimization.models.specification import ForecastQuantile
from agentic_optimization.models.types import (
    ConstraintParameterDefinition,
    ModelInputField,
)
from agentic_optimization.i18n import t

FieldDefinition: TypeAlias = (
    ModelInputField | ConstraintParameterDefinition
)


def _build_field_label(
    field_definition: FieldDefinition,
) -> str:
    label = t(field_definition["label"])
    unit = t(field_definition.get("unit"))

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


def _format_existing_forecast_quantiles(
    existing_value: Any,
) -> str:
    """Formatiert vorhandene Quantile für das Eingabefeld."""

    if not isinstance(existing_value, list):
        return ""

    formatted_quantiles: list[str] = []

    for quantile in existing_value:
        if not isinstance(quantile, dict):
            continue

        level = quantile.get("level")
        value = quantile.get("value")

        if level is None or value is None:
            continue

        formatted_quantiles.append(
            f"{level}: {value}"
        )

    return "; ".join(formatted_quantiles)


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

    help_text = t(field_definition.get("help"))
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

    if field_type == "forecast_quantiles":
        return st.text_input(
            label,
            value=_format_existing_forecast_quantiles(
                existing_value
            ),
            help=help_text,
            placeholder=(
                placeholder
                or "0.1: 70; 0.5: 100; 0.9: 140"
            ),
            key=widget_key,
            disabled=disabled,
        )

    raise ValueError(
        t("Der Feldtyp '{field_type}' wird noch nicht unterstützt.", field_type=field_type)
    )


def _validate_numeric_bounds(
    *,
    field_definition: FieldDefinition,
    values: list[float],
) -> None:
    """Prüft Werte gegen min_value und max_value der Field-Definition."""

    min_value = field_definition.get("min_value")
    max_value = field_definition.get("max_value")
    label = t(field_definition["label"])

    if min_value is not None:
        for value in values:
            if value < min_value:
                raise ValueError(
                    t("'{label}' darf keine Werte kleiner als {min_value} enthalten.", label=label, min_value=min_value)
                )

    if max_value is not None:
        for value in values:
            if value > max_value:
                raise ValueError(
                    t("'{label}' darf keine Werte größer als {max_value} enthalten.", label=label, max_value=max_value)
                )


def _parse_forecast_quantiles(
    raw_value: Any,
) -> list[ForecastQuantile]:
    """Konvertiert Quantile aus UI-Text oder strukturierter Form."""

    if raw_value in (None, "", []):
        return []

    raw_quantiles: list[tuple[Any, Any]] = []

    if isinstance(raw_value, str):
        entries = [
            entry.strip()
            for entry in raw_value.split(";")
            if entry.strip()
        ]

        for entry in entries:
            raw_level, separator, raw_forecast_value = (
                entry.partition(":")
            )

            if not separator:
                raise ValueError(
                    t("Jedes Nachfragequantil muss im Format 'Niveau: Wert' angegeben werden.")
                )

            raw_quantiles.append(
                (
                    raw_level.strip().replace(",", "."),
                    raw_forecast_value.strip().replace(
                        ",",
                        ".",
                    ),
                )
            )

    elif isinstance(raw_value, list):
        for quantile in raw_value:
            if not isinstance(quantile, dict):
                raise TypeError(
                    t("Jedes Nachfragequantil muss ein Objekt sein.")
                )

            if (
                "level" not in quantile
                or "value" not in quantile
            ):
                raise ValueError(
                    t("Jedes Nachfragequantil benötigt 'level' und 'value'.")
                )

            raw_quantiles.append(
                (
                    quantile["level"],
                    quantile["value"],
                )
            )

    else:
        raise TypeError(
            t("Nachfragequantile müssen als Text oder Liste angegeben werden.")
        )

    parsed_quantiles: list[ForecastQuantile] = []

    for raw_level, raw_forecast_value in raw_quantiles:
        if (
            isinstance(raw_level, bool)
            or isinstance(raw_forecast_value, bool)
        ):
            raise TypeError(
                t("Quantilniveau und Nachfragewert müssen Zahlen sein.")
            )

        try:
            level = float(raw_level)
            forecast_value = float(raw_forecast_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                t("Quantilniveau und Nachfragewert müssen Zahlen sein.")
            ) from error

        if not (
            math.isfinite(level)
            and math.isfinite(forecast_value)
        ):
            raise ValueError(
                t("Quantilniveau und Nachfragewert müssen endliche Zahlen sein.")
            )

        if not 0.0 < level < 1.0:
            raise ValueError(
                t("Das Quantilniveau muss zwischen 0 und 1 liegen.")
            )

        if forecast_value < 0.0:
            raise ValueError(
                t("Der Nachfragewert darf nicht negativ sein.")
            )

        parsed_quantiles.append(
            {
                "level": level,
                "value": forecast_value,
            }
        )

    return parsed_quantiles


def parse_input_value(
    field_definition: FieldDefinition,
    raw_value: Any,
) -> int | float | list[float] | list[ForecastQuantile] | None:
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
                t("'{label}' muss eine Zahl sein.", label=field_definition['label'])
            )

        parsed_value = float(raw_value)

        _validate_numeric_bounds(
            field_definition=field_definition,
            values=[parsed_value],
        )

        if field_type == "integer":
            if not parsed_value.is_integer():
                raise ValueError(
                    t("'{label}' muss in ganzen Einheiten angegeben werden.", label=field_definition['label'])
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
                t("'{label}' muss eine kommagetrennte Zahlenliste sein.", label=field_definition['label'])
            )

        parsed_values: list[float] = []

        for raw_item in raw_items:
            if (
                not isinstance(raw_item, (int, float, str))
                or isinstance(raw_item, bool)
            ):
                raise TypeError(
                    t("'{label}' enthält einen Wert mit ungültigem Datentyp.", label=field_definition['label'])
                )

            try:
                parsed_values.append(float(raw_item))
            except ValueError as error:
                raise ValueError(
                    t("'{label}' muss ausschließlich Zahlen enthalten.", label=field_definition['label'])
                ) from error

        _validate_numeric_bounds(
            field_definition=field_definition,
            values=parsed_values,
        )

        return parsed_values

    if field_type == "forecast_quantiles":
        return _parse_forecast_quantiles(raw_value)

    raise ValueError(
        t("Der Feldtyp '{field_type}' wird noch nicht unterstützt.", field_type=field_type)
    )
