"""
Konvertiert modulare UI-Eingaben in Workflow-Einreichungen.
"""

from typing import Any, TypedDict

from agentic_optimization.models.validator import (
    MissingInput,
)
from agentic_optimization.state import (
    ModularInputSubmission,
)
from agentic_optimization.ui.field_renderer import (
    parse_input_value,
)


class RenderedModularInput(TypedDict):
    """Ein gerenderter UI-Rohwert mit seiner Zieladresse."""

    missing_input: MissingInput
    raw_value: Any


def _is_empty_input(value: Any) -> bool:
    return value is None or value == "" or value == []


def parse_modular_input_submissions(
    rendered_inputs: list[RenderedModularInput],
) -> list[ModularInputSubmission]:
    """Prüft und konvertiert die modularen Formulareingaben."""

    submissions: list[ModularInputSubmission] = []

    for rendered_input in rendered_inputs:
        missing_input = rendered_input["missing_input"]
        field = missing_input["field"]

        parsed_value = parse_input_value(
            field,
            rendered_input["raw_value"],
        )

        if (
            field["required"]
            and _is_empty_input(parsed_value)
        ):
            raise ValueError(
                "Bitte geben Sie einen Wert für "
                f"„{field['label']}“ ein."
            )

        if _is_empty_input(parsed_value):
            continue

        submissions.append(
            {
                "missing_input": missing_input,
                "value": parsed_value,
            }
        )

    return submissions
