from collections.abc import Callable
from typing import Any, TypedDict


class StructuredConstraint(TypedDict):
    """Ein validierbarer, strukturierter optionaler Constraint."""

    type: str
    parameters: dict[str, int | float]
    source_text: str | None


class ModelInputField(TypedDict):
    """Beschreibt ein Eingabefeld einer konkreten Modellvariante."""

    name: str
    label: str
    type: str
    required: bool
    help: str | None


class ConstraintParameterDefinition(TypedDict):
    """Beschreibt einen Parameter eines optionalen Constraints."""

    name: str
    label: str
    type: str
    required: bool
    help: str | None


class ConstraintDefinition(TypedDict):
    """Beschreibt einen von einer Modellvariante unterstützten Constraint."""

    type: str
    name: str
    parameters: list[ConstraintParameterDefinition]


InputValidator = Callable[
    [dict[str, Any]],
    list[str],
]


ConstraintValidator = Callable[
    [
        dict[str, Any],
        list[StructuredConstraint],
    ],
    list[str],
]


ModelOptimizer = Callable[
    [
        dict[str, Any],
        list[StructuredConstraint],
    ],
    dict[str, Any],
]


class ModelVariantDefinition(TypedDict):
    """Vollständiger Datenvertrag einer konkreten Modellvariante."""

    variant_id: str
    variant_name: str
    required_fields: list[str]
    input_fields: list[ModelInputField]
    supported_constraints: dict[str, ConstraintDefinition]
    input_validator: InputValidator
    constraint_validator: ConstraintValidator
    optimizer: ModelOptimizer


class ModelFamilyDefinition(TypedDict):
    """Beschreibt eine Modellfamilie und ihre registrierten Varianten."""

    family_id: str
    family_name: str
    variants: dict[str, ModelVariantDefinition]

