"""
Gemeinsame Typverträge für Registry, Modellvarianten, Validierung und UI.
"""

from collections.abc import Callable
from typing import Any, Literal, NotRequired, TypedDict

from agentic_optimization.models.specification import ObjectiveSense

class StructuredConstraint(TypedDict):
    """Ein gemappter oder vom Nutzer bestätigter Constraint mit Parametern."""

    type: str
    parameters: dict[str, int | float]
    source_text: str | None


class ResultMetric(TypedDict):
    """Eine prominent dargestellte fachliche Ergebniskennzahl."""

    label: str
    value: str


class VariantResultPresentation(TypedDict):
    """
    Variantenspezifische, bereits formatierte Ergebnisdarstellung.

    Der zentrale UI-Builder ergänzt Status, Fehler,
    Rahmenbedingungen und technische Details.
    """

    metrics: list[ResultMetric]
    technical_metrics: list[ResultMetric]
    detail_title: str | None
    detail_rows: list[dict[str, str]]


class ModelInputField(TypedDict):
    """Beschreibt ein Eingabefeld einer Modellvariante."""

    name: str
    label: str
    type: str
    required: bool
    help: str | None

    placeholder: NotRequired[str | None]
    min_value: NotRequired[int | float | None]
    max_value: NotRequired[int | float | None]
    step: NotRequired[int | float | None]
    unit: NotRequired[str | None]


QuantityRole = Literal[
    "decision",
    "derived",
]

QuantityScope = Literal[
    "per_product",
    "per_product_and_forecast_point",
]


class ModelQuantityDefinition(TypedDict):
    """Beschreibt eine mathematische Größe eines Grundmodells."""

    quantity_id: str
    name: str
    role: QuantityRole
    scope: QuantityScope
    description: str


class ModelCoreDefinition(TypedDict):
    """Beschreibt den fachlichen Vertrag eines Grundmodells."""

    core_id: str
    name: str
    description: str
    decision_description: str
    assumptions: list[str]
    minimum_products: int
    required_forecast_fields: list[ModelInputField]
    required_product_facts: list[ModelInputField]
    required_global_facts: list[ModelInputField]
    required_objective_component_ids: list[str]
    required_objective_sense: NotRequired[ObjectiveSense]
    provided_quantities: dict[str, ModelQuantityDefinition]

class ConstraintParameterDefinition(TypedDict):
    """Beschreibt einen Parameter eines optionalen Constraints."""

    name: str
    label: str
    type: str
    required: bool
    help: str | None

    placeholder: NotRequired[str | None]
    min_value: NotRequired[int | float | None]
    max_value: NotRequired[int | float | None]
    step: NotRequired[int | float | None]
    unit: NotRequired[str | None]


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


ResultPresentationBuilder = Callable[
    [dict[str, Any]],
    VariantResultPresentation,
]


class ModelSelectionCriteria(TypedDict):
    """Requirements-Felder für die deterministische Auswahl einer Modellvariante."""

    decision: str
    objective: str
    demand_type: str
    product_scope: str
    planning_horizon: str
    review_type: NotRequired[str]


class ModelVariantDefinition(TypedDict):
    """Vollständiger Datenvertrag einer konkreten Modellvariante."""

    variant_id: str
    variant_name: str
    selection_criteria: ModelSelectionCriteria
    input_fields: list[ModelInputField]
    supported_constraints: dict[str, ConstraintDefinition]
    input_validator: InputValidator
    constraint_validator: ConstraintValidator
    optimizer: ModelOptimizer
    result_builder: ResultPresentationBuilder


class ModelFamilyDefinition(TypedDict):
    """Beschreibt eine Modellfamilie und ihre registrierten Varianten."""

    family_id: str
    family_name: str
    variants: dict[str, ModelVariantDefinition]
