"""
Datenvertrag für modular zusammengesetzte Optimierungsmodelle.
"""

from typing import Any, Literal, NotRequired, TypedDict


ObjectiveSense = Literal[
    "maximize",
    "minimize",
]

SelectionStatus = Literal[
    "complete",
    "needs_clarification",
    "unsupported",
]


UnresolvedRequirementResolution = Literal[
    "clarification",
    "unsupported",
]


class ForecastQuantile(TypedDict):
    """Ein Nachfragequantil mit Wahrscheinlichkeitsniveau und Prognosewert."""

    level: float
    value: float


class DemandForecast(TypedDict, total=False):
    """Möglicherweise noch unvollständige Nachfrageprognose."""

    expected_value: float
    quantiles: list[ForecastQuantile]
    horizon_days: int


class ProductSpecification(TypedDict):
    """Möglicherweise noch unvollständige Angaben eines Produkts."""

    product_id: str
    forecast: NotRequired[DemandForecast]
    facts: NotRequired[dict[str, Any]]


class ObjectiveSpecification(TypedDict):
    """Beschreibt Zielrichtung und ausgewählte Zielfunktionsbausteine."""

    sense: ObjectiveSense
    component_ids: list[str]


class ConstraintSpecification(TypedDict):
    """Beschreibt eine ausgewählte Nebenbedingung."""

    component_id: str
    parameters: dict[str, Any]
    product_id: NotRequired[str | None]


class ExcludedRequirement(TypedDict):
    """Eine vom Nutzer bewusst ausgeschlossene Anforderung."""

    description: str
    reason: str
    sources: list[str]
    confirmation_source: str


class ModelSpecification(TypedDict):
    """Vollständige Spezifikation eines modularen Optimierungsmodells."""

    core_id: str
    products: list[ProductSpecification]
    global_facts: dict[str, Any]
    objective: ObjectiveSpecification
    constraints: list[ConstraintSpecification]
    excluded_requirements: NotRequired[
        list[ExcludedRequirement]
    ]


class ProposedObjectiveComponent(TypedDict):
    """Vom Agent vorgeschlagener Zielfunktionsbaustein."""

    component_id: str
    reason: str
    sources: list[str]


class ProposedConstraint(TypedDict):
    """Vom Agent vorgeschlagener Constraint-Baustein."""

    component_id: str
    parameters: dict[str, Any]
    reason: str
    sources: list[str]
    product_id: NotRequired[str]


class SelectionEvidence(TypedDict):
    """Eine für die Bausteinauswahl verfügbare Informationsquelle."""

    source_id: str
    content: str


class UnresolvedRequirement(TypedDict):
    """Eine noch nicht sicher abbildbare Anforderung."""

    description: str
    resolution: UnresolvedRequirementResolution
    candidate_component_ids: list[str]
    reason: str
    sources: list[str]


class ComponentSelectionDecision(TypedDict):
    """Fachliche Auswahlentscheidung des LLM."""

    objective_sense: ObjectiveSense
    objective_components: list[
        ProposedObjectiveComponent
    ]
    constraints: list[ProposedConstraint]
    unresolved_requirements: list[
        UnresolvedRequirement
    ]


class ComponentSelectionProposal(TypedDict):
    """Nachvollziehbarer Vorschlag des Component-Selection-Agents."""

    core_id: str
    selection_status: SelectionStatus
    objective_sense: ObjectiveSense
    objective_components: list[
        ProposedObjectiveComponent
    ]
    constraints: list[ProposedConstraint]
    unresolved_requirements: list[
        UnresolvedRequirement
    ]