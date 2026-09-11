"""
Gemeinsamer Ausführungsvertrag für Optimierungsmodelle.
"""

from collections.abc import Callable
from typing import Literal, NotRequired, TypedDict

from agentic_optimization.models.specification import (
    ModelSpecification,
)


ExecutionStatus = Literal[
    "optimal",
    "feasible",
    "infeasible",
    "unbounded",
    "iteration_limit",
    "failed",
]


SolutionMethod = Literal[
    "linear_programming",
    "analytical",
    "numerical",
]


class DecisionValue(TypedDict):
    """Ein berechneter Wert einer Entscheidungsgröße."""

    quantity_id: str
    value: float

    product_id: NotRequired[str]
    forecast_point_index: NotRequired[int]


class MetricValue(TypedDict):
    """Eine zusätzliche, unformatierte Ergebniskennzahl."""

    metric_id: str
    value: int | float

    product_id: NotRequired[str]


class OptimizationExecutionResult(TypedDict):
    """Solverunabhängiges Ergebnis einer Modellausführung."""

    status: ExecutionStatus
    solution_method: SolutionMethod
    objective_value: float | None
    decisions: list[DecisionValue]
    metrics: list[MetricValue]
    technical_representation: str | None
    errors: list[str]


OptimizationExecutor = Callable[
    [ModelSpecification],
    OptimizationExecutionResult,
]


def execute_model_specification(
    specification: ModelSpecification,
) -> OptimizationExecutionResult:
    """Führt eine Spezifikation mit dem registrierten Executor aus."""

    from agentic_optimization.models.core_registry import (
        CORE_REGISTRY,
    )

    core_id = specification["core_id"]
    registered_core = CORE_REGISTRY.get(core_id)

    if registered_core is None:
        raise ValueError(
            f"Für das Grundmodell '{core_id}' "
            "ist kein Executor registriert."
        )

    return registered_core["executor"](specification)