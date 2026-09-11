"""
Registry für verfügbare Grundmodelle.
"""

from collections.abc import Callable
from typing import NotRequired, TypedDict

from agentic_optimization.models.component_registry import (
    ModelBuildContext,
)
from agentic_optimization.models.newsvendor.single_period import (
    CORE_DEFINITION as SINGLE_PERIOD_NEWSVENDOR_DEFINITION,
    build_single_period_newsvendor_core,
    execute_single_period_newsvendor,
    validate_single_period_newsvendor_semantics,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.types import (
    ModelCoreDefinition,
)
from agentic_optimization.models.execution import (
    OptimizationExecutor,
    SolutionMethod,
)


CoreBuilder = Callable[
    [ModelSpecification],
    ModelBuildContext,
]


CoreSemanticValidator = Callable[
    [ModelSpecification],
    list[str],
]


class RegisteredModelCore(TypedDict):
    """Verknüpft Definition, Ausführung und fachliche Prüfung."""

    definition: ModelCoreDefinition
    solution_method: SolutionMethod
    semantic_validator: CoreSemanticValidator
    executor: OptimizationExecutor
    builder: NotRequired[CoreBuilder]


CORE_REGISTRY: dict[str, RegisteredModelCore] = {
    SINGLE_PERIOD_NEWSVENDOR_DEFINITION["core_id"]: {
        "definition": SINGLE_PERIOD_NEWSVENDOR_DEFINITION,
        "solution_method": "linear_programming",
        "builder": build_single_period_newsvendor_core,
        "semantic_validator": validate_single_period_newsvendor_semantics,
        "executor": execute_single_period_newsvendor,
    },
}