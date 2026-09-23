"""
Registry für verfügbare Grundmodelle.
"""

from collections.abc import Callable
from typing import NotRequired, TypedDict

from agentic_optimization.models.component_registry import (
    ModelBuildContext,
)
from agentic_optimization.models.eoq.core import (
    CORE_DEFINITION as EOQ_DEFINITION,
    execute_eoq,
    validate_eoq_forecast,
    validate_eoq_semantics,
)
from agentic_optimization.models.execution import (
    OptimizationExecutor,
    SolutionMethod,
)
from agentic_optimization.models.reorder_point.core import (
    CORE_DEFINITION as CONTINUOUS_REVIEW_Q_R_DEFINITION,
    execute_continuous_review_q_r,
    validate_q_r_forecast,
    validate_q_r_semantics,
)
from agentic_optimization.models.newsvendor.single_period import (
    CORE_DEFINITION as SINGLE_PERIOD_NEWSVENDOR_DEFINITION,
    build_single_period_newsvendor_core,
    execute_single_period_newsvendor,
    validate_single_period_newsvendor_forecast,
    validate_single_period_newsvendor_semantics,
)
from agentic_optimization.models.specification import (
    DemandForecast,
    ModelSpecification,
)
from agentic_optimization.models.types import (
    ModelCoreDefinition,
)

CoreBuilder = Callable[
    [ModelSpecification],
    ModelBuildContext,
]


CoreSemanticValidator = Callable[
    [ModelSpecification],
    list[str],
]


CoreForecastValidator = Callable[
    [DemandForecast],
    list[str],
]


class RegisteredModelCore(TypedDict):
    """Verknüpft Definition, Ausführung und fachliche Prüfung."""

    definition: ModelCoreDefinition
    solution_method: SolutionMethod
    semantic_validator: CoreSemanticValidator
    executor: OptimizationExecutor
    builder: NotRequired[CoreBuilder]
    forecast_validator: CoreForecastValidator


CORE_REGISTRY: dict[str, RegisteredModelCore] = {
    SINGLE_PERIOD_NEWSVENDOR_DEFINITION["core_id"]: {
        "definition": SINGLE_PERIOD_NEWSVENDOR_DEFINITION,
        "solution_method": "linear_programming",
        "builder": build_single_period_newsvendor_core,
        "semantic_validator": validate_single_period_newsvendor_semantics,
        "executor": execute_single_period_newsvendor,
        "forecast_validator": validate_single_period_newsvendor_forecast,
    },
    EOQ_DEFINITION["core_id"]: {
        "definition": EOQ_DEFINITION,
        "solution_method": "analytical",
        "semantic_validator": validate_eoq_semantics,
        "executor": execute_eoq,
        "forecast_validator": validate_eoq_forecast,
    },
    CONTINUOUS_REVIEW_Q_R_DEFINITION["core_id"]: {
        "definition": CONTINUOUS_REVIEW_Q_R_DEFINITION,
        "solution_method": "numerical",
        "semantic_validator": validate_q_r_semantics,
        "executor": execute_continuous_review_q_r,
        "forecast_validator": validate_q_r_forecast,
    },
}
