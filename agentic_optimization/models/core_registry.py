"""
Registry für verfügbare Grundmodelle.
"""

from collections.abc import Callable
from typing import TypedDict

from agentic_optimization.models.component_registry import (
    ModelBuildContext,
)
from agentic_optimization.models.newsvendor.single_period import (
    CORE_DEFINITION as SINGLE_PERIOD_NEWSVENDOR_DEFINITION,
    build_single_period_newsvendor_core,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.types import (
    ModelCoreDefinition,
)


CoreBuilder = Callable[
    [ModelSpecification],
    ModelBuildContext,
]


class RegisteredModelCore(TypedDict):
    """Verknüpft die Definition eines Grundmodells mit seinem Builder."""

    definition: ModelCoreDefinition
    builder: CoreBuilder


CORE_REGISTRY: dict[str, RegisteredModelCore] = {
    SINGLE_PERIOD_NEWSVENDOR_DEFINITION["core_id"]: {
        "definition": SINGLE_PERIOD_NEWSVENDOR_DEFINITION,
        "builder": build_single_period_newsvendor_core,
    },
}