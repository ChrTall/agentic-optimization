"""
Deterministische Zusammensetzung modularer Optimierungsmodelle.
"""

import pulp

from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
    OBJECTIVE_COMPONENTS,
    ModelBuildContext,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)
from agentic_optimization.models.registry_validator import (
    validate_registry_integrity,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.validator import (
    validate_model_specification,
)

def compile_model_specification(
    specification: ModelSpecification,
) -> ModelBuildContext:
    """Validiert und kompiliert eine Modellspezifikation."""

    registry_errors = validate_registry_integrity()

    if registry_errors:
        raise RuntimeError(
            "Die Optimierungs-Registry ist ungültig: "
            + "; ".join(registry_errors)
        )

    validation_result = validate_model_specification(
        specification
    )

    if not validation_result["is_valid"]:
        messages = list(validation_result["errors"])

        messages.extend(
            (
                "Fehlende Eingabe: "
                f"{missing_input['field']['name']}"
            )
            for missing_input
            in validation_result["missing_inputs"]
        )

        raise ValueError(
            "Die Modellspezifikation ist nicht vollständig: "
            + "; ".join(messages)
        )

    core_id = specification["core_id"]
    registered_core = CORE_REGISTRY.get(core_id)

    if registered_core is None:
        raise ValueError(
            f"Für das Grundmodell '{core_id}' "
            "ist kein Builder registriert."
        )

    context = registered_core["builder"](specification)

    objective_sense = specification["objective"]["sense"]
    objective_terms = []

    for component_id in specification[
        "objective"
    ]["component_ids"]:
        definition = OBJECTIVE_COMPONENTS[component_id]

        component_expression = definition["builder"](
            specification,
            context,
        )

        if objective_sense == "maximize":
            factor = (
                1
                if definition["effect"] == "benefit"
                else -1
            )
        else:
            factor = (
                1
                if definition["effect"] == "cost"
                else -1
            )

        objective_terms.append(
            factor * component_expression
        )

    context.model += (
        pulp.lpSum(objective_terms),
        "assembled_objective",
    )

    for constraint_index, constraint in enumerate(
        specification["constraints"]
    ):
        definition = CONSTRAINT_COMPONENTS[
            constraint["component_id"]
        ]

        definition["builder"](
            specification,
            context,
            constraint,
            constraint_index,
        )

    return context