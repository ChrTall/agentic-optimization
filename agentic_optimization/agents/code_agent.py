import pulp

from agentic_optimization.models.registry import (
    get_model_variant_definition,
)
from agentic_optimization.state import (
    WORKFLOW_OPTIMIZATION_COMPLETED,
    WORKFLOW_OPTIMIZATION_FAILED,
    GraphState,
    with_control,
)


def optimize_model(
    state: GraphState,
) -> GraphState:
    """
    Baut und löst eine vollständig validierte Modellvariante.
    """

    if not state.get("ready_for_optimization"):
        return with_control(
            {
                **state,
                "optimization_result": None,
                "optimization_errors": [
                    "Die Optimierung ist erst nach erfolgreicher Validierung aller Eingaben und Constraints möglich."
                ],
            },
            event=None,
            workflow_status=WORKFLOW_OPTIMIZATION_FAILED,
        )

    if not state.get("review_confirmed"):
        return with_control(
            {
                **state,
                "optimization_result": None,
                "optimization_errors": [
                    "Die Optimierung wurde noch nicht durch den Nutzer bestätigt."
                ],
            },
            event=None,
            workflow_status=WORKFLOW_OPTIMIZATION_FAILED,
        )

    family = state.get("selected_model_family")
    variant = state.get("selected_model_variant")

    if not family or not variant:
        return with_control(
            {
                **state,
                "optimization_result": None,
                "optimization_errors": [
                    "Für die Optimierung fehlen Modellfamilie oder Modellvariante."
                ],
            },
            event=None,
            workflow_status=WORKFLOW_OPTIMIZATION_FAILED,
        )

    model_definition = get_model_variant_definition(
        family,
        variant,
    )

    if model_definition is None:
        return with_control(
            {
                **state,
                "optimization_result": None,
                "optimization_errors": [
                    f"Die Modellvariante '{family}/{variant}' ist nicht registriert."
                ],
            },
            event=None,
            workflow_status=WORKFLOW_OPTIMIZATION_FAILED,
        )

    inputs = (
        state.get("model_specific_inputs", {})
        .get(family, {})
        .get(variant, {})
    )

    constraints = state.get(
        "structured_constraints",
        [],
    )

    optimizer = model_definition["optimizer"]

    try:
        optimization_result = optimizer(
            inputs,
            constraints,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        pulp.PulpError,
    ) as error:
        return with_control(
            {
                **state,
                "optimization_result": None,
                "optimization_errors": [str(error)],
            },
            event=None,
            workflow_status=WORKFLOW_OPTIMIZATION_FAILED,
        )

    return with_control(
        {
            **state,
            "optimization_result": optimization_result,
            "optimization_errors": [],
        },
        event=None,
        workflow_status=WORKFLOW_OPTIMIZATION_COMPLETED,
    )