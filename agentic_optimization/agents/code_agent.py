from agentic_optimization.models.registry import get_model_variant_definition
from agentic_optimization.state import (
    WORKFLOW_CODE_GENERATED,
    WORKFLOW_CODE_GENERATION_FAILED,
    GraphState,
    with_control,
)


def generate_pulp_code(state: GraphState) -> GraphState:
    """
    Erzeugt PuLP-Code ausschließlich nach erfolgreicher Validierung.
    """

    if not state.get("ready_for_generation"):
        return with_control(
            {
                **state,
                "pulp_code": None,
                "generation_errors": [
                    "Die Codegenerierung ist erst nach erfolgreicher Validierung aller Eingaben und Constraints möglich."
                ],
            },
            event=None,
            workflow_status=WORKFLOW_CODE_GENERATION_FAILED,
        )

    family = state.get("selected_model_family")
    variant = state.get("selected_model_variant")

    if not family or not variant:
        return with_control(
            {
                **state,
                "pulp_code": None,
                "generation_errors": [
                    "Für die Codegenerierung fehlen Modellfamilie oder Modellvariante."
                ],
            },
            event=None,
            workflow_status=WORKFLOW_CODE_GENERATION_FAILED,
        )

    model_definition = get_model_variant_definition(
        family,
        variant,
    )

    if model_definition is None:
        return with_control(
            {
                **state,
                "pulp_code": None,
                "generation_errors": [
                    f"Die Modellvariante '{family}/{variant}' ist nicht registriert."
                ],
            },
            event=None,
            workflow_status=WORKFLOW_CODE_GENERATION_FAILED,
        )

    inputs = (
        state.get("model_specific_inputs", {})
        .get(family, {})
        .get(variant, {})
    )

    constraints = state.get("structured_constraints", [])
    code_builder = model_definition["code_builder"]

    try:
        pulp_code = code_builder(
            inputs,
            constraints,
        )
    except (KeyError, TypeError, ValueError) as error:
        return with_control(
            {
                **state,
                "pulp_code": None,
                "generation_errors": [str(error)],
            },
            event=None,
            workflow_status=WORKFLOW_CODE_GENERATION_FAILED,
        )

    return with_control(
        {
            **state,
            "pulp_code": pulp_code,
            "generation_errors": [],
        },
        event=None,
        workflow_status=WORKFLOW_CODE_GENERATED,
    )