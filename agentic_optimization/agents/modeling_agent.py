from agentic_optimization.models.registry import (
    find_matching_model_variant,
)
from agentic_optimization.state import (
    WORKFLOW_MODEL_SELECTED,
    WORKFLOW_MODEL_UNSUPPORTED,
    GraphState,
    with_control,
)


def modeling_agent(state: GraphState) -> GraphState:
    """
    Wählt anhand validierter strukturierter Requirements
    deterministisch eine registrierte Modellvariante aus.
    """

    requirements = state.get("requirements", {})

    # Die Modellauswahl erfolgt ausschließlich über die in der Registry
    # definierten Regeln. Das LLM entscheidet nicht über das Modell.
    match = find_matching_model_variant(requirements)

    if match is None:
        return with_control(
            {
                **state,
                "selected_model_family": None,
                "selected_model_variant": None,
                "model_selection_error": (
                    "Die aktuelle Version unterstützt diese "
                    "Kombination aus Entscheidung, Ziel, "
                    "Nachfrageart, Planungshorizont und Art "
                    "der Bestandsüberprüfung noch nicht."
                ),
                "ready_for_optimization": False,
            },
            event=None,
            workflow_status=WORKFLOW_MODEL_UNSUPPORTED,
        )

    family_definition, variant_definition = match

    family_id = family_definition["family_id"]
    variant_id = variant_definition["variant_id"]

    return with_control(
        {
            **state,
            "selected_model_family": family_id,
            "selected_model_variant": variant_id,
            "model_selection_error": None,
            "ready_for_optimization": False,
        },
        event=None,
        workflow_status=WORKFLOW_MODEL_SELECTED,
    )
