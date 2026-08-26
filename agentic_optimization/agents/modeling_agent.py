from agentic_optimization.models.registry import (
    find_matching_model_variant,
)
from agentic_optimization.state import (
    WORKFLOW_MODEL_SELECTED,
    WORKFLOW_MODEL_UNSUPPORTED,
    GraphState,
    ModelConfig,
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
                "model_config": {
                    "explanation": (
                        "Für diese Problemstruktur ist derzeit "
                        "keine Modellvariante verfügbar."
                    ),
                    "objective": requirements.get("objective"),
                    "decision": requirements.get("decision"),
                },
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

    model_config: ModelConfig = {
        "family_id": family_id,
        "family_name": family_definition["family_name"],
        "variant_id": variant_id,
        "variant_name": variant_definition["variant_name"],
        "explanation": (
            "Die strukturierte Problemdefinition entspricht "
            f"der Modellfamilie "
            f"'{family_definition['family_name']}' und der "
            f"Variante '{variant_definition['variant_name']}'."
        ),
        "objective": requirements.get("objective"),
        "decision": requirements.get("decision"),
    }

    return with_control(
        {
            **state,
            "selected_model_family": family_id,
            "selected_model_variant": variant_id,
            "model_config": model_config,
            "model_selection_error": None,
            "ready_for_optimization": False,
        },
        event=None,
        workflow_status=WORKFLOW_MODEL_SELECTED,
    )
