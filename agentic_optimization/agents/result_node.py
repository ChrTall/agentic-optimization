from agentic_optimization.state import (
    GraphState,
    WORKFLOW_READY_FOR_RESULT,
    with_control,
)
from agentic_optimization.visualization.markdown_renderer import render_model_markdown


def result_node(state: GraphState) -> GraphState:
    """
    Fasst die Ergebnisse des Workflows strukturiert zusammen.

    Dieser Node bereitet nur das Ergebnis für UI/API/Chat auf.
    Fachliche Validierung und spätere Code-Generierung gehören in eigene Nodes.
    """

    requirements = state.get("requirements", {})
    model_config = state.get("model_config", {})
    selected_model = (
        state.get("selected_model")
        or model_config.get("model_type")
    )
    model_specific_inputs = state.get("model_specific_inputs", {})
    selected_model_inputs = model_specific_inputs.get(selected_model, {})
    pulp_code = state.get("pulp_code", "")

    result = {
        "summary": _build_summary(
            requirements=requirements,
            model_config=model_config,
            selected_model=selected_model,
        ),
        "requirements": requirements,
        "selected_model": {
            "type": selected_model,
            "name": model_config.get("model_name"),
            "explanation": model_config.get("explanation"),
        },
        "model_config": model_config,
        "model_specific_inputs": model_specific_inputs,
        "selected_model_inputs": selected_model_inputs,
        "model_markdown": render_model_markdown(
            model_config,
            _requirements_with_model_inputs(
                requirements=requirements,
                selected_model=selected_model,
                selected_model_inputs=selected_model_inputs,
            ),
        ),
        "generated_code": pulp_code,
        "next_steps": [
            "Modellannahmen validieren.",
            "PuLP-Code mit echten Daten befüllen.",
            "Optimierungsmodell lösen.",
            "Ergebnis fachlich erklären und Empfehlung ausgeben.",
        ],
    }

    return with_control(
        {
            **state,
            "ready_for_generation": True,
            "result": result,
        },
        event=None,
        workflow_status=WORKFLOW_READY_FOR_RESULT,
        
    )


def _requirements_with_model_inputs(
    requirements: dict,
    selected_model: str | None,
    selected_model_inputs: dict,
) -> dict:
    if not selected_model or not selected_model_inputs:
        return requirements

    model_parameters = {
        **requirements.get("model_parameters", {}),
        selected_model: selected_model_inputs,
    }

    return {
        **requirements,
        "model_parameters": model_parameters,
    }


def _build_summary(
    requirements: dict,
    model_config: dict,
    selected_model: str | None,
) -> str:
    decision = requirements.get("decision", "nicht angegeben")
    objective = requirements.get("objective", "nicht angegeben")
    model_name = model_config.get("model_name") or selected_model or "kein Modell ausgewählt"

    return (
        f"Für die Entscheidung '{decision}' mit dem Ziel '{objective}' "
        f"wurde als passender Startpunkt das Modell '{model_name}' ausgewählt."
    )