from agentic_optimization.state import GraphState
from agentic_optimization.visualization.markdown_renderer import render_model_markdown

def result_node(state: GraphState) -> GraphState:
    """
    Fasst die Ergebnisse des Workflows strukturiert zusammen.

    Dieser Node ist später praktisch für Frontend, API oder Chat-Ausgabe.
    """

    requirements = state.get("requirements", {})
    model_config = state.get("model_config", {})
    pulp_code = state.get("pulp_code", "")

    result = {
        "summary": _build_summary(requirements, model_config),
        "requirements": requirements,
        "selected_model": {
            "type": model_config.get("model_type"),
            "name": model_config.get("model_name"),
            "explanation": model_config.get("explanation"),
        },
        "model_markdown": render_model_markdown(model_config, requirements),
        "generated_code": pulp_code,
        "next_steps": [
            "Konkrete Parameter aus der Kundeneingabe extrahieren.",
            "Modellannahmen validieren.",
            "PuLP-Code mit echten Daten befüllen.",
            "Optimierungsmodell lösen.",
            "Ergebnis fachlich erklären und Empfehlung ausgeben.",
        ],
    }

    state["result"] = result

    return state


def _build_summary(requirements: dict, model_config: dict) -> str:
    decision = requirements.get("decision", "nicht angegeben")
    objective = requirements.get("objective", "nicht angegeben")
    model_name = model_config.get("model_name", "kein Modell ausgewählt")

    return (
        f"Für die Entscheidung '{decision}' mit dem Ziel '{objective}' "
        f"wurde als passender Startpunkt das Modell '{model_name}' ausgewählt."
    )