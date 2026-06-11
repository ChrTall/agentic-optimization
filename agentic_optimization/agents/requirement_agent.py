from agentic_optimization.state import GraphState


REQUIRED_FIELDS = [
    "decision",
    "objective",
    "demand_type",
    "constraints",
]


def requirement_agent(state: GraphState) -> GraphState:
    """
    Analysiert die Eingaben des Nutzers und prüft,
    ob die wichtigsten Informationen für ein Optimierungsmodell vorhanden sind.

    Aktuell ist diese Version bewusst regelbasiert.
    Später kann hier ein LLM eingebaut werden.
    """

    form_data = state.get("form_data", {})
    raw_input = state.get("raw_input", "")

    requirements = {
        "decision": form_data.get("decision"),
        "objective": form_data.get("objective"),
        "demand_type": form_data.get("demand_type"),
        "constraints": form_data.get("constraints"),
        "raw_description": raw_input,
    }

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if not requirements.get(field)
    ]

    state["requirements"] = requirements
    state["missing_fields"] = missing_fields

    return state