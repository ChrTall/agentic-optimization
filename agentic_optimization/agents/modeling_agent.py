from agentic_optimization.state import (
    GraphState,
    WORKFLOW_MODEL_SELECTED,
    with_control,
)


def modeling_agent(state: GraphState) -> GraphState:
    """
    Wählt auf Basis der erkannten Anforderungen einen passenden Modelltyp aus.

    Dieser Node trifft nur die Modellauswahl.
    Ob danach modellspezifische Eingaben fehlen, entscheidet der nächste Node.
    """

    requirements = state.get("requirements", {})

    decision = str(requirements.get("decision", "")).lower()
    objective = str(requirements.get("objective", "")).lower()
    demand_type = str(requirements.get("demand_type", "")).lower()
    constraints = str(requirements.get("constraints", "")).lower()
    raw_description = str(requirements.get("raw_description", "")).lower()

    combined_text = " ".join([
        decision,
        objective,
        demand_type,
        constraints,
        raw_description,
    ])

    if any(keyword in combined_text for keyword in ["unsicher", "saisonal", "nachfrage", "bestellmenge"]):
        model_type = "newsvendor"
        model_name = "Newsvendor-Modell"
        explanation = (
            "Das Problem wirkt wie ein Bestellmengenproblem unter unsicherer oder saisonaler Nachfrage. "
            "Dafür eignet sich grundsätzlich ein Newsvendor-Modell."
        )

    elif any(keyword in combined_text for keyword in ["produktion", "kapazität", "ressourcen", "kosten minimieren"]):
        model_type = "linear_programming"
        model_name = "Lineares Optimierungsmodell"
        explanation = (
            "Das Problem enthält Hinweise auf Produktions-, Kapazitäts- oder Ressourcenplanung. "
            "Dafür eignet sich ein lineares Optimierungsmodell."
        )

    elif any(keyword in combined_text for keyword in ["route", "tour", "transport", "lieferung"]):
        model_type = "routing"
        model_name = "Routing- oder Transportmodell"
        explanation = (
            "Das Problem enthält Hinweise auf Routen, Transporte oder Lieferungen. "
            "Dafür könnte ein Routing- oder Transportmodell geeignet sein."
        )

    else:
        model_type = "generic_linear_programming"
        model_name = "Allgemeines lineares Optimierungsmodell"
        explanation = (
            "Es wurde kein spezieller Modelltyp eindeutig erkannt. "
            "Als Startpunkt wird ein allgemeines lineares Optimierungsmodell vorgeschlagen."
        )

    model_config = {
        "model_type": model_type,
        "model_name": model_name,
        "explanation": explanation,
        "objective": requirements.get("objective"),
        "decision": requirements.get("decision"),
        "constraints": requirements.get("constraints"),
    }

    return with_control(
        {
            **state,
            "selected_model": model_type,
            "model_config": model_config,
        },
        event=None,
        workflow_status=WORKFLOW_MODEL_SELECTED,
    )