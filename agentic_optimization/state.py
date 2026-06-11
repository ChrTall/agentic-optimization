from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    """
    Gemeinsamer Zustand des LangGraph-Workflows.

    Dieser State wird von Knoten zu Knoten weitergegeben.
    Jeder Agent liest relevante Felder aus und ergänzt neue Ergebnisse.
    """

    # Eingaben aus UI / Formular / Chat
    raw_input: str
    form_data: dict[str, Any]
    messages: list[dict[str, str]]

    # Ergebnis des Requirement Agents
    requirements: dict[str, Any]
    missing_fields: list[str]
    clarification_question: str

    # Ergebnis des Modeling Agents
    model_config: dict[str, Any]

    # Ergebnis des Code Agents
    pulp_code: str

    # Ergebnis des Result Nodes
    result: dict[str, Any]