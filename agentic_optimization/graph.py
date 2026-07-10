from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from agentic_optimization.state import (
    WORKFLOW_GENERAL_INPUTS_VALID,
    WORKFLOW_MODEL_INPUTS_INVALID,
    WORKFLOW_MODEL_SPECIFIC_INPUTS_COMPLETE,
    WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_READY_FOR_RESULT,
    GraphState,
    EVENT_NEW_PROBLEM,
    EVENT_GENERAL_CLARIFICATION_SUBMITTED,
    EVENT_MODEL_INPUTS_SUBMITTED,
    get_event,
    get_workflow_status,
)
from agentic_optimization.agents.requirement_agent import create_requirement_agent
from agentic_optimization.agents.clarification import clarification_node
from agentic_optimization.agents.modeling_agent import modeling_agent
from agentic_optimization.agents.result_node import result_node

from agentic_optimization.agents.validation_agent import (
    detect_model_specific_inputs,
    validate_general_requirements,
    validate_model_specific_inputs,
)


def route_workflow(state: GraphState) -> str:
    """Entscheidet, an welcher Stelle der Graph gestartet wird."""

    event = get_event(state)

    if event == EVENT_NEW_PROBLEM:
        return "requirement_agent"

    if event == EVENT_GENERAL_CLARIFICATION_SUBMITTED:
        return "requirement_agent"

    if event == EVENT_MODEL_INPUTS_SUBMITTED:
        return "validate_model_specific_inputs"

    raise ValueError(f"Unexpected workflow event: {event}")


def route_after_general_validation(state: GraphState) -> str:
    """Routet nach der allgemeinen Validierung."""

    status = get_workflow_status(state)

    if status == WORKFLOW_NEEDS_GENERAL_CLARIFICATION:
        return "clarify"

    if status == WORKFLOW_GENERAL_INPUTS_VALID:
        return "select_model"

    raise ValueError(f"Unexpected workflow_status after general validation: {status}")


def route_after_model_selection(state: GraphState) -> str:
    """Routet nach der Modellauswahl."""

    status = get_workflow_status(state)

    if status == WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS:
        return "needs_model_inputs"

    if status == WORKFLOW_MODEL_SPECIFIC_INPUTS_COMPLETE:
        return "validate_model_inputs"

    raise ValueError(f"Unexpected workflow_status after model input detection: {status}")


def route_after_model_input_validation(state: GraphState) -> str:
    """Routet nach der Validierung der modellspezifischen Eingaben."""

    status = get_workflow_status(state)

    if status == WORKFLOW_MODEL_INPUTS_INVALID:
        return "model_inputs_invalid"

    if status == WORKFLOW_READY_FOR_RESULT:
        return "prepare_result"

    raise ValueError(f"Unexpected workflow_status after model input validation: {status}")


def build_graph(llm: BaseChatModel):
    """
    Baut den LangGraph-Workflow für das Optimierungssystem.

    Bei jedem app.invoke()-Aufruf entscheidet das Event,
    an welcher Stelle der Workflow fortgesetzt wird.
    """

    graph = StateGraph(GraphState)

    graph.add_node("requirement_agent", create_requirement_agent(llm))
    graph.add_node("clarification", clarification_node)

    # Bestehender modeling_agent kann erstmal weiterverwendet werden,
    # sollte aber perspektivisch eher "select_model" heißen.
    graph.add_node("modeling_agent", modeling_agent)

    # Validierungs- und Input-Erkennungs-Nodes
    graph.add_node("validate_general_requirements", validate_general_requirements)
    graph.add_node("detect_model_specific_inputs", detect_model_specific_inputs)
    graph.add_node("validate_model_specific_inputs", validate_model_specific_inputs)

    # Bereitet das finale Ergebnis für die UI auf.
    graph.add_node("result_node", result_node)

    # Einstiegspunkt: abhängig vom Event an der richtigen Stelle starten.
    graph.add_conditional_edges(
        START,
        route_workflow,
        {
            "requirement_agent": "requirement_agent",
            "validate_model_specific_inputs": "validate_model_specific_inputs",
        },
    )

    # Nach der Requirement-Extraktion immer allgemein validieren.
    graph.add_edge("requirement_agent", "validate_general_requirements")

    # Je nach Validierung entweder Rückfrage stellen oder Modell auswählen.
    graph.add_conditional_edges(
        "validate_general_requirements",
        route_after_general_validation,
        {
            "clarify": "clarification",
            "select_model": "modeling_agent",
        },
    )

    # Bei Rückfragen endet der Graph, die UI wartet auf Nutzereingabe.
    graph.add_edge("clarification", END)

    # Nach der Modellauswahl prüfen, ob modellabhängige Eingaben fehlen.
    graph.add_edge("modeling_agent", "detect_model_specific_inputs")

    # Fehlen Eingaben, endet der Graph und die UI zeigt das Modellformular.
    graph.add_conditional_edges(
        "detect_model_specific_inputs",
        route_after_model_selection,
        {
            "needs_model_inputs": END,
            "validate_model_inputs": "validate_model_specific_inputs",
        },
    )

    # Ungültige Modellinputs stoppen den Graph, gültige führen zum Ergebnis.
    graph.add_conditional_edges(
        "validate_model_specific_inputs",
        route_after_model_input_validation,
        {
            "model_inputs_invalid": END,
            "prepare_result": "result_node",
        },
    )

    # Nach der Ergebnisaufbereitung ist der Workflow abgeschlossen.
    graph.add_edge("result_node", END)

    return graph.compile()