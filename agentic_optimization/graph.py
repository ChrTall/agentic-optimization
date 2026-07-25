from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agentic_optimization.agents.clarification import clarification_node
from agentic_optimization.agents.code_agent import generate_pulp_code
from agentic_optimization.agents.constraint_mapping_agent import (
    create_constraint_mapping_agent,
)
from agentic_optimization.agents.modeling_agent import modeling_agent
from agentic_optimization.agents.requirement_agent import create_requirement_agent
from agentic_optimization.agents.result_node import result_node
from agentic_optimization.agents.validation_agent import (
    detect_model_specific_inputs,
    validate_general_requirements,
    validate_model_specific_inputs,
    validate_structured_constraints,
)
from agentic_optimization.state import (
    EVENT_GENERAL_CLARIFICATION_SUBMITTED,
    EVENT_MODEL_INPUTS_SUBMITTED,
    EVENT_NEW_PROBLEM,
    WORKFLOW_CODE_GENERATED,
    WORKFLOW_CODE_GENERATION_FAILED,
    WORKFLOW_GENERAL_INPUTS_VALID,
    WORKFLOW_MODEL_INPUTS_INVALID,
    WORKFLOW_MODEL_INPUTS_VALID,
    WORKFLOW_MODEL_SPECIFIC_INPUTS_COMPLETE,
    WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_READY_FOR_GENERATION,
    GraphState,
    get_event,
    get_workflow_status,
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

    if status == WORKFLOW_MODEL_INPUTS_INVALID:
        return "model_inputs_invalid"

    raise ValueError(f"Unexpected workflow_status after model input detection: {status}")


def route_after_model_input_validation(state: GraphState) -> str:
    """Routet nach der Validierung der modellspezifischen Eingaben."""

    status = get_workflow_status(state)

    if status == WORKFLOW_MODEL_INPUTS_INVALID:
        return "model_inputs_invalid"

    if status == WORKFLOW_MODEL_INPUTS_VALID:
        return "validate_constraints"

    raise ValueError(f"Unexpected workflow_status after model input validation: {status}")


def route_after_constraint_validation(state: GraphState) -> str:
    status = get_workflow_status(state)

    if status == WORKFLOW_MODEL_INPUTS_INVALID:
        return "constraints_invalid"

    if status == WORKFLOW_READY_FOR_GENERATION:
        return "generate_code"

    raise ValueError(
        f"Unexpected workflow_status after constraint validation: {status}"
    )


def route_after_code_generation(state: GraphState) -> str:
    status = get_workflow_status(state)

    if status == WORKFLOW_CODE_GENERATED:
        return "prepare_result"

    if status == WORKFLOW_CODE_GENERATION_FAILED:
        return "generation_failed"

    raise ValueError(
        f"Unexpected workflow_status after code generation: {status}"
    )

def route_after_constraint_mapping(
    state: GraphState,
) -> str:
    mapping_errors = state.get(
        "constraint_mapping_errors",
        [],
    )

    if mapping_errors:
        return "mapping_failed"

    return "continue"


def build_graph(llm: BaseChatModel):
    """
    Baut den LangGraph-Workflow für das Optimierungssystem.

    Bei jedem app.invoke()-Aufruf entscheidet das Event,
    an welcher Stelle der Workflow fortgesetzt wird.

    Der Graph kann vorzeitig bei END stoppen, z.B. bei Rückfragen
    oder fehlenden Modellinputs. Dann enthält der State kein `result`;
    `control.workflow_status` signalisiert, wie fortzufahren ist.
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
    graph.add_node("validate_structured_constraints", validate_structured_constraints)

    graph.add_node(
            "constraint_mapping_agent",
            create_constraint_mapping_agent(llm),
        )
    
    # Bereitet das finale Ergebnis für die UI auf.
    graph.add_node("result_node", result_node)
    graph.add_node("generate_pulp_code", generate_pulp_code)

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

    graph.add_edge(
        "modeling_agent",
        "constraint_mapping_agent",
    )

    graph.add_conditional_edges(
        "constraint_mapping_agent",
        route_after_constraint_mapping,
        {
            "mapping_failed": END,
            "continue": "detect_model_specific_inputs",
        },
    )

    # Fehlen Eingaben, endet der Graph und die UI zeigt das Modellformular.
    graph.add_conditional_edges(
        "detect_model_specific_inputs",
        route_after_model_selection,
        {
            "needs_model_inputs": END,
            "validate_model_inputs": "validate_model_specific_inputs",
            "model_inputs_invalid": END,
        },
    )

    # Ungültige Modellinputs stoppen den Graph,
    # gültige werden anschließend auf Constraints geprüft.
    graph.add_conditional_edges(
        "validate_model_specific_inputs",
        route_after_model_input_validation,
        {
            "model_inputs_invalid": END,
            "validate_constraints": "validate_structured_constraints",
        },
    )

    graph.add_conditional_edges(
        "validate_structured_constraints",
        route_after_constraint_validation,
        {
            "constraints_invalid": END,
            "generate_code": "generate_pulp_code",
        },
    )
    graph.add_conditional_edges(
        "generate_pulp_code",
        route_after_code_generation,
        {
            "prepare_result": "result_node",
            "generation_failed": END,
        },
    )

    # Bereitet die Modellvorschau für Formular 4 auf.
    graph.add_edge("result_node", END)

    return graph.compile()