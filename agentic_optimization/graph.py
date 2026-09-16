from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agentic_optimization.agents.clarification import clarification_node
from agentic_optimization.agents.code_agent import optimize_model
from agentic_optimization.agents.constraint_mapping_agent import (
    create_constraint_mapping_agent,
)
from agentic_optimization.agents.model_input_extraction_agent import (
    create_model_input_extraction_agent,
)
from agentic_optimization.agents.modeling_agent import modeling_agent
from agentic_optimization.agents.modular_workflow_nodes import (
    apply_modular_model_inputs_node,
    confirm_modular_requirement_exclusions_node,
    create_modular_model_preparation_node,
    execute_modular_model_node,
    create_modular_product_schema_node,
    create_modular_review_explanation_node,
    create_modular_result_explanation_node,
)
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
    EVENT_OPTIMIZATION_REQUESTED,
    EVENT_MODULAR_INPUTS_SUBMITTED,
    EVENT_MODULAR_MODEL_PREPARATION_REQUESTED,
    EVENT_MODULAR_OPTIMIZATION_REQUESTED,
    EVENT_MODULAR_PRODUCT_SCHEMA_REQUESTED,
    EVENT_MODULAR_REVIEW_EXPLANATION_REQUESTED,
    EVENT_MODULAR_RESULT_EXPLANATION_REQUESTED,
    WORKFLOW_GENERAL_INPUTS_VALID,
    WORKFLOW_MODEL_INPUTS_INVALID,
    WORKFLOW_MODEL_INPUTS_VALID,
    WORKFLOW_MODEL_SELECTED,
    WORKFLOW_MODEL_SPECIFIC_INPUTS_COMPLETE,
    WORKFLOW_MODEL_UNSUPPORTED,
    WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_OPTIMIZATION_COMPLETED,
    WORKFLOW_OPTIMIZATION_FAILED,
    WORKFLOW_READY_FOR_REVIEW,
    EVENT_BUSINESS_INTAKE_REQUESTED,
    EVENT_MODULAR_REQUIREMENT_EXCLUSIONS_CONFIRMED,
    WORKFLOW_BUSINESS_CASE_UNSUPPORTED,
    WORKFLOW_BUSINESS_CORE_SELECTED,
    WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED,
    WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
    GraphState,
    get_event,
    get_workflow_status,
)
from agentic_optimization.agents.modular_state_adapter import (
    initialize_modular_model_state,
)
from agentic_optimization.agents.business_intake_nodes import (
    create_business_intake_node,
    initialize_modular_core_node,
)

def route_workflow(state: GraphState) -> str:
    """Entscheidet, an welcher Stelle der Graph gestartet wird."""

    event = get_event(state)

    if event == EVENT_BUSINESS_INTAKE_REQUESTED:
        return "business_intake"

    if event == EVENT_NEW_PROBLEM:
        return "requirement_agent"

    if event == EVENT_GENERAL_CLARIFICATION_SUBMITTED:
        return "requirement_agent"

    if event == EVENT_MODEL_INPUTS_SUBMITTED:
        return "validate_model_specific_inputs"

    if event == EVENT_OPTIMIZATION_REQUESTED:
        return "optimize_model"

    if event == EVENT_MODULAR_PRODUCT_SCHEMA_REQUESTED:
        return "prepare_modular_product_schema"

    if event == EVENT_MODULAR_REVIEW_EXPLANATION_REQUESTED:
        return "prepare_modular_review_explanation"

    if event == EVENT_MODULAR_RESULT_EXPLANATION_REQUESTED:
        return "prepare_modular_result_explanation"

    if event == EVENT_MODULAR_MODEL_PREPARATION_REQUESTED:
        return "prepare_modular_model"

    if event == EVENT_MODULAR_REQUIREMENT_EXCLUSIONS_CONFIRMED:
        return "confirm_modular_requirement_exclusions"

    if event == EVENT_MODULAR_INPUTS_SUBMITTED:
        return "apply_modular_model_inputs"

    if event == EVENT_MODULAR_OPTIMIZATION_REQUESTED:
        return "execute_modular_model"

    raise ValueError(f"Unexpected workflow event: {event}")


def route_after_business_intake(
    state: GraphState,
) -> str:
    """Routet nach der Analyse des Geschäftsprofils."""

    status = get_workflow_status(state)

    if status == WORKFLOW_BUSINESS_CORE_SELECTED:
        return "initialize_core"

    if status in {
        WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
        WORKFLOW_BUSINESS_CASE_UNSUPPORTED,
        WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED,
    }:
        return "stop"

    raise ValueError(
        "Unexpected workflow_status after business intake: "
        f"{status}"
    )


def route_after_general_validation(state: GraphState) -> str:
    """Routet nach der allgemeinen Validierung."""

    status = get_workflow_status(state)

    if status == WORKFLOW_NEEDS_GENERAL_CLARIFICATION:
        return "clarify"

    if status == WORKFLOW_GENERAL_INPUTS_VALID:
        return "select_model"

    raise ValueError(f"Unexpected workflow_status after general validation: {status}")


def route_after_modeling(
    state: GraphState,
) -> str:
    """Routet nach der deterministischen Modellauswahl."""

    status = get_workflow_status(state)

    if status == WORKFLOW_MODEL_SELECTED:
        if state.get("use_modular_pipeline", False):
            return "initialize_modular_model"

        return "model_selected"

    if status == WORKFLOW_MODEL_UNSUPPORTED:
        return "model_unsupported"

    raise ValueError(
        "Unexpected workflow_status after model selection: "
        f"{status}"
    )


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
    """Routet nach der Validierung der aktiven Constraints."""

    status = get_workflow_status(state)

    if status == WORKFLOW_MODEL_INPUTS_INVALID:
        return "constraints_invalid"

    if status == WORKFLOW_READY_FOR_REVIEW:
        return "review_ready"

    raise ValueError(
        "Unexpected workflow_status after constraint validation: "
        f"{status}"
    )


def route_after_optimization(state: GraphState) -> str:
    """Routet jedes Optimierungsergebnis zur Ergebnisaufbereitung."""

    status = get_workflow_status(state)

    if status == WORKFLOW_OPTIMIZATION_COMPLETED:
        return "prepare_result"

    if status == WORKFLOW_OPTIMIZATION_FAILED:
        return "prepare_result"

    raise ValueError(
        "Unexpected workflow_status after optimization: "
        f"{status}"
    )


def route_after_constraint_mapping(state: GraphState) -> str:
    """Beendet den Graph bei Mappingfehlern."""

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

    graph.add_node(
        "business_intake",
        create_business_intake_node(llm),
    )
    graph.add_node(
        "initialize_business_modular_core",
        initialize_modular_core_node,
    )
    graph.add_node(
        "prepare_modular_product_schema",
        create_modular_product_schema_node(llm),
    )
    graph.add_node(
        "prepare_modular_review_explanation",
        create_modular_review_explanation_node(llm),
    )
    graph.add_node(
        "prepare_modular_result_explanation",
        create_modular_result_explanation_node(llm),
    )
    graph.add_node(
        "prepare_modular_model",
        create_modular_model_preparation_node(llm),
    )
    graph.add_node(
        "confirm_modular_requirement_exclusions",
        confirm_modular_requirement_exclusions_node,
    )
    graph.add_node(
        "apply_modular_model_inputs",
        apply_modular_model_inputs_node,
    )
    graph.add_node(
        "execute_modular_model",
        execute_modular_model_node,
    )
    graph.add_node(
        "initialize_modular_model",
        initialize_modular_model_state,
    )

    graph.add_node("requirement_agent", create_requirement_agent(llm))
    graph.add_node("clarification", clarification_node)

    # Die Modellauswahl erfolgt deterministisch anhand der validierten
    # Requirements und der in der Registry definierten Auswahlkriterien.
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
    graph.add_node(
        "model_input_extraction_agent",
        create_model_input_extraction_agent(llm),
    )

    # Der Result-Node bündelt Solverausgabe und Workflowdaten zu
    # einem UI-unabhängigen finalen Ergebnisobjekt.
    graph.add_node("result_node", result_node)
    graph.add_node("optimize_model", optimize_model)

    # Einstiegspunkt: abhängig vom Event an der richtigen Stelle starten.
    graph.add_conditional_edges(
        START,
        route_workflow,
        {
            "business_intake": "business_intake",
            "requirement_agent": "requirement_agent",
            "validate_model_specific_inputs": (
                "validate_model_specific_inputs"
            ),
            "optimize_model": "optimize_model",
            "prepare_modular_product_schema": (
                "prepare_modular_product_schema"
            ),
            "prepare_modular_review_explanation": (
                "prepare_modular_review_explanation"
            ),
            "prepare_modular_result_explanation": (
                "prepare_modular_result_explanation"
            ),
            "prepare_modular_model": (
                "prepare_modular_model"
            ),
            "apply_modular_model_inputs": (
                "apply_modular_model_inputs"
            ),
            "execute_modular_model": (
                "execute_modular_model"
            ),
            "confirm_modular_requirement_exclusions": (
                "confirm_modular_requirement_exclusions"
            ),
        },
    )

    graph.add_conditional_edges(
        "business_intake",
        route_after_business_intake,
        {
            "initialize_core": (
                "initialize_business_modular_core"
            ),
            "stop": END,
        },
    )
    graph.add_edge(
        "initialize_business_modular_core",
        END,
    )

    # Modulare Aufrufe sind zunächst eigenständige Graph-Durchläufe.
    graph.add_edge(
        "prepare_modular_product_schema",
        END,
    )
    graph.add_edge(
        "prepare_modular_review_explanation",
        END,
    )
    graph.add_edge(
        "prepare_modular_result_explanation",
        END,
    )
    graph.add_edge("prepare_modular_model", END)
    graph.add_edge(
        "confirm_modular_requirement_exclusions",
        END,
    )
    graph.add_edge("apply_modular_model_inputs", END)
    graph.add_edge("execute_modular_model", END)



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

    # Für nicht unterstützte Problemstrukturen endet der Graph
    # bewusst. Die UI zeigt den Modellfehler und ermöglicht die
    # Bearbeitung der grundlegenden Angaben oder Beschreibung.
    graph.add_conditional_edges(
        "modeling_agent",
        route_after_modeling,
        {
            "model_selected": "constraint_mapping_agent",
            "initialize_modular_model": (
                "initialize_modular_model"
            ),
            "model_unsupported": END,
        },
    )

    graph.add_edge(
        "initialize_modular_model",
        "prepare_modular_model",
    )

    # Das LLM erzeugt zunächst unbestätigte Constraint-Kandidaten.
    # Mappingfehler stoppen den Graph; danach werden ebenfalls unbestätigte
    # Kandidaten für die registrierten Modellfelder aus dem Freitext gelesen.
    graph.add_conditional_edges(
        "constraint_mapping_agent",
        route_after_constraint_mapping,
        {
            "mapping_failed": END,
            "continue": "model_input_extraction_agent",
        },
    )
    graph.add_edge(
        "model_input_extraction_agent",
        "detect_model_specific_inputs",
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

    # Nach erfolgreicher Validierung endet dieser Graph-Durchlauf
    # beim Review. Die Optimierung startet erst durch ein separates
    # Nutzer-Event nach der ausdrücklichen Bestätigung.
    graph.add_conditional_edges(
        "validate_structured_constraints",
        route_after_constraint_validation,
        {
            "constraints_invalid": END,
            "review_ready": END,
        },
    )

    # Sowohl erfolgreiche als auch fehlgeschlagene Optimierungen
    # werden für Formular 5 aufbereitet, damit Solverfehler und
    # Infeasibility verständlich dargestellt werden können.
    graph.add_conditional_edges(
        "optimize_model",
        route_after_optimization,
        {
            "prepare_result": "result_node",
        },
    )

    # Nach der finalen Ergebnisaufbereitung übernimmt die UI
    # anhand des Workflow-Status die Darstellung von Formular 5.
    graph.add_edge("result_node", END)

    return graph.compile()
