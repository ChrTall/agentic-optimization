from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agentic_optimization.agents.modular_workflow_nodes import (
    apply_modular_model_inputs_node,
    confirm_modular_requirement_exclusions_node,
    create_modular_model_preparation_node,
    execute_modular_model_node,
    create_modular_product_schema_node,
    create_modular_review_explanation_node,
    create_modular_result_explanation_node,
)
from agentic_optimization.state import (
    EVENT_MODULAR_INPUTS_SUBMITTED,
    EVENT_MODULAR_MODEL_PREPARATION_REQUESTED,
    EVENT_MODULAR_OPTIMIZATION_REQUESTED,
    EVENT_MODULAR_PRODUCT_SCHEMA_REQUESTED,
    EVENT_MODULAR_REVIEW_EXPLANATION_REQUESTED,
    EVENT_MODULAR_RESULT_EXPLANATION_REQUESTED,
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
from agentic_optimization.agents.business_intake_nodes import (
    create_business_intake_node,
    initialize_modular_core_node,
)


def route_workflow(state: GraphState) -> str:
    """Entscheidet, an welcher Stelle der Graph gestartet wird."""

    event = get_event(state)

    if event == EVENT_BUSINESS_INTAKE_REQUESTED:
        return "business_intake"

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
    # Einstiegspunkt: abhängig vom Event an der richtigen Stelle starten.
    graph.add_conditional_edges(
        START,
        route_workflow,
        {
            "business_intake": "business_intake",
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

    # Modulare Aufrufe sind eigenständige Graph-Durchläufe.
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

    return graph.compile()
