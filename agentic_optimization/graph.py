from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END

from agentic_optimization.state import GraphState
from agentic_optimization.agents.requirement_agent import create_requirement_agent
from agentic_optimization.agents.clarification import clarification_node
from agentic_optimization.agents.modeling_agent import modeling_agent
from agentic_optimization.agents.code_agent import code_agent
from agentic_optimization.agents.result_node import result_node


def route_after_requirements(state: GraphState) -> str:
    """
    Entscheidet, ob Rückfragen nötig sind oder ob der Workflow weiterlaufen kann.
    """

    if state.get("missing_fields"):
        return "clarify"

    return "complete"


def build_graph(llm: BaseChatModel):
    """
    Baut den LangGraph-Workflow für das Optimierungssystem.
    """

    graph = StateGraph(GraphState)

    graph.add_node("requirement_agent", create_requirement_agent(llm))
    graph.add_node("clarification", clarification_node)
    graph.add_node("modeling_agent", modeling_agent)
    graph.add_node("code_agent", code_agent)
    graph.add_node("result_node", result_node)

    graph.set_entry_point("requirement_agent")

    graph.add_conditional_edges(
        "requirement_agent",
        route_after_requirements,
        {
            "clarify": "clarification",
            "complete": "modeling_agent",
        },
    )

    graph.add_edge("clarification", END)
    graph.add_edge("modeling_agent", "code_agent")
    graph.add_edge("code_agent", "result_node")
    graph.add_edge("result_node", END)

    return graph.compile()
