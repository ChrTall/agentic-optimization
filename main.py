from agentic_optimization.configuration.llm_config import load_llm_configuration
from langchain_core.language_models import BaseChatModel
from agentic_optimization.state import get_workflow_status
from agentic_optimization.graph import build_graph
from agentic_optimization.llm.factory import create_chat_model


def run_example_complete_input(llm: BaseChatModel):
    app = build_graph(llm)

    state = {
        "raw_input": (
            "Wir verkaufen saisonale Produkte und wollen die optimale "
            "Bestellmenge bestimmen."
        ),
        "form_data": {
            "decision": "Bestellmenge",
            "objective": "Gewinn maximieren",
            "demand_type": "unsicher",
            "constraints": "begrenztes Budget und begrenzter Lagerplatz",
        },
    }

    result_state = app.invoke(state)
    print_result(result_state)


def run_example_missing_input(llm: BaseChatModel):
    app = build_graph(llm)

    state = {
        "raw_input": (
            "Wir verkaufen saisonale Produkte und wollen die optimale "
            "Bestellmenge bestimmen."
        ),
        "form_data": {
            "decision": "Bestellmenge",
            "objective": "Gewinn maximieren",
            "demand_type": "unsicher",
        },
    }

    result_state = app.invoke(state)
    print_result(result_state)


def print_result(result_state: dict) -> None:
    clarification_question = result_state.get(
        "clarification_question"
    )

    if clarification_question:
        print("Rückfrage:")
        print(clarification_question)
        return

    result = result_state.get("result")

    if result is None:
        print("Workflowstatus:")
        print(get_workflow_status(result_state))
        return

    print("Zusammenfassung:")
    print(result["summary"])
    print()

    selected_model = result["selected_model"]

    print("Ausgewähltes Modell:")
    print(
        f"{selected_model['family_name']} / "
        f"{selected_model['variant_name']}"
    )
    print()

    optimization_errors = result.get(
        "optimization_errors",
        [],
    )

    if optimization_errors:
        print("Optimierungsfehler:")

        for error in optimization_errors:
            print("-", error)

        return

    print("Optimierungsergebnis:")
    print(result.get("optimization_result"))


if __name__ == "__main__":
    g_llm_config = load_llm_configuration()
    g_llm = create_chat_model(g_llm_config)
    run_example_complete_input(g_llm)
    # run_example_missing_input()
