from agentic_optimization.configuration.llm_config import load_llm_configuration
from langchain_core.language_models import BaseChatModel

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
        "messages": [],
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
        "messages": [],
    }

    result_state = app.invoke(state)
    print_result(result_state)


def print_result(result_state: dict):
    if result_state.get("clarification_question"):
        print("Rückfrage:")
        print(result_state["clarification_question"])
        return

    result = result_state["result"]

    print("Zusammenfassung:")
    print(result["summary"])
    print()

    print("Ausgewähltes Modell:")
    print(result["selected_model"]["name"])
    print()

    print("Begründung:")
    print(result["selected_model"]["explanation"])
    print()

    print("Nächste Schritte:")
    for step in result["next_steps"]:
        print("-", step)

    print()
    print("Generierter Code:")
    print(result["generated_code"])


if __name__ == "__main__":
    g_llm_config = load_llm_configuration()
    g_llm = create_chat_model(g_llm_config)
    run_example_complete_input(g_llm)
    # run_example_missing_input()
