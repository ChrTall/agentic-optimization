from agentic_optimization.graph import build_graph


def run_example_complete_input():
    app = build_graph()

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


def run_example_missing_input():
    app = build_graph()

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
    run_example_complete_input()
    # run_example_missing_input()