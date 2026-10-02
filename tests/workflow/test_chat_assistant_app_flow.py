"""End-to-End: kompletter Ablauf nur über den Chat-Assistenten."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[2] / "app.py"

PROPOSAL = {
    "core_id": "eoq",
    "selection_status": "complete",
    "objective_sense": "minimize",
    "objective_components": [
        {
            "component_id": "annual_ordering_cost",
            "reason": "Bestellkosten",
            "sources": ["raw_input"],
        },
        {
            "component_id": "annual_holding_cost",
            "reason": "Lagerkosten",
            "sources": ["raw_input"],
        },
    ],
    "constraints": [],
    "unresolved_requirements": [],
}

VALUES = {
    "expected_value": 3500,
    "horizon_days": 365,
    "ordering_cost": 120,
    "holding_cost": 12,
}


def _chat_reply(context: dict) -> dict:
    step = context["step"]

    if step == "business_intake":
        return {
            "reply": "Danke, ich schaue mir das an.",
            "use_message_as_description": True,
        }

    if step == "business_clarification":
        return {
            "reply": "Verstanden – Kosten senken.",
            "question_answers": [
                {
                    "question_id": "primary_goal",
                    "option_id": "minimize_cost",
                }
            ],
        }

    if step == "model_input_form":
        return {
            "reply": "Werte übernommen.",
            "input_values": [
                {
                    "input_id": item["input_id"],
                    "value": VALUES.get(item["name"], 1),
                }
                for item in context["missing_inputs"]
            ],
        }

    if step == "review":
        return {
            "reply": "Ich starte die Berechnung.",
            "action": "start_optimization",
        }

    return {
        "reply": "Gern erkläre ich das.",
        "action": "explain_result",
    }


class FakeLLM:
    """Simuliert alle Agenten anhand ihres Systemprompts."""

    def invoke(self, messages):
        system = messages[0].content
        human = messages[-1].content

        if system.startswith("Du bist ein freundlicher Assistent"):
            raw_context = human.split("KONTEXT:\n", 1)[1].split(
                "\n\nNACHRICHT DES NUTZERS:",
                1,
            )[0]
            return AIMessage(
                content=json.dumps(_chat_reply(json.loads(raw_context)))
            )

        if system.startswith("Du analysierst"):
            return AIMessage(
                content=json.dumps(
                    {
                        "inferences": [
                            {
                                "field_id": "recommendation",
                                "value": "order_quantity",
                                "excerpt": "wie viel",
                            },
                            {
                                "field_id": "procurement_pattern",
                                "value": "recurring",
                                "excerpt": "regelmäßig",
                            },
                            {
                                "field_id": "product_scope",
                                "value": "single_product",
                                "excerpt": "Artikel",
                            },
                        ],
                        "ambiguities": [],
                    }
                )
            )

        if system.startswith("Du wählst"):
            return AIMessage(content=json.dumps(PROPOSAL))

        if system.startswith("Du prüfst"):
            return AIMessage(
                content=json.dumps({"uncovered_requirements": []})
            )

        if system.startswith("Du extrahierst"):
            return AIMessage(
                content=json.dumps({"products": [], "global_facts": {}})
            )

        if system.startswith("Du erklärst"):
            return AIMessage(
                content=json.dumps({"explanation": "Erklärungstext."})
            )

        raise AssertionError(f"Unerwarteter Prompt: {system[:80]}")


def _say(app: AppTest, text: str) -> None:
    app.chat_input("chat_assistant_input").set_value(text).run()
    assert not app.exception, app.exception


def test_complete_workflow_through_chat() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = FakeLLM()
    app.run()

    assert not app.exception
    assert app.session_state["chat_messages"][0]["role"] == "assistant"

    _say(
        app,
        "Wir bestellen einen Artikel regelmäßig nach und wollen "
        "wissen, wie viel wir jeweils bestellen sollen.",
    )
    assert app.session_state["step"] == "business_clarification"

    _say(app, "Vor allem sollen die Kosten niedrig bleiben.")
    assert app.session_state["step"] == "model_input_form"

    _say(app, "3500 Stück im Jahr, 120 € je Bestellung, 12 € Lager.")
    assert app.session_state["step"] == "review"

    _say(app, "Passt, bitte berechnen.")
    assert app.session_state["step"] == "result"

    result = app.session_state["graph_state"]["modular_model"][
        "optimization_result"
    ]
    assert result["status"] == "optimal"
    assert round(result["decisions"][0]["value"]) == 265