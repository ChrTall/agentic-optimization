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
        question_ids = {
            question["question_id"] for question in context["open_questions"]
        }
        if "core_condition:recurring_orders" in question_ids:
            return {
                "reply": "Ja, wir bestellen regelmäßig nach.",
                "question_answers": [
                    {
                        "question_id": "core_condition:recurring_orders",
                        "option_id": "yes",
                    }
                ],
            }
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
                if not item.get("activates_component")
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

        if system.startswith("Prüfe ausschließlich"):
            return AIMessage(content=json.dumps({"findings": []}))

        if system.startswith("Vergleiche die Geschäftssituation"):
            return AIMessage(content=json.dumps({
                "candidates": [{
                    "core_id": "eoq",
                    "source_id": "raw_input",
                    "excerpt": "regelmäßig nach",
                }],
                "reason": "Wiederkehrende Bestellungen.",
            }))

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


class RetryProfileLLM(FakeLLM):
    fail_profile = True

    def invoke(self, messages):
        if self.fail_profile and messages[0].content.startswith("Du analysierst"):
            raise ValueError("Profilanalyse vorübergehend nicht verfügbar")
        return super().invoke(messages)


def _say(app: AppTest, text: str) -> None:
    app.chat_input("chat_assistant_input").set_value(text).run()
    assert not app.exception, app.exception


def test_form_retry_replays_answer_after_profile_failure() -> None:
    llm = RetryProfileLLM()
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = llm
    app.run()
    raw = (
        "Wir bestellen einen Artikel regelmäßig nach und möchten wissen, "
        "wie viel wir bestellen sollen."
    )
    state = app.session_state["graph_state"]
    state["raw_input"] = raw
    state["business_intake"]["pending_question_ids"] = ["primary_goal"]
    state["business_intake"]["asked_question_ids"] = ["primary_goal"]
    state["business_intake"]["pending_answers"] = [{
        "question_id": "primary_goal",
        "option_id": "minimize_cost",
        "source_id": "business_clarification_form",
    }]
    app.session_state["graph_state"] = state
    app.session_state["raw_problem_description"] = raw
    app.run()
    next(
        button for button in app.button
        if button.label == "Angaben auswerten"
    ).click().run()
    assert not app.exception
    assert app.session_state["graph_state"]["business_intake"][
        "pending_answers"
    ][0]["option_id"] == "minimize_cost"

    llm.fail_profile = False
    assert app.session_state["step"] == "preparation_failed"
    app.button("retry_preparation").click().run()
    assert not app.exception
    profile = app.session_state["graph_state"]["business_intake"]["profile"]
    assert profile["primary_goal"]["value"] == "minimize_cost"


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
    if app.session_state["step"] == "business_clarification":
        _say(app, "Ja, wir bestellen regelmäßig nach.")
    assert app.session_state["step"] == "model_input_form"

    _say(app, "3500 Stück im Jahr, 120 € je Bestellung, 12 € Lager.")
    assert app.session_state["step"] == "model_input_form"
    assert app.number_input("modular_forecast_product_1_expected_value").value == 3500
    assert app.button("submit_modular_model_inputs").disabled
    app.checkbox("modular_additions_confirmed").check().run()
    app.button("submit_modular_model_inputs").click().run()
    assert not app.exception
    assert app.session_state["step"] == "review"
    assert not any(
        "Aus Ihrer Beschreibung abgeleitet" in item.value
        for item in app.markdown
    )

    pending_conditions = app.session_state["graph_state"][
        "modular_model"
    ]["pending_core_assumptions"]
    assert len(pending_conditions) > 1
    app.checkbox("confirm_modular_review").check().run()
    assert app.button("start_modular_optimization").disabled
    for index, condition_id in enumerate(pending_conditions):
        app.checkbox(
            "confirm_core_assumption_" + condition_id
        ).check().run()
        if index < len(pending_conditions) - 1:
            assert app.button("start_modular_optimization").disabled
    assert not app.button("start_modular_optimization").disabled

    _say(app, "Passt, bitte berechnen.")
    assert app.session_state["step"] == "result"

    result = app.session_state["graph_state"]["modular_model"][
        "optimization_result"
    ]
    assert result["status"] == "optimal"
    assert round(result["decisions"][0]["value"]) == 265

    # A new optimization must not reuse answers or uploads from the old one.
    app.session_state["business_question_recommendation"] = "order_quantity"
    app.session_state["business_clarification_primary_goal"] = "minimize_cost"
    app.session_state["business_intake_documents"] = "previous upload"
    previous_editor_version = app.session_state[
        "chat_product_rows_version"
    ]
    next(
        button for button in app.button
        if button.label == "Neue Optimierung starten"
    ).click().run()

    assert not app.exception
    assert app.session_state["step"] == "business_intake"
    assert app.session_state["raw_problem_description"] == ""
    assert "raw_input" not in app.session_state["graph_state"]
    assert app.session_state["business_question_recommendation"] is None
    assert "business_clarification_primary_goal" not in app.session_state
    assert app.session_state["business_intake_documents"] in (None, [])
    assert app.session_state["chat_product_rows_version"] == (
        previous_editor_version + 1
    )
    assert "llm" in app.session_state
