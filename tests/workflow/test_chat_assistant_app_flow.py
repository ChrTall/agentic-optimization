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

RECALL_QUOTES = {
    "ordering_cost": "Jede Bestellung kostet 120 €",
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
        if (context.get("current_question") or {}).get("target", {}).get(
            "kind"
        ) == "inputs_complete":
            return {
                "reply": "Dann geht es weiter zur Überprüfung.",
                "action": "continue_to_review",
            }
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

        if system.startswith("Du erhältst alle bisherigen Aussagen"):
            payload = json.loads(human)
            answers = []
            for item in payload["open_items"]:
                name = (item.get("field") or {}).get("name")
                quote = RECALL_QUOTES.get(name)
                if quote and quote in payload["statements"]:
                    answers.append({
                        "id": item["id"],
                        "value": VALUES[name],
                        "quote": quote,
                    })
            return AIMessage(content=json.dumps({"answers": answers}))

        if system.startswith("Korrigiere ausschließlich"):
            return AIMessage(content=json.dumps({"repairs": []}))

        if system.startswith("Nenne das Produkt"):
            return AIMessage(content=json.dumps({
                "name": "Gartenschlauch",
                "excerpt": "Artikel",
            }))

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
    # Eine frühere Analyse ist fehlgeschlagen; Formular 1 bietet die
    # Wiederholung an (die Beschreibung stammt aus dem Chat).
    state["business_intake"]["analysis_error"] = "Vorübergehender Fehler"
    app.session_state["graph_state"] = state
    app.session_state["raw_problem_description"] = raw
    app.run()
    next(
        button for button in app.button
        if button.label == "Erneut auswerten"
    ).click().run()
    assert not app.exception
    assert app.session_state["graph_state"]["business_intake"][
        "pending_answers"
    ][0]["option_id"] == "minimize_cost"

    llm.fail_profile = False
    next(
        button for button in app.button
        if button.label == "Erneut auswerten"
    ).click().run()
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
    products = app.session_state["graph_state"]["modular_model"]["products"]
    assert [product["product_id"] for product in products] == [
        "Gartenschlauch"
    ]

    _say(app, "3500 Stück im Jahr, 120 € je Bestellung, 12 € Lager.")
    # Formular 3 bleibt sichtbar; der Chat weist auf Zusatzbausteine hin.
    assert app.session_state["step"] == "model_input_form"
    assert any(
        "Alle Pflichtangaben sind vollständig" in message["content"]
        for message in app.session_state["chat_messages"]
    )
    _say(app, "Passt so, weiter.")
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
    assert app.session_state["chat_product_rows_version"] == (
        previous_editor_version + 1
    )
    assert "llm" in app.session_state


def test_values_from_earlier_messages_are_not_asked_again() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = FakeLLM()
    app.run()

    _say(
        app,
        "Wir bestellen einen Artikel regelmäßig nach und wollen "
        "wissen, wie viel wir jeweils bestellen sollen. "
        "Jede Bestellung kostet 120 €.",
    )
    _say(app, "Vor allem sollen die Kosten niedrig bleiben.")
    if app.session_state["step"] == "business_clarification":
        _say(app, "Ja, wir bestellen regelmäßig nach.")
    assert app.session_state["step"] == "model_input_form"

    facts = app.session_state["graph_state"]["modular_model"][
        "specification"
    ]["products"][0]["facts"]
    assert facts["ordering_cost"] == 120

    # Eine Sprechblase kann mehrere Abschnitte enthalten.
    assistant_messages = [
        part
        for message in app.session_state["chat_messages"]
        if message["role"] == "assistant"
        for part in message.get("parts", [message["content"]])
    ]
    assert any(
        "Aus Ihren bisherigen Angaben übernommen" in message
        for message in assistant_messages
    )
    assert not any(
        "Bestellkosten" in message and "?" in message
        for message in assistant_messages
    )


class UnhelpfulInputLLM(FakeLLM):
    """Antwortet im Eingabeschritt nie mit einem Wert."""

    def invoke(self, messages):
        human = messages[-1].content
        if (
            messages[0].content.startswith("Du bist ein freundlicher")
            and "keine Ahnung" in human
        ):
            return AIMessage(content=json.dumps({
                "reply": "Kein Problem. Wie hoch ist der Wert ungefähr?",
            }))
        return super().invoke(messages)


def test_offers_way_out_once_after_repeated_unanswered_question() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = UnhelpfulInputLLM()
    app.run()

    _say(
        app,
        "Wir bestellen einen Artikel regelmäßig nach und wollen "
        "wissen, wie viel wir jeweils bestellen sollen.",
    )
    _say(app, "Vor allem sollen die Kosten niedrig bleiben.")
    if app.session_state["step"] == "business_clarification":
        _say(app, "Ja, wir bestellen regelmäßig nach.")
    assert app.session_state["step"] == "model_input_form"

    for _ in range(3):
        _say(app, "keine Ahnung")

    way_out_messages = [
        message["content"]
        for message in app.session_state["chat_messages"]
        if "hänge ich fest" in message["content"]
    ]
    assert len(way_out_messages) == 1


def test_answers_business_question_with_button() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = FakeLLM()
    app.run()

    _say(
        app,
        "Wir bestellen einen Artikel regelmäßig nach und wollen "
        "wissen, wie viel wir jeweils bestellen sollen.",
    )
    assert app.session_state["step"] == "business_clarification"

    app.button("chat_quick_reply:business:primary_goal:minimize_cost").click().run()
    assert not app.exception

    profile = app.session_state["graph_state"]["business_intake"]["profile"]
    assert profile["primary_goal"]["value"] == "minimize_cost"
    assert any(
        message["role"] == "user"
        and message["content"] == "Gesamtkosten reduzieren"
        for message in app.session_state["chat_messages"]
    )


def test_detected_requirements_can_be_confirmed_discarded_and_changed() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = FakeLLM()
    app.run()

    _say(
        app,
        "Wir bestellen einen Artikel regelmäßig nach und wollen "
        "wissen, wie viel wir jeweils bestellen sollen.",
    )

    def profile() -> dict:
        return app.session_state["graph_state"]["business_intake"]["profile"]

    assert profile()["procurement_pattern"]["status"] == "inferred"
    assert any(
        "Erkannt: " in message["content"]
        for message in app.session_state["chat_messages"]
    )

    # Bestätigen
    app.button(
        "requirements_profile:procurement_pattern_recurring_confirm"
    ).click().run()
    assert not app.exception
    assert profile()["procurement_pattern"]["status"] == "confirmed"

    # Ändern über das Auswahlfeld
    app.selectbox(
        "requirements_profile:recommendation_order_quantity_select"
    ).set_value("both").run()
    assert not app.exception
    assert profile()["recommendation"]["value"] == (
        "order_quantity_and_reorder_point"
    )
    assert profile()["recommendation"]["status"] == "confirmed"

    # Verwerfen: Die Angabe gilt wieder als offen und verschwindet aus
    # der Übersicht. Ob nachgefragt wird, entscheidet die Fragenplanung.
    app.button(
        "requirements_profile:recommendation_both_discard"
    ).click().run()
    assert not app.exception
    assert profile()["recommendation"]["status"] == "missing"
    assert not any(
        "requirements_profile:recommendation" in (button.key or "")
        for button in app.button
    )



class ComponentChatLLM(FakeLLM):
    """Nennt im Eingabeschritt eine zusätzliche Höchstbestellmenge."""

    def invoke(self, messages):
        human = messages[-1].content
        if (
            messages[0].content.startswith("Du bist ein freundlicher")
            and "höchstens 500" in human
        ):
            return AIMessage(content=json.dumps({
                "reply": "Ich ergänze eine Höchstbestellmenge.",
                "add_components": ["maximum_order_quantity"],
            }))
        return super().invoke(messages)


def test_chat_adds_compatible_component_after_required_inputs() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = ComponentChatLLM()
    app.run()

    _say(
        app,
        "Wir bestellen einen Artikel regelmäßig nach und wollen "
        "wissen, wie viel wir jeweils bestellen sollen.",
    )
    _say(app, "Vor allem sollen die Kosten niedrig bleiben.")
    if app.session_state["step"] == "business_clarification":
        _say(app, "Ja, wir bestellen regelmäßig nach.")
    _say(app, "3500 Stück im Jahr, 120 € je Bestellung, 12 € Lager.")
    assert app.session_state["step"] == "model_input_form"

    _say(app, "Wir dürfen höchstens 500 Stück bestellen.")
    specification = app.session_state["graph_state"]["modular_model"][
        "specification"
    ]
    assert [c["component_id"] for c in specification["constraints"]] == [
        "maximum_order_quantity"
    ]
    assert app.session_state["step"] == "model_input_form"


def test_form_input_is_applied_and_chat_asks_next_question() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = FakeLLM()
    app.run()

    _say(
        app,
        "Wir bestellen einen Artikel regelmäßig nach und wollen "
        "wissen, wie viel wir jeweils bestellen sollen.",
    )
    _say(app, "Vor allem sollen die Kosten niedrig bleiben.")
    if app.session_state["step"] == "business_clarification":
        _say(app, "Ja, wir bestellen regelmäßig nach.")
    assert app.session_state["step"] == "model_input_form"

    def last_assistant_message() -> str:
        return [
            message["content"]
            for message in app.session_state["chat_messages"]
            if message["role"] == "assistant"
        ][-1]

    assert "Erwartete Nachfrage" in last_assistant_message()
    user_messages_before = sum(
        message["role"] == "user"
        for message in app.session_state["chat_messages"]
    )

    # Wert direkt im Formular eintragen – ohne Button, ohne Chatnachricht.
    demand_input = next(
        widget for widget in app.number_input
        if widget.label.startswith("Erwartete Nachfrage")
    )
    demand_input.set_value(3500.0).run()
    assert not app.exception

    forecast = app.session_state["graph_state"]["modular_model"][
        "specification"
    ]["products"][0]["forecast"]
    assert forecast["expected_value"] == 3500.0
    assert "Erwartete Nachfrage" not in last_assistant_message()
    assert "Prognosezeitraum" in last_assistant_message()
    assert sum(
        message["role"] == "user"
        for message in app.session_state["chat_messages"]
    ) == user_messages_before


def test_form_answer_is_applied_without_button() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = FakeLLM()
    app.run()

    _say(
        app,
        "Wir bestellen einen Artikel regelmäßig nach und wollen "
        "wissen, wie viel wir jeweils bestellen sollen.",
    )
    assert app.session_state["step"] == "business_clarification"

    app.radio("business_clarification_primary_goal").set_value(
        "minimize_cost"
    ).run()
    assert not app.exception

    profile = app.session_state["graph_state"]["business_intake"]["profile"]
    assert profile["primary_goal"]["value"] == "minimize_cost"
    assert "Was ist Ihnen hauptsächlich wichtig?" not in [
        message["content"].split("\n")[0].strip("*")
        for message in app.session_state["chat_messages"][-1:]
    ]


def _reach_input_form(app: AppTest) -> None:
    _say(
        app,
        "Wir bestellen einen Artikel regelmäßig nach und wollen "
        "wissen, wie viel wir jeweils bestellen sollen.",
    )
    _say(app, "Vor allem sollen die Kosten niedrig bleiben.")
    if app.session_state["step"] == "business_clarification":
        _say(app, "Ja, wir bestellen regelmäßig nach.")
    assert app.session_state["step"] == "model_input_form"


def _specification(app: AppTest) -> dict:
    return app.session_state["graph_state"]["modular_model"]["specification"]


class SilentComponentLLM(FakeLLM):
    """Der Assistent selbst erkennt die Höchstbestellmenge nicht."""

    def invoke(self, messages):
        human = messages[-1].content
        if (
            messages[0].content.startswith("Du bist ein freundlicher")
            and "Höchstbestellmenge" in human.rsplit("NACHRICHT DES NUTZERS:", 1)[-1]
        ):
            return AIMessage(content=json.dumps({"reply": "Alles klar."}))
        return super().invoke(messages)


def test_named_component_is_added_and_value_filled() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = SilentComponentLLM()
    app.run()
    _reach_input_form(app)

    _say(app, "Es gibt eine Höchstbestellmenge von 250 Stück.")

    constraints = _specification(app)["constraints"]
    assert [c["component_id"] for c in constraints] == ["maximum_order_quantity"]
    assert constraints[0]["parameters"] == {"quantity": 250}
    last = app.session_state["chat_messages"][-1]["content"]
    assert "Höchstbestellmenge" in last and "250" in last


def test_component_from_description_is_added_on_input_form() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = FakeLLM()
    app.run()
    _say(
        app,
        "Wir bestellen einen Artikel regelmäßig nach und wollen "
        "wissen, wie viel wir jeweils bestellen sollen. Die "
        "Höchstbestellmenge liegt bei 300 Stück.",
    )
    _say(app, "Vor allem sollen die Kosten niedrig bleiben.")
    if app.session_state["step"] == "business_clarification":
        _say(app, "Ja, wir bestellen regelmäßig nach.")

    constraints = _specification(app)["constraints"]
    assert [c["component_id"] for c in constraints] == ["maximum_order_quantity"]
    assert constraints[0]["parameters"] == {"quantity": 300}

    # Im Formular abgewählt -> wird nicht erneut hinzugefügt.
    app.checkbox(
        key="modular_component_toggle_maximum_order_quantity_1"
    ).uncheck().run()
    assert not app.exception
    assert _specification(app)["constraints"] == []
    app.run()
    assert _specification(app)["constraints"] == []


class WrongFieldLLM(FakeLLM):
    """Ordnet Lagerkosten fälschlich auch den Bestellkosten zu."""

    def invoke(self, messages):
        system = messages[0].content
        human = messages[-1].content
        if system.startswith("Du bist ein freundlicher") and (
            "Lagerkosten" in human.rsplit("NACHRICHT DES NUTZERS:", 1)[-1]
        ):
            context = json.loads(
                human.split("KONTEXT:\n", 1)[1].split(
                    "\n\nNACHRICHT DES NUTZERS:", 1
                )[0]
            )
            return AIMessage(content=json.dumps({
                "reply": "Notiert.",
                "input_values": [
                    {"input_id": item["input_id"], "value": 12}
                    for item in context["missing_inputs"]
                    if item["name"] in {"ordering_cost", "holding_cost"}
                ],
            }))
        if system.startswith("Du erhältst alle bisherigen Aussagen"):
            payload = json.loads(human)
            return AIMessage(content=json.dumps({"answers": [
                {"id": item["id"], "value": 12,
                 "quote": "Die Lagerkosten betragen 12 €"}
                for item in payload["open_items"]
                if (item.get("field") or {}).get("name") == "ordering_cost"
                and "Lagerkosten" in payload["statements"]
            ]}))
        return super().invoke(messages)


def test_value_is_not_assigned_to_a_different_field() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = WrongFieldLLM()
    app.run()
    _reach_input_form(app)

    _say(app, "Die Lagerkosten betragen 12 € pro Stück und Jahr.")

    facts = _specification(app)["products"][0]["facts"]
    assert facts.get("holding_cost") == 12
    assert facts.get("ordering_cost") is None


class StrictRecallLLM(SilentComponentLLM):
    """Erinnern liefert unbelegte bzw. falsch zugeordnete Werte."""

    def invoke(self, messages):
        system = messages[0].content
        human = messages[-1].content
        if system.startswith("Du bist ein freundlicher") and any(
            phrase in human.rsplit("NACHRICHT DES NUTZERS:", 1)[-1]
            for phrase in ("Restwert", "jedes Jahr")
        ):
            return AIMessage(content=json.dumps({"reply": "Notiert."}))
        if system.startswith("Du erhältst alle bisherigen Aussagen"):
            payload = json.loads(human)
            answers = []
            for item in payload["open_items"]:
                name = (item.get("field") or {}).get("name")
                if name == "quantity" and "Restwert" in payload["statements"]:
                    answers.append({"id": item["id"], "value": 3,
                                    "quote": "Restwert von 3 Euro"})
                if name == "horizon_days" and "jedes Jahr" in payload["statements"]:
                    answers.append({"id": item["id"], "value": 365,
                                    "quote": "jedes Jahr"})
            return AIMessage(content=json.dumps({"answers": answers}))
        return super().invoke(messages)


def test_recall_rejects_values_that_belong_elsewhere_or_lack_a_number() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = StrictRecallLLM()
    app.run()
    _reach_input_form(app)

    _say(app, "Es liegt ein Restwert von 3 Euro vor. Wir bestellen jedes Jahr.")
    app.checkbox(
        key="modular_component_toggle_maximum_order_quantity_0"
    ).check().run()
    assert not app.exception

    specification = _specification(app)
    assert specification["constraints"][0]["parameters"] == {}
    assert "horizon_days" not in specification["products"][0]["forecast"]


def test_existing_value_is_updated_from_chat() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = SilentComponentLLM()
    app.run()
    _reach_input_form(app)

    _say(app, "Es gibt eine Höchstbestellmenge von 250 Stück.")
    assert _specification(app)["constraints"][0]["parameters"] == {
        "quantity": 250
    }

    _say(app, "Die Höchstbestellmenge beträgt 270 Stück.")
    assert _specification(app)["constraints"][0]["parameters"] == {
        "quantity": 270
    }
    # Das Formularfeld zeigt den neuen Wert.
    assert any(
        widget.value == 270
        for widget in app.number_input
        if "maximum_order_quantity" in (widget.key or "")
    )


class UpdateValueLLM(FakeLLM):
    """Der Assistent ändert einen bereits ausgefüllten Wert (k-Kennung)."""

    def invoke(self, messages):
        system = messages[0].content
        human = messages[-1].content
        if system.startswith("Du bist ein freundlicher") and (
            "Lagerkosten sind jetzt 15" in human.rsplit("NACHRICHT DES NUTZERS:", 1)[-1]
        ):
            context = json.loads(
                human.split("KONTEXT:\n", 1)[1].split(
                    "\n\nNACHRICHT DES NUTZERS:", 1
                )[0]
            )
            return AIMessage(content=json.dumps({
                "reply": "Geändert.",
                "input_values": [
                    {"input_id": item["input_id"], "value": 15}
                    for item in context["filled_inputs"]
                    if item["name"] == "holding_cost"
                ],
            }))
        return super().invoke(messages)


def test_chat_changes_an_already_filled_value() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = UpdateValueLLM()
    app.run()
    _reach_input_form(app)
    _say(app, "3500 Stück im Jahr, 120 € je Bestellung, 12 € Lager.")
    assert _specification(app)["products"][0]["facts"]["holding_cost"] == 12

    _say(app, "Die Lagerkosten sind jetzt 15 € pro Stück und Jahr.")
    assert _specification(app)["products"][0]["facts"]["holding_cost"] == 15


class CountingRecallLLM(FakeLLM):
    recall_calls = 0

    def invoke(self, messages):
        if messages[0].content.startswith("Du erhältst alle bisherigen Aussagen"):
            type(self).recall_calls += 1
        return super().invoke(messages)


def test_renaming_the_product_does_not_trigger_recall_again() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["llm"] = CountingRecallLLM()
    app.run()
    _reach_input_form(app)
    calls = CountingRecallLLM.recall_calls

    name_input = next(
        widget for widget in app.text_input
        if (widget.key or "").startswith("modular_product_name")
    )
    name_input.set_value("Weihnachtsbaum").run()
    assert not app.exception

    assert _specification(app)["products"][0]["product_id"] == "Weihnachtsbaum"
    assert _specification(app)["products"][0]["forecast"] == {}
    assert CountingRecallLLM.recall_calls == calls
