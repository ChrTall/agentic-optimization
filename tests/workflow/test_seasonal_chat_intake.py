"""Die Saisonbeschreibung wird in drei Nachrichten ergänzt, ohne Doppelabfragen."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

FIRST = "Wir verkaufen jedes Jahr zur Weihnachtszeit Weihnachtsbäume und müssen die Bestellmenge für dieses Jahr planen."
SECOND = "Wir wollen den zu erwartenden Gewinn maximieren und die optimale Bestellmenge von Weihnachtsbäumen bestimmen."
THIRD = "Während der Weihnachtszeit sind Nachbestellungen nicht möglich. Wir müssen es vorher planen."
CONTRARY = "Eigentlich wollen wir die Gesamtkosten reduzieren."
CORRECTION = "Korrektur: Statt Gewinnmaximierung möchte ich die Gesamtkosten reduzieren."


class SeasonalLLM:
    def __init__(self):
        self.profile_texts = []

    def invoke(self, messages):
        system, human = messages[0].content, messages[-1].content
        if system.startswith("Du bist ein freundlicher Assistent"):
            message = human.split("\n\nNACHRICHT DES NUTZERS:\n", 1)[1]
            context = json.loads(human.split("KONTEXT:\n", 1)[1].split("\n\nNACHRICHT DES NUTZERS:", 1)[0])
            answers = []
            if message == FIRST:
                answers = [{"question_id": "recommendation", "option_id": "order_quantity", "excerpt": "Bestellmenge"},
                           {"question_id": "product_scope", "option_id": "single_product", "excerpt": "Weihnachtsbäume"},
                           {"question_id": "procurement_pattern", "option_id": "one_time", "excerpt": "jedes Jahr zur Weihnachtszeit"}]
            elif message == SECOND:
                answers = [{"question_id": "primary_goal", "option_id": "maximize_profit",
                            "excerpt": "Gewinn maximieren", "explicit": True},
                           {"question_id": "recommendation", "option_id": "order_quantity",
                            "excerpt": "optimale Bestellmenge", "explicit": True}]
            elif message == THIRD:
                answers = [{"question_id": "procurement_pattern", "option_id": "one_time",
                            "excerpt": "Wir müssen es vorher planen.", "explicit": True},
                           {"question_id": "replenishment_during_period", "option_id": "not_possible",
                            "excerpt": "Nachbestellungen nicht möglich", "explicit": True}]
            elif message == CORRECTION:
                answers = [{"question_id": "primary_goal", "option_id": "minimize_cost",
                            "excerpt": "Gesamtkosten reduzieren", "change_intent": "correction"}]
            elif message == "ja":
                answers = [{"question_id": context["current_question"]["target"]["question_id"],
                            "option_id": "yes", "excerpt": message}]
            return AIMessage(content=json.dumps({"reply": "LLM behauptet ungeprüft: übernommen.",
                                                "use_message_as_description": True, "question_answers": answers}))
        if system.startswith("Prüfe unabhängig die Zuordnung"):
            payload = json.loads(human)
            message = payload["message"]
            findings = []
            for answer in payload["proposed_answers"]:
                findings.append({**answer, "verdict": "explicit", "intent": "answer",
                                 "evidence_basis": "direct_request" if answer["question_id"] == "recommendation" else "direct_statement"})
            if message == FIRST:
                # Die Textstelle existiert, stützt aber die Beschaffungsweise nicht.
                findings[-1]["verdict"] = "unclear"
            elif message == CONTRARY:
                findings = [{"question_id": "primary_goal", "option_id": "minimize_cost",
                             "excerpt": "Gesamtkosten reduzieren", "verdict": "explicit", "intent": "conflict",
                             "evidence_basis": "direct_statement"}]
            elif message == CORRECTION:
                findings[0]["intent"] = "correction"
            elif message == THIRD:
                # Dieser konservative Prüfer simuliert weiterhin eine offene
                # zweite Modellvoraussetzung; Annahmen prüft ein eigener Test.
                findings[-1]["verdict"] = "unclear"
            return AIMessage(content=json.dumps({"findings": findings}))
        if system.startswith("Du analysierst"):
            text = json.loads(human)["description"]
            self.profile_texts.append(text)
            inferences = []
            if text == FIRST:
                inferences = [{"field_id": "recommendation", "value": "order_quantity", "excerpt": "Bestellmenge"},
                              {"field_id": "product_scope", "value": "single_product", "excerpt": "Weihnachtsbäume"}]
            elif text == SECOND:
                inferences = [{"field_id": "primary_goal", "value": "maximize_profit", "excerpt": "Gewinn maximieren"}]
            elif text == THIRD:
                # Die neue Nachricht erwähnt das Gewinnziel absichtlich nicht erneut.
                inferences = [{"field_id": "procurement_pattern", "value": "one_time", "excerpt": "vorher planen"},
                              {"field_id": "replenishment_during_period", "value": "not_possible", "excerpt": "Nachbestellungen nicht möglich"}]
            elif text == CONTRARY:
                inferences = [{"field_id": "primary_goal", "value": "minimize_cost", "excerpt": "Gesamtkosten reduzieren"}]
            return AIMessage(content=json.dumps({"inferences": inferences, "ambiguities": []}))
        if system.startswith("Vergleiche die Geschäftssituation"):
            return AIMessage(content=json.dumps({"candidates": [{"core_id": "single_period_newsvendor",
                                                               "source_id": "raw_input", "excerpt": "Weihnachtsbäume"}],
                                                "reason": "Bestellentscheidung für eine Saison."}))
        if system.startswith("Prüfe ausschließlich"):
            return AIMessage(content='{"findings": []}')
        if system.startswith("Du extrahierst"):
            return AIMessage(content='{"products": [], "global_facts": {}}')
        if system.startswith("Du wählst"):
            return AIMessage(content=json.dumps({
                "core_id": "single_period_newsvendor", "selection_status": "complete", "objective_sense": "maximize",
                "objective_components": [{"component_id": cid, "reason": "Gewinnziel", "sources": ["raw_input"]}
                                         for cid in ["sales_revenue", "purchase_cost"]],
                "constraints": [], "unresolved_requirements": [],
            }))
        if system.startswith("Prüfe einen"):
            return AIMessage(content='{"rejected_constraints": [], "input_only_requirements": []}')
        raise AssertionError("Unerwarteter Agent: " + system[:70])


def say_confirmed(app, message):
    app.chat_input("chat_assistant_input").set_value(message).run()
    assert not app.exception
    if "chat_pending_proposal" in app.session_state:
        app.button("confirm_chat_proposal").click().run()
        assert not app.exception


def test_messages_keep_goal_and_require_targeted_structural_answers():
    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    llm = SeasonalLLM()
    app.session_state["llm"] = llm
    app.run()
    for message in (FIRST, SECOND, THIRD):
        say_confirmed(app, message)
        assert not app.exception
    assert app.session_state["step"] == "business_clarification"
    app.chat_input("chat_assistant_input").set_value("ja").run()
    assert not app.exception
    assert app.session_state["step"] == "model_input_form", app.session_state["graph_state"]["business_intake"]
    intake = app.session_state["graph_state"]["business_intake"]
    assert intake["profile"]["primary_goal"]["value"] == "maximize_profit"
    assert intake["profile"]["primary_goal"]["status"] == "confirmed"
    assert intake["applicability_user_answers"]["single_decision_period"] == "supported"
    assert intake["applicability_user_answers"]["no_replenishment_within_period"] == "supported"
    assert llm.profile_texts == []  # Ungeprüfte Profilextraktion darf nichts ergänzen.
    history = [m["content"] for m in app.session_state["chat_messages"] if m["role"] == "assistant"]
    assert sum("Was ist Ihnen hauptsächlich wichtig?" in message for message in history) <= 1
    second_index = next(i for i, m in enumerate(app.session_state["chat_messages"]) if m["content"] == SECOND)
    assert not any("Was ist Ihnen hauptsächlich wichtig?" in m["content"]
                   for m in app.session_state["chat_messages"][second_index + 1:])
    assert sum("Sind Nachbestellungen während des Verkaufszeitraums ausgeschlossen?" in message for message in history) == 1
    assert not any("LLM behauptet" in message for message in history)


def test_model_candidate_cannot_skip_unanswered_calculation_request():
    class NoCalculationRequestLLM(SeasonalLLM):
        def invoke(self, messages):
            response = super().invoke(messages)
            if messages[0].content.startswith("Du bist ein freundlicher Assistent"):
                decision = json.loads(response.content)
                decision["question_answers"] = [a for a in decision["question_answers"]
                                                if a["question_id"] != "recommendation"]
                return AIMessage(content=json.dumps(decision))
            return response

    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.session_state["llm"] = NoCalculationRequestLLM()
    app.run()
    for message in (FIRST, SECOND, THIRD):
        say_confirmed(app, message)
        assert not app.exception
        intake = app.session_state["graph_state"]["business_intake"]
        assert "recommendation" in intake["pending_question_ids"]
        assert app.session_state["step"] == "business_clarification"
    assert intake["profile"]["recommendation"]["status"] == "missing"


def test_empty_form_submission_keeps_verified_chat_mode_and_open_request():
    class NoRequestLLM(SeasonalLLM):
        def invoke(self, messages):
            response = super().invoke(messages)
            if messages[0].content.startswith("Du bist ein freundlicher Assistent"):
                decision = json.loads(response.content)
                decision["question_answers"] = [a for a in decision["question_answers"]
                                                if a["question_id"] != "recommendation"]
                return AIMessage(content=json.dumps(decision))
            return response

    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    llm = NoRequestLLM()
    app.session_state["llm"] = llm
    app.run()
    say_confirmed(app, FIRST)
    assert app.session_state["step"] == "business_clarification"
    assert app.radio[0].value is None
    next(b for b in app.button if b.label == "Weiter").click().run()
    assert not app.exception
    intake = app.session_state["graph_state"]["business_intake"]
    assert intake["chat_verified_mode"] is True
    assert intake["profile"]["recommendation"]["status"] == "missing"
    assert "recommendation" in intake["pending_question_ids"]
    assert app.session_state["step"] == "business_clarification"
    assert llm.profile_texts == []


def test_new_conflicting_goal_is_asked_about_instead_of_silently_replacing_saved_goal():
    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.session_state["llm"] = SeasonalLLM()
    app.run()
    for message in (FIRST, SECOND):
        say_confirmed(app, message)
    app.chat_input("chat_assistant_input").set_value(CONTRARY).run()
    assert not app.exception
    intake = app.session_state["graph_state"]["business_intake"]
    assert intake["profile"]["primary_goal"]["value"] == "maximize_profit"
    assert app.session_state["chat_pending_proposal"]["answers"][0]["option_id"] == "minimize_cost"
    app.button("confirm_chat_proposal").click().run()
    assert not app.exception
    assert app.session_state["graph_state"]["business_intake"]["profile"]["primary_goal"]["value"] == "minimize_cost"


def test_literal_but_unsupported_procurement_proposal_is_not_saved_or_acknowledged():
    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.session_state["llm"] = SeasonalLLM()
    app.run()
    say_confirmed(app, FIRST)
    assert not app.exception
    profile = app.session_state["graph_state"]["business_intake"]["profile"]
    assert profile["procurement_pattern"]["status"] == "missing"
    assert profile["recommendation"]["status"] == "confirmed"
    assert not any("Übernommen:" in m["content"] and "Einmal vor" in m["content"]
                   for m in app.session_state["chat_messages"])
    # Im Formular erscheint jeweils nur die aktuelle offene Frage.
    assert len(app.radio) == 1


def test_explicit_correction_updates_goal_and_overview_without_silent_conflict():
    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.session_state["llm"] = SeasonalLLM()
    app.run()
    for message in (FIRST, SECOND, CORRECTION):
        say_confirmed(app, message)
        assert not app.exception
    field = app.session_state["graph_state"]["business_intake"]["profile"]["primary_goal"]
    assert field["status"] == "confirmed"
    assert field["value"] == "minimize_cost"
    assert any("Ihre Angaben sind übernommen." in m["content"] for m in app.session_state["chat_messages"])
    assert any("Hauptziel: Gesamtkosten reduzieren" in text.value for text in app.text)


def test_previously_answered_goal_can_be_edited_in_the_summary():
    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.session_state["llm"] = SeasonalLLM()
    app.run()
    for message in (FIRST, SECOND):
        say_confirmed(app, message)
    app.button("edit_business_primary_goal").click().run()
    assert not app.exception
    assert app.radio("business_edit_value_primary_goal").value == "maximize_profit"
    app.radio("business_edit_value_primary_goal").set_value("minimize_cost")
    next(b for b in app.button if b.label == "Angabe aktualisieren").click().run()
    assert not app.exception
    assert app.session_state["graph_state"]["business_intake"]["profile"]["primary_goal"]["value"] == "minimize_cost"


def test_free_description_waits_for_confirmation_and_can_be_extended():
    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.session_state["llm"] = SeasonalLLM()
    app.run()
    app.chat_input("chat_assistant_input").set_value(FIRST).run()
    assert not app.exception
    assert app.session_state["step"] == "business_intake"
    assert app.session_state["graph_state"]["business_intake"]["profile"]["recommendation"]["status"] == "missing"
    assert "Soll ich diese Angaben übernehmen?" in app.session_state["chat_messages"][-1]["content"]
    app.chat_input("chat_assistant_input").set_value(SECOND).run()
    assert not app.exception
    pending = app.session_state["chat_pending_proposal"]
    assert FIRST in pending["description"] and SECOND in pending["description"]
    assert app.session_state["graph_state"]["business_intake"]["profile"]["primary_goal"]["status"] == "missing"
    app.button("confirm_chat_proposal").click().run()
    assert not app.exception
    assert "chat_pending_proposal" not in app.session_state
    profile = app.session_state["graph_state"]["business_intake"]["profile"]
    assert profile["recommendation"]["value"] == "order_quantity"
    assert profile["primary_goal"]["value"] == "maximize_profit"
    assert app.session_state["step"] == "business_clarification"
    follow_up = app.session_state["chat_messages"][-1]["content"]
    assert follow_up.startswith("Ihre Angaben sind übernommen.")
    assert "Ihre Geschäftssituation weiter beschreiben" in follow_up
    assert "Danke, das habe ich ausgewertet" not in follow_up


def test_proposal_can_be_corrected_before_any_profile_value_is_saved():
    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.session_state["llm"] = SeasonalLLM()
    app.run()
    app.chat_input("chat_assistant_input").set_value(SECOND).run()
    assert not app.exception
    app.button("edit_chat_proposal").click().run()
    assert not app.exception
    goal_radio = next(r for r in app.radio if r.key.startswith("proposal_edit_")
                      and r.label == "Was ist Ihnen hauptsächlich wichtig?")
    goal_radio.set_value("minimize_cost")
    next(b for b in app.button if b.label == "Änderungen übernehmen").click().run()
    assert not app.exception
    profile = app.session_state["graph_state"]["business_intake"]["profile"]
    assert profile["primary_goal"]["value"] == "minimize_cost"
    assert "chat_pending_proposal" not in app.session_state


def test_short_answer_to_current_question_does_not_require_second_confirmation():
    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.session_state["llm"] = SeasonalLLM()
    app.run()
    for message in (FIRST, SECOND, THIRD):
        say_confirmed(app, message)
    app.chat_input("chat_assistant_input").set_value("ja").run()
    assert not app.exception
    assert "chat_pending_proposal" not in app.session_state
    assert app.session_state["step"] == "model_input_form"


def test_discard_saves_nothing_and_next_proposal_can_be_confirmed_in_chat():
    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.session_state["llm"] = SeasonalLLM()
    app.run()
    app.chat_input("chat_assistant_input").set_value(SECOND).run()
    app.button("discard_chat_proposal").click().run()
    assert not app.exception
    assert "chat_pending_proposal" not in app.session_state
    assert app.session_state["graph_state"]["business_intake"]["profile"]["primary_goal"]["status"] == "missing"
    app.chat_input("chat_assistant_input").set_value(FIRST).run()
    app.chat_input("chat_assistant_input").set_value("ja").run()
    assert not app.exception
    assert "chat_pending_proposal" not in app.session_state
    profile = app.session_state["graph_state"]["business_intake"]["profile"]
    assert profile["recommendation"]["value"] == "order_quantity"
    assert profile["primary_goal"]["status"] == "missing"


def test_confirmed_derived_ordering_rule_does_not_need_another_confirmation():
    combined = "\n".join((FIRST, SECOND, THIRD))

    class AssumptionLLM(SeasonalLLM):
        def invoke(self, messages):
            if messages[0].content.startswith("Prüfe unabhängig die Zuordnung"):
                payload = json.loads(messages[-1].content)
                if payload["message"] == combined:
                    return AIMessage(content=json.dumps({"findings": [
                        {"question_id": qid, "option_id": oid, "excerpt": combined,
                         "verdict": "explicit", "intent": "answer", "evidence_basis": basis}
                        for qid, oid, basis in [
                            ("recommendation", "order_quantity", "direct_request"),
                            ("primary_goal", "maximize_profit", "direct_statement"),
                            ("product_scope", "single_product", "direct_statement"),
                            ("replenishment_during_period", "not_possible", "direct_statement"),
                            ("procurement_pattern", "one_time", "context_inference"),
                        ]]}))
            return super().invoke(messages)

    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.session_state["llm"] = AssumptionLLM()
    app.run()
    app.chat_input("chat_assistant_input").set_value(combined).run()
    assert not app.exception
    assert "Annahmen" in app.session_state["chat_messages"][-1]["content"]
    profile = app.session_state["graph_state"]["business_intake"]["profile"]
    assert profile["procurement_pattern"]["status"] == "missing"
    app.button("confirm_chat_proposal").click().run()
    assert not app.exception
    assert app.session_state["step"] == "model_input_form"
    intake = app.session_state["graph_state"]["business_intake"]
    assert intake["profile"]["procurement_pattern"]["status"] == "confirmed"
    assert intake["applicability_user_answers"]["single_decision_period"] == "supported"


def test_exact_current_choice_is_saved_without_assumption_confirmation():
    class MissingRequestLLM(SeasonalLLM):
        def invoke(self, messages):
            response = super().invoke(messages)
            if messages[0].content.startswith("Du bist ein freundlicher Assistent"):
                decision = json.loads(response.content)
                decision["question_answers"] = [a for a in decision["question_answers"]
                                                if a["question_id"] != "recommendation"]
                return AIMessage(content=json.dumps(decision))
            return response

    app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py"), default_timeout=30)
    app.session_state["llm"] = MissingRequestLLM()
    app.run()
    say_confirmed(app, FIRST)
    assert app.session_state["graph_state"]["business_intake"]["pending_question_ids"][0] == "recommendation"
    app.chat_input("chat_assistant_input").set_value("bestellmenge bestimmen").run()
    assert not app.exception
    assert "chat_pending_proposal" not in app.session_state
    assert app.session_state["graph_state"]["business_intake"]["profile"]["recommendation"]["value"] == "order_quantity"
    assert "Annahmen" not in app.session_state["chat_messages"][-1]["content"]
