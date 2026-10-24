"""Formular 3: Ergänzungen, erhaltene Eingaben und Chat-Korrekturen."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from agentic_optimization.ui.chat_assistant_builder import build_chat_context
from agentic_optimization.models.validator import validate_model_specification

APP_PATH = Path(__file__).parents[2] / "app.py"


def make_app(product_count=1, *, incomplete=False):
    specification = {
        "core_id": "single_period_newsvendor",
        "products": [{
            "product_id": f"p{i}",
            "forecast": {"expected_value": 100, "horizon_days": 30,
                         "quantiles": [{"level": 0.1, "value": 70},
                                       {"level": 0.5, "value": 100},
                                       {"level": 0.9, "value": 140}]},
            "facts": {"selling_price": 25, "purchase_cost": 10},
        } for i in range(product_count)],
        "global_facts": {},
        "objective": {"sense": "maximize", "component_ids": ["sales_revenue", "purchase_cost"]},
        "constraints": [],
    }
    proposal = {
        "core_id": specification["core_id"], "selection_status": "complete",
        "objective_sense": "maximize", "unresolved_requirements": [],
        "objective_components": [{"component_id": cid, "reason": "Gewinnziel",
                                  "sources": ["raw_input"]}
                                 for cid in specification["objective"]["component_ids"]],
        "constraints": [],
    }
    if incomplete:
        for product in specification["products"]:
            product["forecast"] = {}
            product["facts"] = {}
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["step"] = "model_input_form"
    app.session_state["graph_state"] = {"modular_model": {
        "specification": specification, "proposal": proposal,
        "validation": validate_model_specification(specification),
    }}
    app.run()
    assert not app.exception
    return app


class InputLLM:
    def __init__(self, values):
        self.values = values

    def invoke(self, messages):
        context = json.loads(messages[-1].content.split("KONTEXT:\n", 1)[1].split(
            "\n\nNACHRICHT DES NUTZERS:", 1)[0])
        values = []
        for name, product_id, component, value in self.values:
            item = next(item for item in context["missing_inputs"]
                        if item["name"] == name and item["product_id"] == product_id
                        and item["activates_component"] == component)
            values.append({"input_id": item["input_id"], "value": value})
        return AIMessage(content=json.dumps({"reply": "Werte eingetragen.", "input_values": values}))


def test_add_remove_preserves_values_and_resets_confirmation():
    app = make_app()
    app.number_input("modular_fact_p0_selling_price").set_value(27).run()
    app.checkbox("modular_additions_confirmed").check().run()
    app.checkbox("modular_component_objective:salvage_revenue").check().run()
    assert not app.exception
    assert app.number_input("modular_fact_p0_selling_price").value == 27
    assert app.number_input("modular_fact_p0_salvage_value").value is None
    area = next(expander for expander in app.expander
                if expander.label == "Weitere Angaben berücksichtigen")
    assert area.number_input("modular_fact_p0_salvage_value").value is None
    assert not app.selectbox
    assert not app.checkbox("modular_additions_confirmed").value
    app.number_input("modular_fact_p0_salvage_value").set_value(3).run()
    app.checkbox("modular_additions_confirmed").check().run()
    app.checkbox("modular_component_objective:salvage_revenue").uncheck().run()
    assert not app.exception
    assert app.number_input("modular_fact_p0_selling_price").value == 27
    assert not app.checkbox("modular_additions_confirmed").value
    spec = app.session_state["graph_state"]["modular_model"]["specification"]
    assert "salvage_revenue" not in spec["objective"]["component_ids"]
    app.checkbox("modular_component_objective:salvage_revenue").check().run()
    assert not app.exception
    assert app.number_input("modular_fact_p0_salvage_value").value == 3


def test_chat_adds_and_corrects_rest_value_without_leaving_form():
    app = make_app()
    app.number_input("modular_fact_p0_selling_price").set_value(27).run()
    app.checkbox("modular_additions_confirmed").check().run()
    app.session_state["llm"] = InputLLM([
        ("salvage_value", "p0", "objective:salvage_revenue", 3),
    ])
    app.chat_input("chat_assistant_input").set_value("Restwert = 3").run()
    assert not app.exception
    assert app.session_state["step"] == "model_input_form"
    assert app.number_input("modular_fact_p0_salvage_value").value == 3
    assert app.checkbox("modular_component_objective:salvage_revenue").value
    assert app.number_input("modular_fact_p0_selling_price").value == 27
    assert not app.checkbox("modular_additions_confirmed").value
    app.session_state["llm"] = InputLLM([("salvage_value", "p0", None, 4)])
    app.chat_input("chat_assistant_input").set_value("Korrektur: Restwert = 4").run()
    assert not app.exception
    assert app.number_input("modular_fact_p0_salvage_value").value == 4
    assert app.session_state["step"] == "model_input_form"


def test_chat_adds_multiple_product_constraints_with_correct_targets():
    app = make_app(2)
    app.session_state["llm"] = InputLLM([
        ("quantity", "p0", "constraint:maximum_order_quantity:p0", 120),
        ("quantity", "p1", "constraint:maximum_order_quantity:p1", 150),
        ("budget", None, "constraint:budget_limit:None", 2000),
    ])
    app.chat_input("chat_assistant_input").set_value("Höchstens 120 für p0, 150 für p1, Budget 2000").run()
    assert not app.exception
    spec = app.session_state["graph_state"]["modular_model"]["specification"]
    assert [(c["component_id"], c.get("product_id"), c["parameters"])
            for c in spec["constraints"]] == [
        ("maximum_order_quantity", "p0", {"quantity": 120}),
        ("maximum_order_quantity", "p1", {"quantity": 150}),
        ("budget_limit", None, {"budget": 2000}),
    ]
    assert validate_model_specification(spec)["is_valid"]
    proposal = app.session_state["graph_state"]["modular_model"]["proposal"]
    assert proposal["constraints"][1]["parameters"] == {"quantity": 150}


def test_question_id_points_to_missing_field_after_completed_values():
    app = make_app()
    graph = app.session_state["graph_state"]
    graph["modular_model"]["specification"]["products"][0]["facts"].pop("purchase_cost")
    context = build_chat_context("model_input_form", graph)
    target = context["current_question"]["target"]["input_id"]
    item = next(item for item in context["missing_inputs"] if item["input_id"] == target)
    assert item["name"] == "purchase_cost"
    assert item["activates_component"] is None


def test_added_component_reaches_review_with_entered_values():
    app = make_app()
    app.checkbox("modular_component_constraint:maximum_order_quantity:p0").check().run()
    app.number_input("modular_constraint_maximum_order_quantity_p0_quantity").set_value(120).run()
    app.checkbox("modular_additions_confirmed").check().run()

    class ExplanationLLM:
        def invoke(self, messages):
            return AIMessage(content=json.dumps({"explanation": "Angaben zur Prüfung."}))

    app.session_state["llm"] = ExplanationLLM()
    app.button("submit_modular_model_inputs").click().run()
    assert not app.exception
    assert app.session_state["step"] == "review"
    assert app.session_state["graph_state"]["modular_model"]["specification"]["constraints"] == [
        {"component_id": "maximum_order_quantity", "product_id": "p0", "parameters": {"quantity": 120}},
    ]


def test_minimum_and_maximum_remain_selectable_and_conflicting_values_are_reported():
    app = make_app()
    minimum = "modular_component_constraint:minimum_order_quantity:p0"
    maximum = "modular_component_constraint:maximum_order_quantity:p0"
    app.checkbox(minimum).check().run()
    assert not app.checkbox(maximum).value
    app.checkbox(maximum).check().run()
    app.number_input("modular_constraint_minimum_order_quantity_p0_quantity").set_value(150).run()
    app.number_input("modular_constraint_maximum_order_quantity_p0_quantity").set_value(120).run()
    app.checkbox("modular_additions_confirmed").check().run()
    app.session_state["llm"] = InputLLM([])
    app.button("submit_modular_model_inputs").click().run()
    assert not app.exception
    assert app.session_state["step"] == "model_input_form"
    assert app.checkbox(minimum).value
    assert app.checkbox(maximum).value
    assert any("Mindestbestellmenge die Höchstbestellmenge" in error.value for error in app.error)


def test_manual_inputs_advance_questions_and_prompt_for_optional_components_once():
    app = make_app(incomplete=True)
    initial_order = [field.key for field in app.number_input]
    assert "Erwartete Nachfrage" in app.session_state["chat_messages"][-1]["content"]
    app.number_input("modular_forecast_p0_expected_value").set_value(100).run()
    assert not app.exception
    assert [field.key for field in app.number_input] == initial_order
    assert "Nachfragequantile" in app.session_state["chat_messages"][-1]["content"]
    app.text_input("modular_forecast_p0_quantiles").set_value("0.1: 70; 0.5: 100; 0.9: 140").run()
    app.number_input("modular_forecast_p0_horizon_days").set_value(30).run()
    app.number_input("modular_fact_p0_selling_price").set_value(25).run()
    app.number_input("modular_fact_p0_purchase_cost").set_value(10).run()
    assert not app.exception
    assert [field.key for field in app.number_input] == initial_order
    assert app.session_state["step"] == "model_input_form"
    last = app.session_state["chat_messages"][-1]["content"]
    assert "Pflichtfelder sind ausgefüllt" in last
    assert "Weitere Angaben berücksichtigen" in last
    assert "Restwerterlös" in last
    assert "korrekt" not in last and "richtig" not in last
    count = len(app.session_state["chat_messages"])
    app.run()
    assert len(app.session_state["chat_messages"]) == count
    app.checkbox("modular_additions_confirmed").check().run()
    assert "Sie können zur Überprüfung weitergehen" in app.session_state["chat_messages"][-1]["content"]


def test_invalid_manual_correction_stays_open_and_preserves_raw_input():
    app = make_app()
    app.text_input("modular_forecast_p0_quantiles").set_value("keine Zahlen").run()
    assert not app.exception
    spec = app.session_state["graph_state"]["modular_model"]["specification"]
    assert spec["products"][0]["forecast"]["quantiles"] is None
    assert app.text_input("modular_forecast_p0_quantiles").value == "keine Zahlen"
    assert "Nachfragequantile" in app.session_state["chat_messages"][-1]["content"]
    assert app.error
    app.run()
    assert app.text_input("modular_forecast_p0_quantiles").value == "keine Zahlen"
    app.text_input("modular_forecast_p0_quantiles").set_value("0.1: 70; 0.5: 100; 0.9: 140").run()
    assert not app.exception
    assert "Pflichtfelder sind ausgefüllt" in app.session_state["chat_messages"][-1]["content"]
