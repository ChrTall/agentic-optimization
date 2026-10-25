import json
from pathlib import Path

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from agentic_optimization.agents.modular_workflow_nodes import create_modular_model_preparation_node
from agentic_optimization.state import get_workflow_status, WORKFLOW_MODEL_PREPARATION_FAILED

APP_PATH = Path(__file__).parents[2] / "app.py"


class RetryLLM:
    def __init__(self):
        self.fail_selection = True
        self.facts_calls = 0
        self.selection_calls = 0

    def invoke(self, messages):
        system = messages[0].content
        if system.startswith("Du extrahierst"):
            self.facts_calls += 1
            return AIMessage(content=json.dumps({"products": [{
                "product_id": "product_1", "forecast": {"horizon_days": 365},
                "facts": {"holding_cost": 12},
            }]}))
        if system.startswith("Du wählst"):
            self.selection_calls += 1
            if self.fail_selection:
                raise ValueError("Malformed response with SECRET user input")
            return AIMessage(content=json.dumps({
                "core_id": "eoq", "selection_status": "complete", "objective_sense": "minimize",
                "objective_components": [{"component_id": cid, "reason": "Kosten",
                                          "sources": ["raw_input"]}
                                         for cid in ["annual_ordering_cost", "annual_holding_cost"]],
                "constraints": [], "unresolved_requirements": [],
            }))
        if system.startswith("Prüfe einen"):
            return AIMessage(content='{"rejected_constraints": [], "input_only_requirements": []}')
        raise AssertionError("Unerwarteter Agent im Wiederholungsversuch")


def failed_app():
    llm = RetryLLM()
    state = {
        "raw_input": "Wir bestellen diesen Artikel regelmäßig nach und wollen die Kosten minimieren.",
        "requirements": {"objective": "minimize_cost"},
        "modular_model": {"core_id": "eoq", "products": [{
            "product_id": "product_1", "display_name": "Schrauben",
            "forecast": {"expected_value": 3500}, "facts": {"ordering_cost": 120},
        }], "global_facts": {}, "evidence": [{
            "source_id": "raw_input", "content": "Die jährlichen Bestell- und Lagerkosten minimieren."}]},
    }
    state = create_modular_model_preparation_node(llm)(state)
    assert get_workflow_status(state) == WORKFLOW_MODEL_PREPARATION_FAILED
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["llm"] = llm
    app.session_state["step"] = "preparation_failed"
    app.session_state["graph_state"] = state
    app.session_state["raw_problem_description"] = state["raw_input"]
    app.run()
    assert not app.exception
    return app, llm


def test_error_page_retries_from_checkpoint_and_retains_all_values():
    app, llm = failed_app()
    assert app.button("retry_preparation")
    assert app.button("edit_failed_description")
    assert app.button("restart_failed_preparation")
    assert any(expander.label == "Technische Details" for expander in app.expander)
    displayed = " ".join(str(item.value) for item in [*app.text, *app.markdown, *app.error])
    assert "SECRET" not in displayed
    assert "technisch unterbrochen" in app.session_state["chat_messages"][-1]["content"]
    reference = app.session_state["graph_state"]["modular_model"]["preparation_error_details"]["reference"]
    app.button("retry_preparation").click().run()
    assert not app.exception
    assert app.session_state["step"] == "preparation_failed"
    assert app.session_state["graph_state"]["modular_model"]["preparation_error_details"]["reference"] != reference
    assert llm.facts_calls == 1
    llm.fail_selection = False
    app.button("retry_preparation").click().run()
    assert not app.exception
    assert app.session_state["step"] == "model_input_form"
    assert llm.facts_calls == 1
    assert app.number_input("modular_forecast_product_1_expected_value").value == 3500
    assert app.number_input("modular_forecast_product_1_horizon_days").value == 365
    assert app.number_input("modular_fact_product_1_ordering_cost").value == 120
    assert app.number_input("modular_fact_product_1_holding_cost").value == 12
    assert app.text_input("modular_product_name_product_1").value == "Schrauben"
    assert app.session_state["graph_state"]["modular_model"]["preparation_checkpoint"] is None


def test_edit_keeps_description_and_entered_products():
    app, _ = failed_app()
    original = app.session_state["raw_problem_description"]
    app.button("edit_failed_description").click().run()
    assert not app.exception
    assert app.session_state["step"] == "business_intake"
    assert not app.text_area
    assert app.session_state["raw_problem_description"] == original
    assert app.session_state["graph_state"]["modular_model"]["products"][0]["facts"]["ordering_cost"] == 120


def test_complete_restart_removes_case_answers_chat_and_drafts():
    app, _ = failed_app()
    app.session_state["modular_input_drafts"] = {"old": 42}
    app.session_state["business_clarification_primary_goal"] = "minimize_cost"
    app.session_state["chat_product_rows"] = [{"product_id": "old"}]
    app.button("restart_failed_preparation").click().run()
    assert not app.exception
    assert app.session_state["step"] == "business_intake"
    assert app.session_state["raw_problem_description"] == ""
    assert "modular_model" not in app.session_state["graph_state"]
    assert "raw_input" not in app.session_state["graph_state"]
    assert "modular_input_drafts" not in app.session_state
    assert "business_clarification_primary_goal" not in app.session_state
    assert app.session_state["chat_product_rows"] == []
    assert len(app.session_state["chat_messages"]) == 1
