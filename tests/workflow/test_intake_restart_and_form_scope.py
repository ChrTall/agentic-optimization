from pathlib import Path

from streamlit.testing.v1 import AppTest

from agentic_optimization.agents.business_intake import create_initial_business_intake_state
from agentic_optimization.models.problem_profile import create_profile_field
from agentic_optimization.state import WORKFLOW_BUSINESS_CASE_UNSUPPORTED, with_control


APP_PATH = Path(__file__).parents[2] / "app.py"


def known(value):
    return create_profile_field(status="confirmed", value=value,
                               evidence=[{"source_id": "form", "source_type": "user_selection"}])


def test_unsupported_intake_has_complete_restart_without_needing_llm():
    intake = create_initial_business_intake_state()
    intake["core_selection"] = {"status": "unsupported", "reason": "Keine passende Struktur", "selected_core_id": None}
    state = with_control({"business_intake": intake, "raw_input": "Bisherige Beschreibung"},
                         event=None, workflow_status=WORKFLOW_BUSINESS_CASE_UNSUPPORTED)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["graph_state"] = state
    app.session_state["step"] = "business_intake"
    app.session_state["raw_problem_description"] = state["raw_input"]
    app.run()
    assert not app.exception
    assert any("nicht zuverlässig abgebildet" in error.value for error in app.error)
    app.button("restart_business_intake").click().run()
    assert not app.exception
    assert app.session_state["raw_problem_description"] == ""
    assert "raw_input" not in app.session_state["graph_state"]
    assert len(app.session_state["chat_messages"]) == 1


def test_numeric_form_does_not_show_business_answer_summary():
    intake = create_initial_business_intake_state()
    intake["profile"]["primary_goal"] = known("minimize_cost")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["step"] = "model_input_form"
    app.session_state["graph_state"] = {"business_intake": intake, "modular_model": {
        "specification": {"core_id": "eoq", "products": [{
            "product_id": "product_1", "forecast": {}, "facts": {},
        }], "global_facts": {}, "objective": {"sense": "minimize", "component_ids": [
            "annual_ordering_cost", "annual_holding_cost"]}, "constraints": []},
    }}
    app.run()
    assert not app.exception
    assert not any("Bereits geklärt" in expander.label for expander in app.expander)
    assert not any(button.key == "edit_business_primary_goal" for button in app.button)
    assert app.number_input("modular_forecast_product_1_expected_value")
