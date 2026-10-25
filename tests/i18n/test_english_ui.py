"""Die Oberfläche auf Englisch: kein übrig gebliebenes Deutsch."""

import re
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agentic_optimization.i18n import set_language

sys.path.insert(0, str(Path(__file__).parents[1] / "workflow"))
import test_chat_assistant_app_flow as flow  # noqa: E402

GERMAN = re.compile(
    r"[äöüÄÖÜß]|\b(der|die|das|und|nicht|Sie|Ihre|Bitte|mit|für|eine?|wird|"
    r"ist|noch|zur|Angaben|Produkt|Weiter|Neu)\b"
)
# Nutzereingaben, feste Antworten des Test-Sprachmodells und daraus
# übernommene Zitate/Namen sind bewusst deutsch.
ALLOWED = (
    "Gartenschlauch", "Artikel", "wie viel", "regelmäßig",
    "Danke, ich schaue mir das an.", "Werte übernommen.",
    "Verstanden – Kosten senken.", "Ja, wir bestellen regelmäßig nach.",
    "Ich starte die Berechnung.", "Gern erkläre ich das.",
    "Dann geht es weiter zur Überprüfung.", "Erklärungstext.",
    "Bestellkosten", "Lagerkosten",
)


def _visible_texts(app: AppTest) -> list[str]:
    texts = []
    for kind in ("markdown", "caption", "subheader", "info", "success",
                 "warning", "error"):
        texts += [str(element.value) for element in getattr(app, kind)]
    for kind in ("button", "checkbox", "radio", "selectbox", "number_input",
                 "text_input", "expander"):
        for element in getattr(app, kind):
            texts.append(str(getattr(element, "label", "") or ""))
    texts += [str(metric.label) for metric in app.metric]
    texts += [
        message["content"]
        for message in app.session_state["chat_messages"]
        if message["role"] == "assistant"
    ]
    return texts


def _german_leftovers(app: AppTest, user_messages: list[str]) -> list[str]:
    leftovers = []
    for text in _visible_texts(app):
        if text.strip() in user_messages:
            continue
        clean = text
        for allowed in ALLOWED:
            clean = clean.replace(allowed, "")
        if GERMAN.search(clean):
            leftovers.append(text)
    return leftovers


@pytest.fixture(autouse=True)
def reset_language():
    yield
    set_language("de")


def test_complete_flow_in_english_has_no_german_text() -> None:
    app = AppTest.from_file(str(flow.APP_PATH), default_timeout=60)
    app.session_state["llm"] = flow.FakeLLM()
    app.session_state["ui_language"] = "en"
    app.run()

    user_messages = [
        "Wir bestellen einen Artikel regelmäßig nach und wollen wissen, "
        "wie viel wir jeweils bestellen sollen.",
        "Vor allem sollen die Kosten niedrig bleiben.",
        "Ja, wir bestellen regelmäßig nach.",
        "3500 Stück im Jahr, 120 € je Bestellung, 12 € Lager.",
        "Passt, weiter.",
        "Bitte berechnen.",
    ]
    leftovers: list[str] = []

    leftovers += _german_leftovers(app, user_messages)
    flow._say(app, user_messages[0])
    leftovers += _german_leftovers(app, user_messages)
    flow._say(app, user_messages[1])
    if app.session_state["step"] == "business_clarification":
        flow._say(app, user_messages[2])
    leftovers += _german_leftovers(app, user_messages)
    flow._say(app, user_messages[3])
    leftovers += _german_leftovers(app, user_messages)
    flow._say(app, user_messages[4])
    leftovers += _german_leftovers(app, user_messages)
    for checkbox in app.checkbox:
        if not checkbox.value:
            checkbox.check().run()
    flow._say(app, user_messages[5])
    leftovers += _german_leftovers(app, user_messages)

    assert app.session_state["step"] == "result"
    assert sorted(set(leftovers)) == []
    assert any(
        metric.label == "Annual ordering and holding cost"
        for metric in app.metric
    )


def test_other_messages_and_results_render_in_english() -> None:
    from agentic_optimization.models.execution import (
        execute_model_specification,
    )
    from agentic_optimization.models.problem_profile import (
        create_empty_business_problem_profile,
    )
    from agentic_optimization.ui.chat_assistant_builder import (
        build_next_question,
        build_step_intro,
        build_stuck_message,
    )
    from agentic_optimization.ui.modular_result_builder import (
        build_modular_result_data,
    )

    set_language("en")

    unsupported = build_next_question("business_intake", {
        "business_intake": {
            "profile": create_empty_business_problem_profile(),
            "analyzed_raw_input": "x",
            "pending_question_ids": [],
            "core_selection": {
                "status": "unsupported",
                "selected_core_id": None,
                "candidate_core_ids": [],
                "reason": "",
            },
        }
    })
    texts = [unsupported["text"]]
    texts += [
        build_stuck_message({
            "key": "k", "text": "", "target": {"kind": kind}, "remaining": 1,
        })
        for kind in ("business_question", "model_input", "description",
                     "product_table")
    ]
    texts += [
        build_step_intro(step, {})
        for step in ("business_intake", "business_clarification",
                     "product_table", "model_input_form", "review", "result")
    ]

    newsvendor = {
        "core_id": "single_period_newsvendor",
        "products": [{
            "product_id": "Tree",
            "forecast": {
                "expected_value": 276.0,
                "horizon_days": 55,
                "quantiles": [
                    {"level": 0.1, "value": 150.0},
                    {"level": 0.5, "value": 300.0},
                    {"level": 0.9, "value": 350.0},
                ],
            },
            "facts": {"selling_price": 25.0, "purchase_cost": 10.0},
        }],
        "global_facts": {},
        "objective": {
            "sense": "maximize",
            "component_ids": ["sales_revenue", "purchase_cost"],
        },
        "constraints": [],
    }
    result = build_modular_result_data(
        newsvendor, execute_model_specification(newsvendor)
    )
    assert result["objective_label"] == "Expected profit"
    texts += [result["core_name"], result["status_label"]]
    texts += [value["label"] for value in result["metrics"]]
    texts += [value["label"] for value in result["decisions"]]

    for text in texts:
        assert text and not GERMAN.search(text), text


def test_switching_language_translates_the_assistant_texts() -> None:
    app = AppTest.from_file(str(flow.APP_PATH), default_timeout=60)
    app.session_state["llm"] = flow.FakeLLM()
    app.run()
    assert app.session_state["chat_messages"][0]["content"].startswith("Hallo")

    app.segmented_control(key="ui_language").set_value("en").run()
    messages = app.session_state["chat_messages"]
    assert messages[0]["content"].startswith("Hello")
    assert all(not GERMAN.search(m["content"]) for m in messages)

    flow._say(
        app,
        "Wir bestellen einen Artikel regelmäßig nach und wollen wissen, "
        "wie viel wir jeweils bestellen sollen.",
    )
    app.segmented_control(key="ui_language").set_value("de").run()
    messages = app.session_state["chat_messages"]
    assert messages[0]["content"].startswith("Hello")  # Verlauf bleibt
    assert messages[-1]["role"] == "assistant"
    assert "?" in messages[-1]["content"]
    assert not messages[-1]["content"].startswith(("What", "Which", "How"))
