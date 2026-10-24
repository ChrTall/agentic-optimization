import json

from langchain_core.messages import AIMessage

from agentic_optimization.agents.business_intake import (
    create_initial_business_intake_state,
)
from agentic_optimization.agents.chat_assistant_agent import (
    create_chat_assistant_agent,
    sanitize_chat_decision,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.ui.chat_assistant_builder import (
    CHAT_SOURCE_ID,
    GREETING,
    append_to_description,
    build_chat_context,
    build_follow_up_messages,
    build_next_question,
    check_product_rows,
    flatten_missing_inputs,
    merge_product_rows,
    resolve_input_submissions,
    resolve_question_answers,
)
from agentic_optimization.ui.business_problem_form import (
    BUSINESS_QUESTIONS,
)
from agentic_optimization.ui.modular_input_builder import (
    build_modular_input_layout,
)


def create_incomplete_specification() -> ModelSpecification:
    return {
        "core_id": "single_period_newsvendor",
        "products": [
            {
                "product_id": "product_1",
                "forecast": {},
                "facts": {},
            }
        ],
        "global_facts": {},
        "objective": {
            "sense": "maximize",
            "component_ids": [
                "sales_revenue",
                "purchase_cost",
            ],
        },
        "constraints": [],
    }


def create_schema():
    return {
        "forecast_fields": [
            {
                "name": "expected_value",
                "label": "Erwartete Nachfrage",
                "type": "number",
                "required": True,
                "help": None,
                "min_value": 0,
            }
        ],
        "fact_fields": [
            {
                "name": "selling_price",
                "label": "Verkaufspreis",
                "type": "number",
                "required": True,
                "help": None,
                "unit": "EUR",
                "min_value": 0,
            }
        ],
    }


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.content)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def test_sanitize_drops_outputs_not_allowed_in_step() -> None:
    decision = sanitize_chat_decision(
        {
            "reply": "Ok",
            "use_message_as_description": True,
            "input_values": [{"input_id": "i1", "value": 3}],
            "action": "start_optimization",
        },
        allowed_outputs=["use_message_as_description"],
        allowed_actions=["none"],
    )

    assert decision["use_message_as_description"] is True
    assert decision["input_values"] == []
    assert decision["action"] == "none"


def test_sanitize_ignores_malformed_items() -> None:
    decision = sanitize_chat_decision(
        {
            "question_answers": [
                {"question_id": "primary_goal"},
                "unsinn",
                {
                    "question_id": "primary_goal",
                    "option_id": "minimize_cost",
                },
            ],
            "products": [
                {"product_id": "  ", "values": {}},
                {"product_id": "A", "values": "x"},
            ],
            "action": "unbekannt",
        },
        allowed_outputs=["question_answers", "products"],
        allowed_actions=["none", "unbekannt"],
    )

    assert decision["question_answers"] == [
        {
            "question_id": "primary_goal",
            "option_id": "minimize_cost",
        }
    ]
    assert decision["products"] == [
        {"product_id": "A", "values": {}}
    ]
    assert decision["action"] == "none"


def test_agent_parses_json_code_fence() -> None:
    llm = FakeLLM(
        "```json\n"
        + json.dumps(
            {
                "reply": "Übernommen.",
                "input_values": [
                    {"input_id": "i2", "value": 12}
                ],
                "action": "none",
            }
        )
        + "\n```"
    )
    agent = create_chat_assistant_agent(llm)

    decision = agent(
        context={
            "step": "model_input_form",
            "allowed_outputs": ["input_values"],
            "allowed_actions": ["none"],
        },
        history=[{"role": "assistant", "content": "Hallo"}],
        user_message="Preis 12 Euro",
    )

    assert decision["reply"] == "Übernommen."
    assert decision["input_values"] == [
        {"input_id": "i2", "value": 12}
    ]
    assert "Preis 12 Euro" in llm.messages[-1].content


def test_agent_falls_back_to_plain_reply_without_json() -> None:
    agent = create_chat_assistant_agent(
        FakeLLM("Ein Servicegrad beschreibt …")
    )

    decision = agent(
        context={
            "allowed_outputs": ["input_values"],
            "allowed_actions": ["none", "restart"],
        },
        history=[],
        user_message="Was ist ein Servicegrad?",
    )

    assert decision["reply"] == "Ein Servicegrad beschreibt …"
    assert decision["input_values"] == []
    assert decision["action"] == "none"


# ---------------------------------------------------------------------------
# Geschäftseinstieg
# ---------------------------------------------------------------------------


def test_resolves_only_open_questions_and_valid_options() -> None:
    questions = [
        BUSINESS_QUESTIONS["primary_goal"],
        BUSINESS_QUESTIONS["product_scope"],
    ]

    answers = resolve_question_answers(
        [
            {
                "question_id": "primary_goal",
                "option_id": "minimize_cost",
            },
            {
                "question_id": "primary_goal",
                "option_id": "maximize_profit",
            },
            {
                "question_id": "product_scope",
                "option_id": "erfunden",
            },
            {
                "question_id": "inventory_review",
                "option_id": "continuous",
            },
        ],
        questions,
    )

    assert answers == [
        {
            "question_id": "primary_goal",
            "option_id": "minimize_cost",
            "source_id": CHAT_SOURCE_ID,
        }
    ]


def test_appends_chat_message_to_description() -> None:
    assert append_to_description("", " Wir bestellen ") == (
        "Wir bestellen"
    )
    assert append_to_description("A", "B") == "A\nB"
    assert append_to_description("A", "   ") == "A"


def test_intake_context_lists_open_questions() -> None:
    graph_state = {
        "business_intake": create_initial_business_intake_state(),
    }

    context = build_chat_context(
        "business_intake",
        graph_state,
        raw_description="",
    )

    question_ids = [
        question["question_id"]
        for question in context["open_questions"]
    ]
    assert "primary_goal" in question_ids
    assert context["allowed_outputs"] == [
        "use_message_as_description",
        "question_answers",
    ]


def test_initial_message_is_greeting_only() -> None:
    graph_state = {
        "business_intake": create_initial_business_intake_state(),
    }

    messages, key = build_follow_up_messages(
        "business_intake",
        graph_state,
        product_rows=[],
        previous_step=None,
        previous_question_key=None,
    )

    assert messages == [GREETING]
    assert key == "business_intake:description"


def test_clarification_asks_one_question_at_a_time() -> None:
    intake_state = create_initial_business_intake_state()
    intake_state["pending_question_ids"] = [
        "primary_goal",
        "product_scope",
    ]
    intake_state["analyzed_raw_input"] = "Text"
    graph_state = {"business_intake": intake_state}

    messages, key = build_follow_up_messages(
        "business_clarification",
        graph_state,
        product_rows=[],
        previous_step="business_intake",
        previous_question_key=None,
    )

    assert len(messages) == 2
    assert BUSINESS_QUESTIONS["primary_goal"]["prompt"] in messages[1]
    assert (
        BUSINESS_QUESTIONS["product_scope"]["prompt"]
        not in messages[1]
    )
    assert "Noch 2 Angaben" in messages[1]

    # Gleiche offene Frage: der Assistent wiederholt sich nicht.
    again, _ = build_follow_up_messages(
        "business_clarification",
        graph_state,
        product_rows=[],
        previous_step="business_clarification",
        previous_question_key=key,
    )
    assert again == []

    # Nach der Antwort folgt genau die nächste Frage.
    intake_state["pending_question_ids"] = ["product_scope"]
    following, _ = build_follow_up_messages(
        "business_clarification",
        graph_state,
        product_rows=[],
        previous_step="business_clarification",
        previous_question_key=key,
    )
    assert len(following) == 1
    assert (
        BUSINESS_QUESTIONS["product_scope"]["prompt"]
        in following[0]
    )


# ---------------------------------------------------------------------------
# Modelleingaben
# ---------------------------------------------------------------------------


def test_resolves_model_inputs_and_reports_invalid_values() -> None:
    specification = create_incomplete_specification()
    missing_inputs = flatten_missing_inputs(
        build_modular_input_layout(specification)
    )
    field_ids = {
        missing_input["field"]["name"]: f"i{index + 1}"
        for index, missing_input in enumerate(missing_inputs)
    }

    submissions, errors = resolve_input_submissions(
        [
            {
                "input_id": field_ids["selling_price"],
                "value": "12,50",
            },
            {
                "input_id": field_ids["purchase_cost"],
                "value": "teuer",
            },
            {"input_id": "i999", "value": 1},
        ],
        missing_inputs,
    )

    assert [
        submission["missing_input"]["field"]["name"]
        for submission in submissions
    ] == ["selling_price"]
    assert submissions[0]["value"] == 12.5
    assert len(errors) == 1


def test_model_input_context_uses_stable_ids() -> None:
    specification = create_incomplete_specification()
    context = build_chat_context(
        "model_input_form",
        {"modular_model": {"specification": specification}},
    )

    ids = [item["input_id"] for item in context["missing_inputs"]]
    assert ids == [f"i{index + 1}" for index in range(len(ids))]
    assert context["allowed_outputs"] == ["input_values"]

    assert context["current_question"]["target"] == {
        "kind": "model_input",
        "input_id": "i1",
    }
    assert "known_parameters" in context

    question = build_next_question(
        "model_input_form",
        {"modular_model": {"specification": specification}},
    )
    assert question["remaining"] == len(flatten_missing_inputs(build_modular_input_layout(specification)))
    # Bei nur einem Produkt wird die Produktkennung nicht genannt.
    assert "product_1" not in question["text"]


# ---------------------------------------------------------------------------
# Produkttabelle
# ---------------------------------------------------------------------------


def test_merges_product_rows_case_insensitive() -> None:
    schema = create_schema()

    rows = merge_product_rows(
        [],
        [
            {
                "product_id": "Apfel",
                "values": {"expected_value": 100},
            }
        ],
        schema,
    )
    rows = merge_product_rows(
        rows,
        [
            {
                "product_id": "apfel",
                "values": {
                    "selling_price": "2,5",
                    "unbekannt": 1,
                },
            },
            {
                "product_id": "Birne",
                "values": {"expected_value": 50},
            },
        ],
        schema,
    )

    assert rows == [
        {
            "product_id": "Apfel",
            "expected_value": 100,
            "selling_price": 2.5,
        },
        {
            "product_id": "Birne",
            "expected_value": 50,
            "selling_price": None,
        },
    ]

    check = check_product_rows(rows, schema)
    assert check["errors"]
    assert "Birne" in check["errors"][0]
