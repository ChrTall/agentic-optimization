import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agentic_optimization.agents.chat_assistant_agent import (
    recall_from_statements,
)
from agentic_optimization.ui.chat_assistant_builder import (
    collect_user_statements,
    format_recall_note,
)

STATEMENTS = (
    "Ein Baum wird für 25 € verkauft. "
    "Während der Saison können wir nicht nachbestellen."
)
ITEMS = [
    {
        "id": "i1",
        "kind": "value",
        "question": "Verkaufspreis",
        "field": {"name": "selling_price", "unit": "€"},
    },
    {
        "id": "replenishment_during_period",
        "kind": "choice",
        "question": "Kann während der Saison nachbestellt werden?",
        "options": [
            {"option_id": "no", "label": "Nein"},
            {"option_id": "yes", "label": "Ja"},
        ],
    },
]


def _llm(answers: list[dict]) -> Mock:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps({"answers": answers})
    )
    return llm


def test_recalls_values_with_verbatim_quotes() -> None:
    llm = _llm([
        {"id": "i1", "value": 25, "quote": "für 25 € verkauft"},
        {
            "id": "replenishment_during_period",
            "value": "no",
            "quote": "während der Saison können wir nicht nachbestellen",
        },
    ])

    recalled = recall_from_statements(
        llm, statements=STATEMENTS, open_items=ITEMS
    )

    assert [answer["id"] for answer in recalled] == [
        "i1",
        "replenishment_during_period",
    ]
    assert recalled[1]["quote"] == (
        "Während der Saison können wir nicht nachbestellen"
    )


def test_rejects_invented_quotes_unknown_ids_and_options() -> None:
    llm = _llm([
        {"id": "i1", "value": 30, "quote": "für 30 € verkauft"},
        {"id": "i9", "value": 1, "quote": "Ein Baum"},
        {
            "id": "replenishment_during_period",
            "value": "vielleicht",
            "quote": "nicht nachbestellen",
        },
    ])

    assert recall_from_statements(
        llm, statements=STATEMENTS, open_items=ITEMS
    ) == []


def test_skips_llm_call_without_statements_or_items() -> None:
    llm = _llm([])

    assert recall_from_statements(llm, statements="", open_items=ITEMS) == []
    assert recall_from_statements(llm, statements="x", open_items=[]) == []
    llm.invoke.assert_not_called()


def test_collects_only_user_statements_and_keeps_newest() -> None:
    messages = [
        {"role": "assistant", "content": "Hallo"},
        {"role": "user", "content": "Erste Aussage"},
        {"role": "user", "content": "Zweite Aussage"},
    ]

    assert collect_user_statements(messages) == (
        "Erste Aussage\n\nZweite Aussage"
    )
    assert collect_user_statements(messages, limit=10).endswith("Aussage")
    assert len(collect_user_statements(messages, limit=10)) == 10


def test_recall_note_lists_values_with_quotes() -> None:
    note = format_recall_note(
        [{**item, "key": item["id"]} for item in ITEMS],
        [
            {"id": "i1", "value": 25, "quote": "für 25 € verkauft"},
            {
                "id": "replenishment_during_period",
                "value": "no",
                "quote": "nicht nachbestellen",
            },
        ],
    )

    assert "**Verkaufspreis**: 25 €" in note
    assert "Nein" in note
    assert "„für 25 € verkauft“" in note
