from unittest.mock import MagicMock

import pytest

from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
)
from agentic_optimization.ui import (
    business_intake_page,
)
from agentic_optimization.ui.business_intake_page import (
    format_business_question_option,
    render_business_intake_page,
)


def create_intake_state():
    return {
        "profile": create_empty_business_problem_profile(),
        "pending_question_ids": [
            "recommendation",
            "primary_goal",
        ],
    }


def test_formats_option_with_help_text() -> None:
    assert format_business_question_option(
        {
            "option_id": "order_quantity",
            "profile_value": "order_quantity",
            "label": "Bestellmenge bestimmen",
            "help": "Wie viele Einheiten werden benötigt?",
        }
    ) == (
        "Bestellmenge bestimmen — "
        "Wie viele Einheiten werden benötigt?"
    )


def test_returns_none_before_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()
    streamlit.text_area.return_value = ""
    streamlit.radio.return_value = None
    streamlit.form_submit_button.return_value = False

    monkeypatch.setattr(
        business_intake_page,
        "st",
        streamlit,
    )

    result = render_business_intake_page(
        create_intake_state()
    )

    assert result is None


def test_returns_description_and_selected_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()
    streamlit.text_area.return_value = (
        "Das Produkt wird regelmäßig nachbestellt."
    )
    streamlit.radio.side_effect = [
        "order_quantity",
        "minimize_cost",
    ]
    streamlit.form_submit_button.return_value = True

    monkeypatch.setattr(
        business_intake_page,
        "st",
        streamlit,
    )

    result = render_business_intake_page(
        create_intake_state(),
        raw_input_value="Das Produkt wird regelmäßig nachbestellt.",
    )

    # Die Beschreibung kommt aus dem Chat, nicht aus einem Textfeld.
    streamlit.text_area.assert_not_called()
    assert result == {
        "raw_input": (
            "Das Produkt wird regelmäßig nachbestellt."
        ),
        "answers": [
            {
                "question_id": "recommendation",
                "option_id": "order_quantity",
                "source_id": "business_intake_form",
            },
            {
                "question_id": "primary_goal",
                "option_id": "minimize_cost",
                "source_id": "business_intake_form",
            },
        ],
    }


def test_has_no_description_field_or_document_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()
    streamlit.radio.return_value = None
    streamlit.form_submit_button.return_value = False

    monkeypatch.setattr(
        business_intake_page,
        "st",
        streamlit,
    )

    render_business_intake_page(
        create_intake_state()
    )

    streamlit.text_area.assert_not_called()
    streamlit.file_uploader.assert_not_called()


def test_can_hide_questions_when_editing_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()
    streamlit.text_area.return_value = (
        "Bearbeitete Beschreibung"
    )
    streamlit.form_submit_button.return_value = False

    monkeypatch.setattr(
        business_intake_page,
        "st",
        streamlit,
    )

    result = render_business_intake_page(
        create_intake_state(),
        raw_input_value="Bisherige Beschreibung",
        show_questions=False,
    )

    assert result is None
    streamlit.radio.assert_not_called()
