from unittest.mock import MagicMock

import pytest

from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
)
from agentic_optimization.ui import (
    business_clarification_page,
)
from agentic_optimization.ui.business_clarification_page import (
    render_business_clarification_page,
)


def create_intake_state():
    return {
        "profile": create_empty_business_problem_profile(),
        "pending_question_ids": [
            "replenishment_during_period",
        ],
    }


def test_renders_only_open_clarification_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()
    streamlit.radio.return_value = "not_possible"
    streamlit.form_submit_button.return_value = True

    monkeypatch.setattr(
        business_clarification_page,
        "st",
        streamlit,
    )

    result = render_business_clarification_page(
        create_intake_state()
    )

    assert result == [
        {
            "question_id": (
                "replenishment_during_period"
            ),
            "option_id": "not_possible",
            "source_id": (
                "business_clarification_form"
            ),
        }
    ]

    streamlit.text_area.assert_not_called()
    streamlit.file_uploader.assert_not_called()


def test_returns_none_before_clarification_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()
    streamlit.radio.return_value = None
    streamlit.form_submit_button.return_value = False

    monkeypatch.setattr(
        business_clarification_page,
        "st",
        streamlit,
    )

    result = render_business_clarification_page(
        create_intake_state()
    )

    assert result is None


def test_does_not_render_form_without_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()

    monkeypatch.setattr(
        business_clarification_page,
        "st",
        streamlit,
    )

    result = render_business_clarification_page(
        {
            "profile": (
                create_empty_business_problem_profile()
            ),
            "pending_question_ids": [],
        }
    )

    assert result is None
    streamlit.form.assert_not_called()
    streamlit.info.assert_called_once_with(
        "Derzeit sind keine konkreten Rückfragen offen."
    )