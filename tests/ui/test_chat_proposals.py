import pytest

from agentic_optimization.ui.business_problem_form import BUSINESS_QUESTIONS
from agentic_optimization.ui.chat_proposals import match_current_option


@pytest.mark.parametrize("question_id, option_id", [
    (qid, option["option_id"])
    for qid, question in BUSINESS_QUESTIONS.items()
    for option in question["options"]
])
def test_exact_registered_options_are_explicit_answers(question_id, option_id):
    question = BUSINESS_QUESTIONS[question_id]
    option = next(o for o in question["options"] if o["option_id"] == option_id)
    result = match_current_option("  " + option["label"].lower() + "!  ", question)
    assert result["option_id"] == option_id
    assert result["source_id"] == "chat_assistant_verified"


def test_embedded_selection_with_additional_information_is_not_an_exact_choice():
    assert match_current_option("Bestellmenge bestimmen, aber vielleicht auch den Meldebestand.",
                                BUSINESS_QUESTIONS["recommendation"]) is None
    assert match_current_option("Bestellmenge bestimmen", None) is None
