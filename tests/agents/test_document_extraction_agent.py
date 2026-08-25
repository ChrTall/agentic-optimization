import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agentic_optimization.agents.document_extraction_agent import (
    extract_document_parameters,
    format_parameter_display,
)


def create_mock_llm(response: object) -> Mock:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(response),
    )
    return llm


def test_extracts_parameters_from_document_text() -> None:
    llm = create_mock_llm([
        {
            "label": "Stückpreis",
            "value": "4.50",
            "unit": "€",
            "evidence": "Der Preis pro Einheit beträgt 4,50 €.",
        }
    ])

    result = extract_document_parameters(
        llm, "Der Preis pro Einheit beträgt 4,50 €."
    )

    assert len(result) == 1
    assert result[0]["label"] == "Stückpreis"
    assert result[0]["value"] == "4.50"
    assert result[0]["unit"] == "€"


def test_returns_empty_list_for_empty_text() -> None:
    llm = create_mock_llm([])

    result = extract_document_parameters(llm, "   ")

    assert result == []
    llm.invoke.assert_not_called()


def test_returns_empty_list_on_invalid_llm_response() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="kein json")

    result = extract_document_parameters(llm, "irgendein Text")

    assert result == []


def test_format_parameter_display_includes_unit() -> None:
    parameter = {
        "label": "Lieferzeit",
        "value": "14",
        "unit": "Tage",
        "evidence": None,
    }

    assert format_parameter_display(parameter) == "Lieferzeit: 14 Tage"