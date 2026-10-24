import json

import pytest

from agentic_optimization.llm.json_response import (
    parse_json_object,
    response_text,
)


def test_parses_plain_json() -> None:
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_parses_code_fence() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_parses_json_after_intro_sentence() -> None:
    content = 'Hier ist die Antwort:\n{"a": 1}\nViele Grüße'

    assert parse_json_object(content) == {"a": 1}


def test_parses_content_blocks_and_skips_thinking() -> None:
    content = [
        {"type": "thinking", "thinking": "..."},
        {"type": "text", "text": '{"a": 1}'},
    ]

    assert parse_json_object(content) == {"a": 1}


def test_rejects_non_object_json() -> None:
    with pytest.raises(TypeError, match="Objekt"):
        parse_json_object("[1, 2]", not_object_message="kein Objekt")


def test_rejects_text_without_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_object("Das ist kein JSON")


def test_rejects_content_without_text() -> None:
    with pytest.raises(TypeError):
        response_text(None)
