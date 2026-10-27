import string
from unittest.mock import Mock

import pytest

from agentic_optimization.i18n import get_language, set_language, t
from agentic_optimization.i18n.catalog import collect_translation_keys
from agentic_optimization.i18n.en import EN


@pytest.fixture(autouse=True)
def reset_language():
    set_language("de")
    yield
    set_language("de")


def _placeholders(text: str) -> set[str]:
    return {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(text)
        if field_name is not None
    }


def test_every_text_has_an_english_translation() -> None:
    missing = sorted(collect_translation_keys() - set(EN))

    assert missing == [], f"Fehlende Übersetzungen: {missing[:10]}"


def test_translations_keep_the_same_placeholders() -> None:
    mismatched = [
        key
        for key, value in EN.items()
        if _placeholders(key) != _placeholders(value)
    ]

    assert mismatched == []


def test_translate_switches_language_and_fills_placeholders() -> None:
    template = "Schritt {number} von {TOTAL_STEPS} · {label}"

    assert t(template, number=2, TOTAL_STEPS=5, label="X") == (
        "Schritt 2 von 5 · X"
    )

    set_language("en")
    assert get_language() == "en"
    assert t(template, number=2, TOTAL_STEPS=5, label="X") == "Step 2 of 5 · X"
    assert t("Text ohne Übersetzung") == "Text ohne Übersetzung"
    assert t(None) is None


def test_unknown_language_falls_back_to_german() -> None:
    set_language("fr")

    assert get_language() == "de"


def test_language_reaches_workflow_nodes() -> None:
    from agentic_optimization.graph import build_graph

    set_language("en")
    graph = build_graph(Mock())

    with pytest.raises(ValueError, match="The business intake state is missing"):
        graph.invoke({"control": {"event": "business_intake_requested"}})
