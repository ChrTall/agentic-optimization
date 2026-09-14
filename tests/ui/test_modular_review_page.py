from unittest.mock import Mock

import pytest

from agentic_optimization.ui import (
    modular_review_page,
)
from agentic_optimization.ui.modular_review_page import (
    _build_constraint_parameter_labels,
    render_modular_selection_issues,
)

def test_uses_component_specific_parameter_labels() -> None:
    labels = _build_constraint_parameter_labels(
        "single_period_newsvendor"
    )

    assert labels["minimum_order_quantity"]["quantity"] == (
        "Mindestbestellmenge"
    )
    assert labels["maximum_order_quantity"]["quantity"] == (
        "Höchstbestellmenge"
    )
    assert labels[
        "minimum_cycle_service_level"
    ]["service_level_percent"] == (
        "Mindestwahrscheinlichkeit ohne Fehlbestand"
    )


def test_renders_unsupported_selection_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = Mock()

    monkeypatch.setattr(
        modular_review_page,
        "st",
        streamlit,
    )

    proposal = {
        "core_id": "eoq",
        "selection_status": "unsupported",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten wurden genannt.",
                "sources": ["source_1"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten wurden genannt.",
                "sources": ["source_1"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [
            {
                "description": "Mengenrabatt",
                "resolution": "unsupported",
                "candidate_component_ids": [],
                "reason": (
                    "Für Mengenrabatte ist kein "
                    "kompatibler Baustein verfügbar."
                ),
                "sources": ["source_1"],
            }
        ],
    }

    render_modular_selection_issues(proposal)

    streamlit.error.assert_called_once()
    streamlit.markdown.assert_called_once_with(
        "**Mengenrabatt**"
    )
    streamlit.write.assert_called_once_with(
        "Für Mengenrabatte ist kein "
        "kompatibler Baustein verfügbar."
    )
    streamlit.caption.assert_any_call(
        "Erkanntes Grundmodell: "
        "Wirtschaftliche Bestellmenge (EOQ)"
    )
    streamlit.caption.assert_any_call(
        "Quelle: source_1"
    )
