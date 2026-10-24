from unittest.mock import Mock

import pytest

import agentic_optimization.ui.modular_review_page as modular_review_page

from agentic_optimization.ui.modular_review_page import (
    _render_core_assumption_confirmations,
    _build_constraint_parameter_labels,
    _build_product_overview_rows,
    _build_constraint_overview_rows,
    render_modular_exclusion_confirmation,
    render_modular_selection_issues,
)
from agentic_optimization.models.core_applicability import CONDITION_LABELS


def test_assumption_evidence_does_not_confirm_checkbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = Mock()
    streamlit.checkbox.return_value = False
    monkeypatch.setattr(modular_review_page, "st", streamlit)

    confirmed = _render_core_assumption_confirmations(
        ["steady_demand"],
        [{
            "condition_id": "steady_demand",
            "status": "supported",
            "source_id": "raw_input",
            "excerpt": "Der Bedarf ist über das Jahr ungefähr gleichmäßig.",
        }],
    )

    assert confirmed is False
    streamlit.markdown.assert_any_call("#### Aus Ihrer Beschreibung abgeleitet")
    streamlit.write.assert_any_call(
        "Ihre Angabe: „Der Bedarf ist über das Jahr ungefähr gleichmäßig.“"
    )
    streamlit.checkbox.assert_called_once_with(
        CONDITION_LABELS["steady_demand"],
        key="confirm_core_assumption_steady_demand",
    )
    assert "Ein Jahresbedarf allein" not in CONDITION_LABELS[
        "steady_demand"
    ]


def test_contrary_text_evidence_is_warned_and_still_needs_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = Mock()
    streamlit.checkbox.return_value = False
    monkeypatch.setattr(modular_review_page, "st", streamlit)

    confirmed = _render_core_assumption_confirmations(
        ["no_shortages"],
        [{
            "condition_id": "no_shortages",
            "status": "contradicted",
            "source_id": "raw_input",
            "excerpt": "Wir planen Rückstände.",
        }],
    )

    assert confirmed is False
    assert "Möglicher Widerspruch" in streamlit.warning.call_args_list[0].args[0]
    streamlit.write.assert_any_call("Ihre Angabe: „Wir planen Rückstände.“")
    streamlit.checkbox.assert_called_once_with(
        CONDITION_LABELS["no_shortages"],
        key="confirm_core_assumption_no_shortages",
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


def test_builds_compact_multi_product_overview() -> None:
    products = [
        {
            "product_id": "product_1",
            "forecast": {
                "expected_value": 100,
                "quantiles": [
                    {
                        "level": 0.1,
                        "value": 70,
                    },
                    {
                        "level": 0.9,
                        "value": 140,
                    },
                ],
                "horizon_days": 30,
            },
            "facts": {
                "selling_price": 12,
                "purchase_cost": 5,
            },
        },
        {
            "product_id": "product_2",
            "forecast": {
                "expected_value": 200,
                "quantiles": [
                    {
                        "level": 0.1,
                        "value": 150,
                    },
                    {
                        "level": 0.9,
                        "value": 260,
                    },
                ],
                "horizon_days": 30,
            },
            "facts": {
                "selling_price": 18,
                "purchase_cost": 8,
            },
        },
    ]

    rows = _build_product_overview_rows(
        products,
        {
            "expected_value": "Erwartete Nachfrage",
            "quantiles": "Nachfragequantile",
            "horizon_days": "Prognosezeitraum",
            "selling_price": "Verkaufspreis",
            "purchase_cost": "Einkaufskosten",
        },
    )

    assert rows == [
        {
            "Produkt": "Produkt 1",
            "Erwartete Nachfrage": "100",
            "Nachfragequantile": "Q10: 70, Q90: 140",
            "Prognosezeitraum": "30",
            "Verkaufspreis": "12",
            "Einkaufskosten": "5",
        },
        {
            "Produkt": "Produkt 2",
            "Erwartete Nachfrage": "200",
            "Nachfragequantile": "Q10: 150, Q90: 260",
            "Prognosezeitraum": "30",
            "Verkaufspreis": "18",
            "Einkaufskosten": "8",
        },
    ]


def test_builds_multi_product_constraint_overview() -> None:
    constraints = [
        {
            "component_id": "minimum_order_quantity",
            "name": "Mindestbestellmenge",
            "scope": "per_product",
            "product_id": "product_1",
            "parameters": {
                "quantity": 50,
            },
            "reason": "Mindestmenge wurde genannt.",
            "sources": ["raw_input"],
        },
        {
            "component_id": "minimum_order_quantity",
            "name": "Mindestbestellmenge",
            "scope": "per_product",
            "product_id": "product_2",
            "parameters": {
                "quantity": 80,
            },
            "reason": "Mindestmenge wurde genannt.",
            "sources": ["raw_input"],
        },
    ]

    rows = _build_constraint_overview_rows(
        constraints,
        {
            "minimum_order_quantity": {
                "quantity": "Mindestbestellmenge",
            }
        },
    )

    assert rows == [
        {
            "Nebenbedingung": "Mindestbestellmenge",
            "Produkt": "Produkt 1",
            "Mindestbestellmenge": "50",
        },
        {
            "Nebenbedingung": "Mindestbestellmenge",
            "Produkt": "Produkt 2",
            "Mindestbestellmenge": "80",
        },
    ]


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


def test_requires_confirmation_before_excluding_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = Mock()
    streamlit.checkbox.return_value = False
    streamlit.button.return_value = True

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
                "sources": ["raw_input"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten wurden genannt.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [
            {
                "description": "Fester Liefertag",
                "resolution": "unsupported",
                "candidate_component_ids": [],
                "reason": "Kein kompatibler Baustein.",
                "sources": ["raw_input"],
            }
        ],
    }

    result = render_modular_exclusion_confirmation(
        proposal
    )

    assert result is False
    streamlit.button.assert_called_once_with(
        "Ohne diese Anforderungen fortfahren",
        width="stretch",
        disabled=True,
        key=(
            "continue_without_unsupported_requirements"
        ),
    )


def test_accepts_confirmed_requirement_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = Mock()
    streamlit.checkbox.return_value = True
    streamlit.button.return_value = True

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
                "sources": ["raw_input"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten wurden genannt.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [
            {
                "description": "Fester Liefertag",
                "resolution": "unsupported",
                "candidate_component_ids": [],
                "reason": "Kein kompatibler Baustein.",
                "sources": ["raw_input"],
            }
        ],
    }

    result = render_modular_exclusion_confirmation(
        proposal
    )

    assert result is True
