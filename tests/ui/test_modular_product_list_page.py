from unittest.mock import MagicMock

import pytest

import agentic_optimization.ui.modular_product_list_page as product_list_page
from agentic_optimization.ui.modular_product_list_page import (
    render_dynamic_product_table_page,
)


def test_renders_and_parses_dynamic_product_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()
    streamlit.data_editor.return_value = [
        {
            "product_id": "Artikel-A",
            "expected_value": 100,
            "quantiles": (
                "0.1:70; 0.5:100; 0.9:140"
            ),
            "horizon_days": 30,
            "selling_price": 12,
            "purchase_cost": 5,
        },
        {
            "product_id": "Artikel-B",
            "expected_value": 180,
            "quantiles": (
                "0.1:130; 0.5:180; 0.9:240"
            ),
            "horizon_days": 30,
            "selling_price": 16,
            "purchase_cost": 7,
        },
    ]
    streamlit.button.return_value = True

    monkeypatch.setattr(
        product_list_page,
        "st",
        streamlit,
    )

    proposal = {
        "core_id": "single_period_newsvendor",
        "selection_status": "complete",
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "reason": "Verkaufserlöse berücksichtigen.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "purchase_cost",
                "reason": "Einkaufskosten berücksichtigen.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }

    result = render_dynamic_product_table_page(
        core_id="single_period_newsvendor",
        proposal=proposal,
    )

    assert result is not None
    assert [
        product["product_id"]
        for product in result
    ] == [
        "Artikel-A",
        "Artikel-B",
    ]
    assert result[0]["facts"]["purchase_cost"] == 5.0

    call_kwargs = streamlit.data_editor.call_args.kwargs

    assert call_kwargs["num_rows"] == "dynamic"
    assert call_kwargs["hide_index"] is False
    assert call_kwargs["column_order"] == [
        "display_name",
        "expected_value",
        "quantiles",
        "horizon_days",
        "selling_price",
        "purchase_cost",
    ]


def test_dynamic_product_table_waits_for_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()
    streamlit.data_editor.return_value = []
    streamlit.button.return_value = False

    monkeypatch.setattr(
        product_list_page,
        "st",
        streamlit,
    )

    proposal = {
        "core_id": "single_period_newsvendor",
        "selection_status": "complete",
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "reason": "Verkaufserlöse berücksichtigen.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "purchase_cost",
                "reason": "Einkaufskosten berücksichtigen.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }

    result = render_dynamic_product_table_page(
        core_id="single_period_newsvendor",
        proposal=proposal,
    )

    assert result is None
