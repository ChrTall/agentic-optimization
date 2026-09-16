from unittest.mock import MagicMock

import pytest

import agentic_optimization.ui.modular_input_page as modular_input_page
from agentic_optimization.ui.modular_form_parser import (
    parse_modular_input_submissions,
)
from agentic_optimization.ui.modular_input_builder import (
    build_modular_input_layout,
)
from agentic_optimization.ui.modular_input_page import (
    _render_product_input_table,
)


def create_multi_product_specification() -> dict:
    return {
        "core_id": "single_period_newsvendor",
        "products": [
            {
                "product_id": f"product_{index}",
                "forecast": {},
                "facts": {},
            }
            for index in range(4)
        ],
        "global_facts": {},
        "objective": {
            "sense": "maximize",
            "component_ids": [
                "sales_revenue",
                "purchase_cost",
            ],
        },
        "constraints": [],
    }


def test_maps_edited_table_values_to_product_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()
    streamlit.data_editor.return_value = [
        {
            "product_id": f"product_{index}",
            "expected_value": 100 + index,
            "quantiles": (
                "0.1: 70; 0.5: 100; 0.9: 140"
            ),
            "horizon_days": 30,
            "selling_price": 12 + index,
            "purchase_cost": 5 + index,
        }
        for index in range(4)
    ]

    monkeypatch.setattr(
        modular_input_page,
        "st",
        streamlit,
    )

    layout = build_modular_input_layout(
        create_multi_product_specification()
    )

    rendered_inputs = _render_product_input_table(
        layout
    )
    submissions = parse_modular_input_submissions(
        rendered_inputs
    )

    assert len(submissions) == 20

    submitted_values = {
        (
            submission["missing_input"]["product_id"],
            submission["missing_input"]["field"]["name"],
        ): submission["value"]
        for submission in submissions
    }

    assert submitted_values[
        (
            "product_0",
            "expected_value",
        )
    ] == 100.0
    assert submitted_values[
        (
            "product_3",
            "selling_price",
        )
    ] == 15.0
    assert submitted_values[
        (
            "product_2",
            "purchase_cost",
        )
    ] == 7.0
    assert submitted_values[
        (
            "product_0",
            "quantiles",
        )
    ] == [
        {
            "level": 0.1,
            "value": 70.0,
        },
        {
            "level": 0.5,
            "value": 100.0,
        },
        {
            "level": 0.9,
            "value": 140.0,
        },
    ]

    streamlit.data_editor.assert_called_once()
    call_kwargs = streamlit.data_editor.call_args.kwargs

    assert call_kwargs["disabled"] == ["product_id"]
    assert call_kwargs["num_rows"] == "fixed"
    assert call_kwargs["hide_index"] is True
    assert call_kwargs["column_order"] == [
        "product_id",
        "expected_value",
        "quantiles",
        "horizon_days",
        "selling_price",
        "purchase_cost",
    ]
