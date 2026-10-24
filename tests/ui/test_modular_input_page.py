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
    render_modular_input_page,
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


def test_table_preserves_edits_when_adding_fields_and_receiving_chat_values(monkeypatch):
    from agentic_optimization.ui.modular_components import add_component

    streamlit = MagicMock()
    streamlit.session_state = {}
    monkeypatch.setattr(modular_input_page, "st", streamlit)
    specification = create_multi_product_specification()
    for product in specification["products"]:
        product["forecast"] = {"expected_value": 100, "horizon_days": 30,
                               "quantiles": [{"level": 0.5, "value": 100}]}
        product["facts"] = {"selling_price": 25, "purchase_cost": 10}

    def edit_rows(rows, **kwargs):
        rows[0]["selling_price"] = 27
        return rows

    streamlit.data_editor.side_effect = edit_rows
    layout = build_modular_input_layout(specification, include_completed=True)
    _render_product_input_table(layout, specification)
    specification = add_component(specification, "objective:salvage_revenue")
    specification["products"][1]["facts"]["selling_price"] = 26
    streamlit.data_editor.side_effect = lambda rows, **kwargs: rows
    layout = build_modular_input_layout(specification, include_completed=True)
    _render_product_input_table(layout, specification)
    rows = streamlit.data_editor.call_args.args[0]
    assert rows[0]["selling_price"] == 27
    assert rows[1]["selling_price"] == 26
    assert rows[0]["salvage_value"] is None
    assert rows[0]["quantiles"] == "0.5: 100"


def test_table_callback_syncs_valid_cells_and_keeps_invalid_cells_open(monkeypatch):
    specification = create_multi_product_specification()
    streamlit = MagicMock()
    streamlit.session_state = {
        "graph_state": {"modular_model": {
            "specification": specification,
            "proposal": {"objective_components": [], "constraints": []},
        }},
        "modular_table_base": [{"product_id": p["product_id"]}
                               for p in specification["products"]],
        "modular_product_input_table": {"edited_rows": {
            0: {"expected_value": 100, "quantiles": "ungültig"},
            1: {"selling_price": 25},
        }},
    }
    monkeypatch.setattr(modular_input_page, "st", streamlit)
    layout = build_modular_input_layout(specification, include_completed=True)
    modular_input_page._sync_product_table(layout)
    model = streamlit.session_state["graph_state"]["modular_model"]
    assert model["specification"]["products"][0]["forecast"]["expected_value"] == 100
    assert model["specification"]["products"][0]["forecast"]["quantiles"] is None
    assert model["specification"]["products"][1]["facts"]["selling_price"] == 25
    assert ("forecast", "product_0", None, "quantiles") in streamlit.session_state["modular_input_drafts"]
    assert model["form_3_review_confirmed"] is False


def test_renders_constraint_component_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()
    streamlit.button.return_value = False
    streamlit.session_state = {}
    streamlit.checkbox.side_effect = lambda *args, **kwargs: streamlit.session_state.get(kwargs["key"], False)
    monkeypatch.setattr(modular_input_page, "render_input_field", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        modular_input_page,
        "st",
        streamlit,
    )

    specification = {
        "core_id": "eoq",
        "products": [
            {
                "product_id": "product_1",
                "forecast": {
                    "expected_value": 3500.0,
                    "horizon_days": 365,
                },
                "facts": {
                    "ordering_cost": 120.0,
                    "holding_cost": 12.0,
                },
            }
        ],
        "global_facts": {},
        "objective": {
            "sense": "minimize",
            "component_ids": [
                "annual_ordering_cost",
                "annual_holding_cost",
            ],
        },
        "constraints": [
            {
                "component_id": "minimum_order_quantity",
                "product_id": "product_1",
                "parameters": {},
            }
        ],
    }

    result = render_modular_input_page(
        specification
    )

    assert result is None
    assert any(call.args[0] == "Mindestbestellmenge – product_1" for call in streamlit.checkbox.call_args_list)


def test_submits_constraint_value_with_one_button_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()
    streamlit.session_state = {}
    streamlit.checkbox.side_effect = lambda *args, **kwargs: (
        True if kwargs["key"] == "modular_additions_confirmed"
        else streamlit.session_state.get(kwargs["key"], False)
    )
    streamlit.selectbox.return_value = None
    streamlit.button.side_effect = lambda *args, **kwargs: kwargs["key"] == "submit_modular_model_inputs"

    monkeypatch.setattr(
        modular_input_page,
        "st",
        streamlit,
    )
    monkeypatch.setattr(
        modular_input_page,
        "render_input_field",
        lambda field, existing, **kwargs: 250.0 if field["name"] == "quantity" else existing,
    )

    specification = {
        "core_id": "single_period_newsvendor",
        "products": [
            {
                "product_id": "product_1",
                "forecast": {
                    "expected_value": 276.0,
                    "quantiles": [
                        {"level": 0.1, "value": 150.0},
                        {"level": 0.5, "value": 300.0},
                        {"level": 0.9, "value": 350.0},
                    ],
                    "horizon_days": 60,
                },
                "facts": {
                    "selling_price": 25.0,
                    "purchase_cost": 10.0,
                    "salvage_value": 3.0,
                },
            }
        ],
        "global_facts": {},
        "objective": {
            "sense": "maximize",
            "component_ids": [
                "sales_revenue",
                "salvage_revenue",
                "purchase_cost",
            ],
        },
        "constraints": [
            {
                "component_id": "maximum_order_quantity",
                "product_id": "product_1",
                "parameters": {},
            }
        ],
    }

    result = render_modular_input_page(specification)

    assert result is not None
    constraint = next(item for item in result if item["missing_input"]["source"] == "constraint_parameter")
    assert constraint["value"] == 250.0
    assert constraint["missing_input"][
        "component_id"
    ] == "maximum_order_quantity"
    streamlit.button.assert_any_call(
        "Weiter zur Überprüfung",
        type="primary",
        width="stretch",
        key="submit_modular_model_inputs",
        disabled=False,
    )
