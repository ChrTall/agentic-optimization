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


def create_streamlit_mock(*, button: bool = False) -> MagicMock:
    """Streamlit-Mock, dessen Widgets ihre Ausgangswerte zurückgeben."""

    streamlit = MagicMock()
    streamlit.button.return_value = button
    streamlit.checkbox.side_effect = lambda label, value=False, **_: value
    streamlit.text_input.side_effect = lambda label, value="", **_: value
    streamlit.columns.side_effect = lambda count: [MagicMock() for _ in range(count)]
    return streamlit


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

    specification = create_multi_product_specification()
    layout = build_modular_input_layout(specification)

    rendered_inputs = _render_product_input_table(
        layout,
        specification,
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


def test_renders_constraint_component_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = create_streamlit_mock()

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
    streamlit.markdown.assert_any_call(
        "**Mindestbestellmenge**"
    )


def test_submits_constraint_value_with_one_button_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = create_streamlit_mock(button=True)

    monkeypatch.setattr(
        modular_input_page,
        "st",
        streamlit,
    )
    monkeypatch.setattr(
        modular_input_page,
        "render_input_field",
        # Ausgefüllte Felder behalten ihren Wert, das offene Feld erhält 250.
        lambda field, existing, **kwargs: (
            existing if existing is not None else 250.0
        ),
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
    assert result["action"] == "submit"
    assert result["renames"] == []
    constraint_submissions = [
        submission
        for submission in result["submissions"]
        if submission["missing_input"]["source"] == "constraint_parameter"
    ]
    assert len(constraint_submissions) == 1
    assert constraint_submissions[0]["value"] == 250.0
    assert constraint_submissions[0]["missing_input"][
        "component_id"
    ] == "maximum_order_quantity"


def create_complete_eoq_specification() -> dict:
    return {
        "core_id": "eoq",
        "products": [
            {
                "product_id": "Produkt 1",
                "forecast": {"expected_value": 1200.0, "horizon_days": 365},
                "facts": {"ordering_cost": 150.0, "holding_cost": 25.0},
            }
        ],
        "global_facts": {},
        "objective": {
            "sense": "minimize",
            "component_ids": ["annual_ordering_cost", "annual_holding_cost"],
        },
        "constraints": [],
    }


def test_filled_fields_stay_visible_in_fixed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = create_streamlit_mock()
    rendered: list[str] = []
    monkeypatch.setattr(modular_input_page, "st", streamlit)
    monkeypatch.setattr(
        modular_input_page,
        "render_input_field",
        lambda field, existing, **_: rendered.append(field["name"]) or existing,
    )

    assert render_modular_input_page(create_complete_eoq_specification()) is None
    assert rendered == [
        "expected_value",
        "horizon_days",
        "ordering_cost",
        "holding_cost",
    ]
    streamlit.success.assert_called_once()


def test_returns_product_rename(monkeypatch: pytest.MonkeyPatch) -> None:
    streamlit = create_streamlit_mock(button=True)
    streamlit.text_input.side_effect = lambda label, value="", **_: "E-Bike"
    monkeypatch.setattr(modular_input_page, "st", streamlit)
    monkeypatch.setattr(
        modular_input_page,
        "render_input_field",
        lambda field, existing, **_: existing,
    )

    result = render_modular_input_page(create_complete_eoq_specification())

    assert result["renames"] == [("Produkt 1", "E-Bike")]


def test_toggling_component_returns_new_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = create_streamlit_mock()
    streamlit.checkbox.side_effect = lambda label, value=False, **_: (
        True if label == "Höchstbestellmenge" else value
    )
    monkeypatch.setattr(modular_input_page, "st", streamlit)
    monkeypatch.setattr(
        modular_input_page,
        "render_input_field",
        lambda field, existing, **_: existing,
    )

    result = render_modular_input_page(create_complete_eoq_specification())

    assert result["action"] == "edit"
    assert "maximum_order_quantity" in result["active_component_ids"]
    assert "annual_ordering_cost" in result["active_component_ids"]
