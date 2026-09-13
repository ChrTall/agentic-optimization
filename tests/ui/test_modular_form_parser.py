import pytest

from agentic_optimization.ui.modular_form_parser import (
    parse_modular_input_submissions,
)


def create_missing_input(
    *,
    name: str,
    field_type: str,
) -> dict:
    return {
        "source": "product_fact",
        "field": {
            "name": name,
            "label": name,
            "type": field_type,
            "required": True,
            "help": None,
        },
        "product_id": "product_1",
    }


def test_parses_numeric_submission() -> None:
    missing_input = create_missing_input(
        name="selling_price",
        field_type="number",
    )

    result = parse_modular_input_submissions(
        [
            {
                "missing_input": missing_input,
                "raw_value": 12.5,
            }
        ]
    )

    assert result == [
        {
            "missing_input": missing_input,
            "value": 12.5,
        }
    ]


def test_parses_forecast_quantiles_submission() -> None:
    missing_input = {
        "source": "forecast",
        "field": {
            "name": "quantiles",
            "label": "Nachfragequantile",
            "type": "forecast_quantiles",
            "required": True,
            "help": None,
        },
        "product_id": "product_1",
    }

    result = parse_modular_input_submissions(
        [
            {
                "missing_input": missing_input,
                "raw_value": (
                    "0.1: 70; 0.5: 100; 0.9: 140"
                ),
            }
        ]
    )

    assert result[0]["value"] == [
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


def test_rejects_empty_required_submission() -> None:
    missing_input = create_missing_input(
        name="selling_price",
        field_type="number",
    )

    with pytest.raises(
        ValueError,
        match="selling_price",
    ):
        parse_modular_input_submissions(
            [
                {
                    "missing_input": missing_input,
                    "raw_value": None,
                }
            ]
        )


def test_does_not_modify_missing_input() -> None:
    missing_input = create_missing_input(
        name="holding_cost",
        field_type="number",
    )

    parse_modular_input_submissions(
        [
            {
                "missing_input": missing_input,
                "raw_value": 4.0,
            }
        ]
    )

    assert missing_input["field"]["name"] == (
        "holding_cost"
    )
