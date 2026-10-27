"""Bei widersprüchlichen Angaben fragt der Chat nicht erneut alles ab."""

import pytest

from agentic_optimization.ui.chat_assistant_builder import (
    build_next_question,
    flatten_filled_inputs,
    open_model_inputs,
)
from agentic_optimization.ui.field_renderer import parse_input_value


def _graph_state(expected_value: float) -> dict:
    return {"modular_model": {"specification": {
        "core_id": "single_period_newsvendor",
        "products": [{
            "product_id": "Weihnachtsbaum",
            "forecast": {
                "expected_value": expected_value,
                "horizon_days": 55,
                "quantiles": [
                    {"level": 0.1, "value": 150.0},
                    {"level": 0.5, "value": 300.0},
                    {"level": 0.9, "value": 350.0},
                ],
            },
            "facts": {"selling_price": 25.0, "purchase_cost": 10.0},
        }],
        "global_facts": {},
        "objective": {
            "sense": "maximize",
            "component_ids": ["sales_revenue", "purchase_cost"],
        },
        "constraints": [],
    }}}


def test_contradiction_is_reported_instead_of_asking_again() -> None:
    graph_state = _graph_state(5000)
    specification = graph_state["modular_model"]["specification"]

    question = build_next_question("model_input_form", graph_state)

    assert question["target"]["kind"] == "validation_error"
    assert "5000" in question["text"]
    assert open_model_inputs(specification) == []
    assert len(flatten_filled_inputs(specification)) == 5


def test_valid_inputs_continue_normally() -> None:
    question = build_next_question("model_input_form", _graph_state(276))

    assert question["target"]["kind"] == "inputs_complete"


def test_decreasing_quantiles_are_rejected_on_input() -> None:
    from agentic_optimization.models.core_registry import CORE_REGISTRY

    field = next(
        field
        for field in CORE_REGISTRY["single_period_newsvendor"]["definition"][
            "required_forecast_fields"
        ]
        if field["name"] == "quantiles"
    )

    with pytest.raises(ValueError, match="monoton steigen"):
        parse_input_value(field, "0.1: 300; 0.5: 150; 0.9: 350")
