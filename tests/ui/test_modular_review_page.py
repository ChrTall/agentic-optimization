from agentic_optimization.ui.modular_review_page import (
    _build_constraint_parameter_labels,
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