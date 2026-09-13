import json

import pytest

from agentic_optimization.models.component_catalog import (
    build_component_catalog,
)
from agentic_optimization.models.component_registry import (
    OBJECTIVE_COMPONENTS,
)


def test_builds_newsvendor_component_catalog() -> None:
    catalog = build_component_catalog(
        "single_period_newsvendor"
    )

    assert catalog["core"]["core_id"] == (
        "single_period_newsvendor"
    )

    objective_ids = {
        component["component_id"]
        for component in catalog["objective_components"]
    }
    constraint_ids = {
        component["component_id"]
        for component in catalog["constraint_components"]
    }

    assert objective_ids == {
        "sales_revenue",
        "salvage_revenue",
        "disposal_cost",
        "purchase_cost",
        "shortage_cost",
    }
    assert constraint_ids == {
        "minimum_order_quantity",
        "maximum_order_quantity",
        "order_multiple",
        "budget_limit",
        "storage_capacity",
    }
    assert catalog["core"][
        "required_objective_component_ids"
    ] == []


def test_catalog_is_json_serializable() -> None:
    catalog = build_component_catalog(
        "single_period_newsvendor"
    )

    serialized_catalog = json.dumps(catalog)

    assert "single_period_newsvendor" in serialized_catalog
    assert "builder" not in serialized_catalog


def test_catalog_does_not_expose_mutable_registry_data() -> None:
    catalog = build_component_catalog(
        "single_period_newsvendor"
    )

    sales_component = next(
        component
        for component in catalog["objective_components"]
        if component["component_id"] == "sales_revenue"
    )

    sales_component["required_product_facts"][0][
        "label"
    ] = "Changed"

    registry_label = OBJECTIVE_COMPONENTS[
        "sales_revenue"
    ]["required_product_facts"][0]["label"]

    assert registry_label != "Changed"


def test_rejects_unknown_core() -> None:
    with pytest.raises(
        ValueError,
        match="unknown_core",
    ):
        build_component_catalog("unknown_core")


def test_builds_eoq_component_catalog() -> None:
    catalog = build_component_catalog("eoq")

    objective_ids = {
        component["component_id"]
        for component in catalog["objective_components"]
    }
    constraint_ids = {
        component["component_id"]
        for component in catalog["constraint_components"]
    }
    forecast_field_names = {
        field["name"]
        for field in catalog["core"][
            "required_forecast_fields"
        ]
    }

    assert objective_ids == {
        "annual_ordering_cost",
        "annual_holding_cost",
    }
    assert constraint_ids == {
        "minimum_order_quantity",
        "maximum_order_quantity",
        "order_multiple",
    }
    assert forecast_field_names == {
        "expected_value",
        "horizon_days",
    }
    assert catalog["core"][
        "required_objective_component_ids"
    ] == [
        "annual_ordering_cost",
        "annual_holding_cost",
    ]


def test_builds_continuous_review_q_r_catalog() -> None:
    catalog = build_component_catalog(
        "continuous_review_q_r"
    )

    objective_ids = {
        component["component_id"]
        for component in catalog["objective_components"]
    }
    constraint_ids = {
        component["component_id"]
        for component in catalog["constraint_components"]
    }

    assert objective_ids == {
        "annual_ordering_cost",
        "annual_holding_cost",
        "annual_shortage_cost",
    }
    assert constraint_ids == {
        "minimum_order_quantity",
        "maximum_order_quantity",
        "order_multiple",
        "minimum_reorder_point",
        "maximum_reorder_point",
    }
    assert catalog["core"][
        "required_objective_component_ids"
    ] == [
        "annual_ordering_cost",
        "annual_holding_cost",
        "annual_shortage_cost",
    ]
