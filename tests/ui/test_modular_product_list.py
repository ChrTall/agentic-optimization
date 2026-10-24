from agentic_optimization.ui.modular_product_list import (
    parse_dynamic_product_table,
)
from agentic_optimization.ui.modular_input_builder import (
    build_dynamic_product_table_schema,
)


def test_product_names_are_editable_metadata_and_generated_ids_do_not_collide():
    schema = {"forecast_fields": [], "fact_fields": []}
    result = parse_dynamic_product_table([
        {"product_id": "product_1", "display_name": "Weihnachtsbäume"},
        {"display_name": "Christbaumständer"},
    ], schema, generate_product_ids=True)
    assert not result["errors"]
    assert [p["product_id"] for p in result["products"]] == ["product_1", "product_2"]
    assert result["products"][1]["display_name"] == "Christbaumständer"
    result = parse_dynamic_product_table([
        {"product_id": "product_1", "display_name": "Nordmanntannen"},
    ], schema, generate_product_ids=True)
    assert result["products"][0]["product_id"] == "product_1"
    assert result["products"][0]["display_name"] == "Nordmanntannen"


def test_parses_complete_dynamic_product_table() -> None:
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
    schema = build_dynamic_product_table_schema(
        "single_period_newsvendor",
        proposal,
    )

    result = parse_dynamic_product_table(
        [
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
        ],
        schema,
    )

    assert result["errors"] == []
    assert result["products"][0] == {
        "product_id": "Artikel-A",
        "forecast": {
            "expected_value": 100.0,
            "quantiles": [
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
            ],
            "horizon_days": 30,
        },
        "facts": {
            "selling_price": 12.0,
            "purchase_cost": 5.0,
        },
    }
    assert result["products"][1]["product_id"] == (
        "Artikel-B"
    )


def test_rejects_incomplete_dynamic_product_row() -> None:
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
    schema = build_dynamic_product_table_schema(
        "single_period_newsvendor",
        proposal,
    )

    result = parse_dynamic_product_table(
        [
            {
                "product_id": "Artikel-A",
                "expected_value": 100,
                "quantiles": (
                    "0.1:70; 0.5:100; 0.9:140"
                ),
                "horizon_days": 30,
                "selling_price": 12,
                "purchase_cost": None,
            }
        ],
        schema,
    )

    assert result["products"] == []
    assert result["errors"] == [
        "Artikel-A: Bitte geben Sie einen Wert für "
        "„Einkaufskosten pro Einheit“ ein."
    ]
