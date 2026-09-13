import pytest

from agentic_optimization.models.fact_extraction import (
    merge_extracted_facts,
)


def create_products() -> list[dict]:
    return [
        {
            "product_id": "product_1",
            "forecast": {},
            "facts": {},
        }
    ]


def test_merges_allowed_extracted_facts() -> None:
    products, global_facts = merge_extracted_facts(
        core_id="single_period_newsvendor",
        products=create_products(),
        global_facts={},
        extraction={
            "products": [
                {
                    "product_id": "product_1",
                    "forecast": {
                        "expected_value": 100.0,
                        "quantiles": [
                            {
                                "level": 0.5,
                                "value": 100.0,
                            }
                        ],
                        "horizon_days": 30,
                    },
                    "facts": {
                        "selling_price": 12.0,
                        "purchase_cost": 5.0,
                    },
                }
            ],
            "global_facts": {},
        },
    )

    assert products[0]["facts"] == {
        "selling_price": 12.0,
        "purchase_cost": 5.0,
    }
    assert products[0]["forecast"][
        "expected_value"
    ] == 100.0
    assert global_facts == {}


def test_does_not_overwrite_existing_values() -> None:
    products = create_products()
    products[0]["facts"]["selling_price"] = 15.0

    updated_products, _ = merge_extracted_facts(
        core_id="single_period_newsvendor",
        products=products,
        global_facts={},
        extraction={
            "products": [
                {
                    "product_id": "product_1",
                    "forecast": {},
                    "facts": {
                        "selling_price": 12.0,
                    },
                }
            ],
            "global_facts": {},
        },
    )

    assert updated_products[0]["facts"][
        "selling_price"
    ] == 15.0


def test_rejects_unknown_product_fact() -> None:
    with pytest.raises(
        ValueError,
        match="Nicht erlaubte Produktfelder",
    ):
        merge_extracted_facts(
            core_id="single_period_newsvendor",
            products=create_products(),
            global_facts={},
            extraction={
                "products": [
                    {
                        "product_id": "product_1",
                        "forecast": {},
                        "facts": {
                            "invented_cost": 10.0,
                        },
                    }
                ],
                "global_facts": {},
            },
        )


def test_rejects_unknown_product() -> None:
    with pytest.raises(
        ValueError,
        match="nicht im Modellentwurf",
    ):
        merge_extracted_facts(
            core_id="single_period_newsvendor",
            products=create_products(),
            global_facts={},
            extraction={
                "products": [
                    {
                        "product_id": "unknown",
                        "forecast": {},
                        "facts": {},
                    }
                ],
                "global_facts": {},
            },
        )


def test_does_not_modify_original_products() -> None:
    products = create_products()

    updated_products, _ = merge_extracted_facts(
        core_id="single_period_newsvendor",
        products=products,
        global_facts={},
        extraction={
            "products": [
                {
                    "product_id": "product_1",
                    "forecast": {},
                    "facts": {
                        "selling_price": 12.0,
                    },
                }
            ],
            "global_facts": {},
        },
    )

    assert updated_products[0]["facts"] == {
        "selling_price": 12.0,
    }
    assert products[0]["facts"] == {}
