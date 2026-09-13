import pytest

from agentic_optimization.agents.modular_state_adapter import (
    initialize_modular_model_state,
)


@pytest.mark.parametrize(
    ("family_id", "variant_id", "core_id"),
    [
        (
            "newsvendor",
            "single_product_single_period",
            "single_period_newsvendor",
        ),
        (
            "eoq",
            "basic_eoq",
            "eoq",
        ),
        (
            "reorder_point",
            "q_r_policy",
            "continuous_review_q_r",
        ),
    ],
)
def test_maps_legacy_selection_to_modular_core(
    family_id: str,
    variant_id: str,
    core_id: str,
) -> None:
    state = {
        "selected_model_family": family_id,
        "selected_model_variant": variant_id,
        "requirements": {
            "product_scope": "single_product",
            "constraints": [],
        },
    }

    result = initialize_modular_model_state(state)

    assert result["modular_model"]["core_id"] == core_id
    assert result["modular_model"]["products"] == [
        {
            "product_id": "product_1",
            "forecast": {},
            "facts": {},
        }
    ]


def test_preserves_products_facts_and_evidence() -> None:
    state = {
        "selected_model_family": "newsvendor",
        "selected_model_variant": (
            "single_product_single_period"
        ),
        "raw_input": "Mindestbestellmenge 50 Stück.",
        "requirements": {
            "product_scope": "single_product",
            "constraints": [
                "Mindestbestellmenge 50 Stück.",
            ],
        },
        "modular_model": {
            "products": [
                {
                    "product_id": "article_1",
                    "forecast": {
                        "expected_value": 100.0,
                    },
                    "facts": {
                        "selling_price": 12.0,
                    },
                }
            ],
            "global_facts": {
                "currency": "EUR",
            },
            "evidence": [
                {
                    "source_id": "document_1",
                    "content": "Lieferantenvertrag",
                }
            ],
        },
    }

    result = initialize_modular_model_state(state)
    modular_state = result["modular_model"]

    assert modular_state["products"][0][
        "product_id"
    ] == "article_1"
    assert modular_state["global_facts"] == {
        "currency": "EUR",
    }
    assert {
        item["source_id"]
        for item in modular_state["evidence"]
    } == {
        "document_1",
        "raw_input",
        "requirement_constraint_0",
    }


def test_requires_product_list_for_multiple_products() -> None:
    state = {
        "selected_model_family": "newsvendor",
        "selected_model_variant": (
            "single_product_single_period"
        ),
        "requirements": {
            "product_scope": "multiple_products",
            "constraints": [],
        },
    }

    with pytest.raises(
        ValueError,
        match="Produktliste",
    ):
        initialize_modular_model_state(state)
