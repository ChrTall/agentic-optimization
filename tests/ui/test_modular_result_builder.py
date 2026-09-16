import pytest

from agentic_optimization.ui.modular_result_builder import (
    build_default_result_explanation,
    build_modular_result_data,
)


def test_builds_modular_eoq_result_data() -> None:
    specification = {
        "core_id": "eoq",
        "products": [],
        "global_facts": {},
        "objective": {
            "sense": "minimize",
            "component_ids": [
                "annual_ordering_cost",
                "annual_holding_cost",
            ],
        },
        "constraints": [],
        "excluded_requirements": [
            {
                "description": (
                    "Jede Lieferung muss an einem "
                    "Dienstag eintreffen."
                ),
                "reason": (
                    "Ein bestimmter Liefertag wird "
                    "nicht unterstützt."
                ),
                "sources": ["raw_input"],
                "confirmation_source": (
                    "modular_unsupported_review"
                ),
            }
        ],
    }

    result = {
        "status": "optimal",
        "solution_method": "analytical",
        "objective_value": 3174.9,
        "decisions": [
            {
                "quantity_id": "order_quantity",
                "product_id": "product_1",
                "value": 264.58,
            }
        ],
        "metrics": [
            {
                "metric_id": "average_inventory",
                "product_id": "product_1",
                "value": 132.29,
            },
            {
                "metric_id": "annual_relevant_cost",
                "product_id": "product_1",
                "value": 3174.9,
            },
        ],
        "technical_representation": (
            "Q* = sqrt(2 * D * S / H)"
        ),
        "errors": [],
    }

    result_data = build_modular_result_data(
        specification,
        result,
    )

    assert result_data["core_name"] == (
        "Wirtschaftliche Bestellmenge (EOQ)"
    )
    assert result_data["status_label"] == (
        "Optimale Lösung gefunden"
    )
    assert result_data["solution_method_label"] == (
        "Analytische Berechnung"
    )
    assert result_data["successful"] is True

    assert result_data["decisions"][0]["label"] == (
        "Bestellmenge"
    )
    assert result_data["metrics"][0]["label"] == (
        "Durchschnittlicher Lagerbestand"
    )
    assert result_data["metrics"][1]["label"] == (
        "Jährliche relevante Gesamtkosten"
    )
    assert result_data["excluded_requirements"] == [
        {
            "description": (
                "Jede Lieferung muss an einem "
                "Dienstag eintreffen."
            ),
            "reason": (
                "Ein bestimmter Liefertag wird "
                "nicht unterstützt."
            ),
            "sources": ["raw_input"],
            "confirmation_source": (
                "modular_unsupported_review"
            ),
        }
    ]


def test_builds_default_result_explanation() -> None:
    result_data = {
        "core_id": "eoq",
        "core_name": "Wirtschaftliche Bestellmenge (EOQ)",
        "status": "optimal",
        "status_label": "Optimale Lösung gefunden",
        "successful": True,
        "solution_method": "analytical",
        "solution_method_label": (
            "Analytische Berechnung"
        ),
        "objective_value": 3174.9,
        "decisions": [
            {
                "value_id": "order_quantity",
                "label": "Bestellmenge",
                "value": 264.58,
                "product_id": "product_1",
                "aggregation": None,
            }
        ],
        "metrics": [],
        "excluded_requirements": [],
        "technical_representation": None,
        "errors": [],
    }

    explanation = build_default_result_explanation(
        result_data
    )

    assert explanation == (
        "Optimale Lösung gefunden. Die zentrale Empfehlung "
        "lautet für product_1: Bestellmenge = 264.58. "
        "Die weiteren Kennzahlen zeigen die betrieblichen "
        "Auswirkungen dieser Empfehlung."
    )


def test_rejects_unknown_core() -> None:
    specification = {
        "core_id": "unknown_core",
        "products": [],
        "global_facts": {},
        "objective": {
            "sense": "minimize",
            "component_ids": [],
        },
        "constraints": [],
    }

    result = {
        "status": "failed",
        "solution_method": "numerical",
        "objective_value": None,
        "decisions": [],
        "metrics": [],
        "technical_representation": None,
        "errors": ["Fehler"],
    }

    with pytest.raises(
        ValueError,
        match="nicht registriert",
    ):
        build_modular_result_data(
            specification,
            result,
        )
