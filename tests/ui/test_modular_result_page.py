from unittest.mock import MagicMock

import pytest

import agentic_optimization.ui.modular_result_page as modular_result_page
from agentic_optimization.ui.modular_result_page import (
    _build_result_overview_rows,
    render_modular_result_page,
)


def test_builds_multi_product_result_overview() -> None:
    rows = _build_result_overview_rows(
        [
            {
                "value_id": "order_quantity",
                "label": "Bestellmenge",
                "value": 100.0,
                "product_id": "product_1",
                "aggregation": None,
            },
            {
                "value_id": "reorder_point",
                "label": "Meldebestand",
                "value": 40.0,
                "product_id": "product_1",
                "aggregation": None,
            },
            {
                "value_id": "order_quantity",
                "label": "Bestellmenge",
                "value": 250.0,
                "product_id": "product_2",
                "aggregation": None,
            },
            {
                "value_id": "reorder_point",
                "label": "Meldebestand",
                "value": 85.0,
                "product_id": "product_2",
                "aggregation": None,
            },
        ]
    )

    assert rows == [
        {
            "Produkt": "product_1",
            "Bestellmenge": "100,00",
            "Meldebestand": "40,00",
        },
        {
            "Produkt": "product_2",
            "Bestellmenge": "250,00",
            "Meldebestand": "85,00",
        },
    ]


def test_builds_overall_result_row() -> None:
    rows = _build_result_overview_rows(
        [
            {
                "value_id": "annual_relevant_cost",
                "label": "Jährliche Gesamtkosten",
                "value": 5000.0,
                "product_id": None,
                "aggregation": "total",
            }
        ]
    )

    assert rows == [
        {
            "Produkt": "Gesamtmodell",
            "Jährliche Gesamtkosten": "5.000,00",
        }
    ]


def test_renders_multi_product_results_as_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()

    monkeypatch.setattr(
        modular_result_page,
        "st",
        streamlit,
    )

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
    }

    result = {
        "status": "optimal",
        "solution_method": "analytical",
        "objective_value": 5000.0,
        "decisions": [
            {
                "quantity_id": "order_quantity",
                "product_id": "product_1",
                "value": 100.0,
            },
            {
                "quantity_id": "order_quantity",
                "product_id": "product_2",
                "value": 250.0,
            },
        ],
        "metrics": [
            {
                "metric_id": "average_inventory",
                "product_id": "product_1",
                "value": 50.0,
            },
            {
                "metric_id": "average_inventory",
                "product_id": "product_2",
                "value": 125.0,
            },
        ],
        "technical_representation": None,
        "errors": [],
    }

    render_modular_result_page(
        specification,
        result,
    )

    streamlit.dataframe.assert_any_call(
        [
            {
                "Produkt": "product_1",
                "Bestellmenge": "100,00",
            },
            {
                "Produkt": "product_2",
                "Bestellmenge": "250,00",
            },
        ],
        use_container_width=True,
        hide_index=True,
    )
    streamlit.dataframe.assert_any_call(
        [
            {
                "Produkt": "product_1",
                "Durchschnittlicher Lagerbestand": (
                    "50,00"
                ),
            },
            {
                "Produkt": "product_2",
                "Durchschnittlicher Lagerbestand": (
                    "125,00"
                ),
            },
        ],
        use_container_width=True,
        hide_index=True,
    )


def test_renders_excluded_requirements_prominently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = MagicMock()

    monkeypatch.setattr(
        modular_result_page,
        "st",
        streamlit,
    )

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
        "decisions": [],
        "metrics": [],
        "technical_representation": None,
        "errors": [],
    }

    render_modular_result_page(
        specification,
        result,
    )

    streamlit.markdown.assert_any_call(
        "### Geltungsbereich der Empfehlung"
    )
    streamlit.warning.assert_called_once_with(
        "Die folgenden Anforderungen wurden bei der "
        "Berechnung nicht berücksichtigt. Die Empfehlung "
        "gilt daher nur, wenn diese Vorgaben außer Acht "
        "gelassen werden können."
    )
    streamlit.markdown.assert_any_call(
        "**Nicht berücksichtigt:** "
        "Jede Lieferung muss an einem Dienstag "
        "eintreffen."
    )
    streamlit.write.assert_called_once_with(
        "Ein bestimmter Liefertag wird nicht unterstützt."
    )