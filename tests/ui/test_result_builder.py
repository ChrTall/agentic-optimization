from agentic_optimization.models.newsvendor.single_product_single_period import (
    VARIANT_DEFINITION,
)
from agentic_optimization.ui.result_builder import (
    build_result_data,
)


def create_result_state() -> dict:
    return {
        "selected_model_family": "newsvendor",
        "selected_model_variant": (
            "single_product_single_period"
        ),
        "requirements": {
            "decision": "order_quantity",
            "objective": "maximize_profit",
            "demand_type": "uncertain",
            "product_scope": "single_product",
            "planning_horizon": "single_period",
        },
        "model_specific_inputs": {
            "newsvendor": {
                "single_product_single_period": {
                    "selling_price": 20.0,
                    "purchase_cost": 12.0,
                    "salvage_value": 3.0,
                    "demand_scenarios": [
                        80.0,
                        100.0,
                        140.0,
                    ],
                    "scenario_probabilities": [
                        0.2,
                        0.5,
                        0.3,
                    ],
                },
            },
        },
        "structured_constraints": [
            {
                "type": "budget_limit",
                "parameters": {
                    "budget": 5000.0,
                },
                "source_text": (
                    "Maximales Budget von 5000 Euro."
                ),
            },
        ],
        "optimization_result": {
            "solver_status": "Optimal",
            "order_quantity": 120.0,
            "expected_profit": 840.0,
            "scenario_results": [
                {
                    "demand": 80.0,
                    "probability": 0.2,
                    "sold": 80.0,
                    "leftover": 40.0,
                },
                {
                    "demand": 100.0,
                    "probability": 0.5,
                    "sold": 100.0,
                    "leftover": 20.0,
                },
                {
                    "demand": 140.0,
                    "probability": 0.3,
                    "sold": 120.0,
                    "leftover": 0.0,
                },
            ],
            "model_representation": (
                "Newsvendor_Model:\nMAXIMIZE\n..."
            ),
        },
        "optimization_errors": [],
    }


def test_builds_successful_newsvendor_result() -> None:
    result_data = build_result_data(
        create_result_state()
    )

    assert result_data["successful"] is True

    assert (
        result_data["status_label"]
        == "Optimale Lösung gefunden"
    )

    assert result_data["metrics"] == [
        {
            "label": "Empfohlene Bestellmenge",
            "value": "120 Einheiten",
        },
        {
            "label": "Erwarteter Gewinn",
            "value": "840,00 €",
        },
    ]


def test_builds_newsvendor_scenario_details() -> None:
    result_data = build_result_data(
        create_result_state()
    )

    assert (
        result_data["detail_title"]
        == "Ergebnis nach Nachfrageszenario"
    )

    assert result_data["detail_rows"] == [
        {
            "Nachfrage": "80",
            "Wahrscheinlichkeit": "20 %",
            "Verkauft": "80",
            "Restbestand": "40",
        },
        {
            "Nachfrage": "100",
            "Wahrscheinlichkeit": "50 %",
            "Verkauft": "100",
            "Restbestand": "20",
        },
        {
            "Nachfrage": "140",
            "Wahrscheinlichkeit": "30 %",
            "Verkauft": "120",
            "Restbestand": "0",
        },
    ]


def test_builds_active_constraints() -> None:
    result_data = build_result_data(
        create_result_state()
    )

    assert result_data["constraints"] == [
        {
            "label": "Budgetgrenze",
            "value": "5000 €",
        }
    ]


def test_allows_result_without_constraints() -> None:
    state = create_result_state()
    state["structured_constraints"] = []

    result_data = build_result_data(state)

    assert result_data["constraints"] == []


def test_does_not_show_metrics_for_infeasible_result() -> None:
    state = create_result_state()

    state["optimization_result"] = {
        "solver_status": "Infeasible",
        "order_quantity": None,
        "expected_profit": None,
        "scenario_results": [],
        "model_representation": "Infeasible model",
    }

    result_data = build_result_data(state)

    assert result_data["successful"] is False
    assert result_data["metrics"] == []
    assert result_data["detail_rows"] == []

    assert (
        result_data["status_label"]
        == "Keine zulässige Lösung"
    )

    assert "keine zulässige Lösung" in (
        result_data["status_message"]
    )


def test_optimization_errors_override_success_status() -> None:
    state = create_result_state()

    state["optimization_errors"] = [
        "Der Solver konnte nicht gestartet werden.",
    ]

    result_data = build_result_data(state)

    assert result_data["successful"] is False
    assert (
        result_data["status_label"]
        == "Berechnung fehlgeschlagen"
    )
    assert result_data["metrics"] == []
    assert result_data["detail_rows"] == []
    assert result_data["errors"] == [
        "Der Solver konnte nicht gestartet werden.",
    ]


def test_builds_technical_details() -> None:
    result_data = build_result_data(
        create_result_state()
    )

    assert result_data["family_name"] == "Newsvendor"

    assert (
        result_data["variant_name"]
        == "Single Product, Single Period"
    )

    assert result_data["solver_status"] == "Optimal"

    assert result_data["model_representation"] == (
        "Newsvendor_Model:\nMAXIMIZE\n..."
    )


def test_variant_registers_result_builder() -> None:
    result_builder = VARIANT_DEFINITION[
        "result_builder"
    ]

    presentation = result_builder(
        create_result_state()[
            "optimization_result"
        ]
    )

    assert presentation["metrics"] == [
        {
            "label": "Empfohlene Bestellmenge",
            "value": "120 Einheiten",
        },
        {
            "label": "Erwarteter Gewinn",
            "value": "840,00 €",
        },
    ]

    assert (
        presentation["detail_title"]
        == "Ergebnis nach Nachfrageszenario"
    )

    assert len(
        presentation["detail_rows"]
    ) == 3