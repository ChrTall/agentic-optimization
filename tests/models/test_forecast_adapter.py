import pytest

from agentic_optimization.models.forecast_adapter import (
    prepare_demand_forecast,
)
from agentic_optimization.models.specification import (
    DemandForecast,
)


def create_forecast() -> DemandForecast:
    return {
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
    }


def test_prepares_weighted_demand_points() -> None:
    result = prepare_demand_forecast(create_forecast())

    assert result["expected_value"] == 100.0
    assert result["horizon_days"] == 30

    assert result["demand_points"] == [
        {
            "value": 70.0,
            "probability": pytest.approx(0.3),
        },
        {
            "value": 100.0,
            "probability": pytest.approx(0.4),
        },
        {
            "value": 140.0,
            "probability": pytest.approx(0.3),
        },
    ]

    assert result["approximated_expected_value"] == pytest.approx(
        103.0
    )


def test_sorts_quantiles_by_level() -> None:
    forecast = create_forecast()
    forecast["quantiles"].reverse()

    result = prepare_demand_forecast(forecast)

    assert [
        point["value"]
        for point in result["demand_points"]
    ] == [
        70.0,
        100.0,
        140.0,
    ]


def test_single_quantile_receives_full_probability() -> None:
    forecast = create_forecast()
    forecast["quantiles"] = [
        {
            "level": 0.5,
            "value": 100.0,
        }
    ]

    result = prepare_demand_forecast(forecast)

    assert result["demand_points"] == [
        {
            "value": 100.0,
            "probability": 1.0,
        }
    ]


def test_rejects_duplicate_quantile_levels() -> None:
    forecast = create_forecast()
    forecast["quantiles"].append(
        {
            "level": 0.5,
            "value": 110.0,
        }
    )

    with pytest.raises(
        ValueError,
        match="Quantilsniveau darf nur einmal",
    ):
        prepare_demand_forecast(forecast)


def test_rejects_decreasing_quantile_values() -> None:
    forecast = create_forecast()
    forecast["quantiles"][2]["value"] = 90.0

    with pytest.raises(
        ValueError,
        match="monoton steigen",
    ):
        prepare_demand_forecast(forecast)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_non_finite_expected_value(
    invalid_value: float,
) -> None:
    forecast = create_forecast()
    forecast["expected_value"] = invalid_value

    with pytest.raises(
        ValueError,
        match="Erwartungswert",
    ):
        prepare_demand_forecast(forecast)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_non_finite_quantile_value(
    invalid_value: float,
) -> None:
    forecast = create_forecast()
    forecast["quantiles"][0]["value"] = invalid_value

    with pytest.raises(
        ValueError,
        match="Quantilswerte",
    ):
        prepare_demand_forecast(forecast)


@pytest.mark.parametrize(
    "missing_field",
    [
        "expected_value",
        "horizon_days",
        "quantiles",
    ],
)
def test_rejects_missing_forecast_field(
    missing_field: str,
) -> None:
    forecast = create_forecast()
    del forecast[missing_field]

    with pytest.raises(ValueError):
        prepare_demand_forecast(forecast)


def test_rejects_non_object_quantile() -> None:
    forecast = create_forecast()
    forecast["quantiles"] = [100.0]

    with pytest.raises(
        ValueError,
        match="muss als Objekt vorliegen",
    ):
        prepare_demand_forecast(forecast)


def test_rejects_quantile_with_missing_value() -> None:
    forecast = create_forecast()
    forecast["quantiles"] = [
        {
            "level": 0.5,
        }
    ]

    with pytest.raises(
        ValueError,
        match="fehlen Quantilsniveau oder Nachfragewert",
    ):
        prepare_demand_forecast(forecast)


def test_rejects_number_too_large_for_float() -> None:
    forecast = create_forecast()
    forecast["expected_value"] = 10**1000

    with pytest.raises(
        ValueError,
        match="Erwartungswert",
    ):
        prepare_demand_forecast(forecast)