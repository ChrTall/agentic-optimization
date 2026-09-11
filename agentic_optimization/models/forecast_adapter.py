"""
Aufbereitung von Nachfrageprognosen für Optimierungsmodelle.
"""

import math
from typing import TypedDict

from agentic_optimization.models.specification import (
    DemandForecast,
    ForecastQuantile,
)


class WeightedDemandPoint(TypedDict):
    """Ein Nachfragewert mit zugeordnetem Wahrscheinlichkeitsgewicht."""

    value: float
    probability: float


class PreparedDemandForecast(TypedDict):
    """Für ein Optimierungsmodell aufbereitete Nachfrageprognose."""

    expected_value: float
    approximated_expected_value: float
    horizon_days: int
    demand_points: list[WeightedDemandPoint]


def _is_number(value: object) -> bool:
    """Prüft auf endliche numerische Werte und schließt bool aus."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        return False

    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def prepare_demand_forecast(
    forecast: DemandForecast,
) -> PreparedDemandForecast:
    """
    Erzeugt eine diskrete Approximation aus Forecast-Quantilen.

    Die Grenzen zwischen zwei Quantilen liegen jeweils in der Mitte
    ihrer Wahrscheinlichkeitsniveaus. Die äußeren Quantile repräsentieren
    zusätzlich die nicht abgedeckten Randbereiche.
    """

    if not isinstance(forecast, dict):
        raise ValueError(
            "Die Nachfrageprognose muss als Objekt vorliegen."
        )

    expected_value = forecast.get("expected_value")
    horizon_days = forecast.get("horizon_days")
    quantiles = forecast.get("quantiles")

    if not _is_number(expected_value) or expected_value < 0:
        raise ValueError(
            "Der Erwartungswert der Nachfrage muss "
            "eine endliche, nicht negative Zahl sein."
        )

    if (
        not isinstance(horizon_days, int)
        or isinstance(horizon_days, bool)
        or horizon_days <= 0
    ):
        raise ValueError(
            "Der Prognosezeitraum muss eine positive "
            "ganze Zahl sein."
        )

    if not isinstance(quantiles, list) or not quantiles:
        raise ValueError(
            "Die Nachfragequantile müssen als "
            "nicht leere Liste vorliegen."
        )

    normalized_quantiles: list[ForecastQuantile] = []

    for index, quantile in enumerate(quantiles):
        if not isinstance(quantile, dict):
            raise ValueError(
                f"Das Nachfragequantil an Position "
                f"{index} muss als Objekt vorliegen."
            )

        if "level" not in quantile or "value" not in quantile:
            raise ValueError(
                f"Beim Nachfragequantil an Position "
                f"{index} fehlen Quantilsniveau "
                "oder Nachfragewert."
            )

        level = quantile["level"]
        value = quantile["value"]

        if (
            not _is_number(level)
            or not 0 < level < 1
        ):
            raise ValueError(
                "Quantilsniveaus müssen endliche Zahlen "
                "zwischen 0 und 1 sein."
            )

        if not _is_number(value) or value < 0:
            raise ValueError(
                "Quantilswerte müssen endliche, "
                "nicht negative Zahlen sein."
            )

        normalized_quantiles.append(
            {
                "level": float(level),
                "value": float(value),
            }
        )

    sorted_quantiles = sorted(
        normalized_quantiles,
        key=lambda quantile: quantile["level"],
    )

    previous_level: float | None = None
    previous_value: float | None = None

    for quantile in sorted_quantiles:
        level = quantile["level"]
        value = quantile["value"]

        if previous_level is not None and level == previous_level:
            raise ValueError(
                "Jedes Quantilsniveau darf nur einmal vorkommen."
            )

        if previous_value is not None and value < previous_value:
            raise ValueError(
                "Quantilswerte müssen mit steigendem "
                "Quantilsniveau monoton steigen."
            )

        previous_level = level
        previous_value = value

    levels = [
        float(quantile["level"])
        for quantile in sorted_quantiles
    ]

    boundaries = [0.0]

    boundaries.extend(
        (left_level + right_level) / 2
        for left_level, right_level in zip(
            levels,
            levels[1:],
        )
    )

    boundaries.append(1.0)

    probabilities_by_value: dict[float, float] = {}

    for index, quantile in enumerate(sorted_quantiles):
        value = float(quantile["value"])
        probability = (
            boundaries[index + 1]
            - boundaries[index]
        )

        probabilities_by_value[value] = (
            probabilities_by_value.get(value, 0.0)
            + probability
        )

    demand_points: list[WeightedDemandPoint] = [
        {
            "value": value,
            "probability": probability,
        }
        for value, probability in probabilities_by_value.items()
    ]

    approximated_expected_value = sum(
        point["value"] * point["probability"]
        for point in demand_points
    )

    return {
        "expected_value": float(expected_value),
        "approximated_expected_value": (
            approximated_expected_value
        ),
        "horizon_days": horizon_days,
        "demand_points": demand_points,
    }