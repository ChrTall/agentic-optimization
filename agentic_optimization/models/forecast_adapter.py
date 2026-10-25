"""
Aufbereitung von Nachfrageprognosen für Optimierungsmodelle.
"""

import math
from statistics import NormalDist
from typing import TypedDict

from agentic_optimization.models.specification import (
    DemandForecast,
    ForecastQuantile,
)
from agentic_optimization.i18n import t


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


class PreparedContinuousReviewForecast(TypedDict):
    """Aufbereiteter Forecast für kontinuierliche Überprüfung."""

    annual_demand: float
    lead_time_demand_mean: float
    lead_time_demand_std: float
    forecast_horizon_days: int
    lead_time_days: int


DEMAND_SCENARIO_COUNT = 100
"""Anzahl gleich wahrscheinlicher Szenarien der diskreten Approximation."""


def _interpolate_quantile(
    levels: list[float],
    values: list[float],
    level: float,
) -> float:
    """Wertet die stückweise lineare Quantilsfunktion aus."""

    if len(levels) == 1:
        return values[0]

    if level <= levels[0]:
        left, right = 0, 1
    elif level >= levels[-1]:
        left, right = len(levels) - 2, len(levels) - 1
    else:
        right = next(
            index
            for index, candidate in enumerate(levels)
            if candidate >= level
        )
        left = right - 1

    slope = (values[right] - values[left]) / (
        levels[right] - levels[left]
    )
    value = values[left] + slope * (level - levels[left])

    return round(max(value, 0.0), 6)


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

    Zwischen zwei angegebenen Quantilen wird die Quantilsfunktion linear
    interpoliert, außerhalb des äußersten Quantils mit der Steigung des
    benachbarten Abschnitts linear fortgesetzt (nie unter null). Die
    Verteilung wird anschließend durch ``DEMAND_SCENARIO_COUNT`` gleich
    wahrscheinliche Nachfrageszenarien dargestellt. Dadurch ist die
    optimale Bestellmenge nicht mehr auf die wenigen angegebenen
    Quantilswerte beschränkt.
    """

    if not isinstance(forecast, dict):
        raise ValueError(
            t("Die Nachfrageprognose muss als Objekt vorliegen.")
        )

    expected_value = forecast.get("expected_value")
    horizon_days = forecast.get("horizon_days")
    quantiles = forecast.get("quantiles")

    if not _is_number(expected_value) or expected_value < 0:
        raise ValueError(
            t("Der Erwartungswert der Nachfrage muss eine endliche, nicht negative Zahl sein.")
        )

    if (
        not isinstance(horizon_days, int)
        or isinstance(horizon_days, bool)
        or horizon_days <= 0
    ):
        raise ValueError(
            t("Der Prognosezeitraum muss eine positive ganze Zahl sein.")
        )

    if not isinstance(quantiles, list) or not quantiles:
        raise ValueError(
            t("Die Nachfragequantile müssen als nicht leere Liste vorliegen.")
        )

    normalized_quantiles: list[ForecastQuantile] = []

    for index, quantile in enumerate(quantiles):
        if not isinstance(quantile, dict):
            raise ValueError(
                t("Das Nachfragequantil an Position {index} muss als Objekt vorliegen.", index=index)
            )

        if "level" not in quantile or "value" not in quantile:
            raise ValueError(
                t("Beim Nachfragequantil an Position {index} fehlen Quantilsniveau oder Nachfragewert.", index=index)
            )

        level = quantile["level"]
        value = quantile["value"]

        if (
            not _is_number(level)
            or not 0 < level < 1
        ):
            raise ValueError(
                t("Quantilsniveaus müssen endliche Zahlen zwischen 0 und 1 sein.")
            )

        if not _is_number(value) or value < 0:
            raise ValueError(
                t("Quantilswerte müssen endliche, nicht negative Zahlen sein.")
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
                t("Jedes Quantilsniveau darf nur einmal vorkommen.")
            )

        if previous_value is not None and value < previous_value:
            raise ValueError(
                t("Quantilswerte müssen mit steigendem Quantilsniveau monoton steigen.")
            )

        previous_level = level
        previous_value = value

    levels = [
        float(quantile["level"])
        for quantile in sorted_quantiles
    ]
    values = [
        float(quantile["value"])
        for quantile in sorted_quantiles
    ]

    # Gleiche Nachfragewerte (z. B. bei identischen Quantilen) werden
    # zu einem Szenario mit entsprechend höherer Wahrscheinlichkeit
    # zusammengefasst.
    counts_by_value: dict[float, int] = {}

    for index in range(DEMAND_SCENARIO_COUNT):
        level = (index + 0.5) / DEMAND_SCENARIO_COUNT
        value = _interpolate_quantile(levels, values, level)
        counts_by_value[value] = counts_by_value.get(value, 0) + 1

    demand_points: list[WeightedDemandPoint] = [
        {
            "value": value,
            "probability": count / DEMAND_SCENARIO_COUNT,
        }
        for value, count in sorted(counts_by_value.items())
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


def prepare_continuous_review_forecast(
    forecast: DemandForecast,
    lead_time_days: int,
) -> PreparedContinuousReviewForecast:
    """
    Leitet Jahresbedarf und Lieferzeitnachfrage aus dem Forecast ab.

    Die Streuung wird unter Annahme einer Normalverteilung aus einem
    unteren und oberen Quantil geschätzt. Beim Übertragen auf einen
    anderen Zeitraum wird eine stationäre, unabhängige Nachfrage mit
    zeitproportionaler Varianz angenommen.
    """

    prepared_forecast = prepare_demand_forecast(
        forecast
    )

    if (
        not isinstance(lead_time_days, int)
        or isinstance(lead_time_days, bool)
        or lead_time_days <= 0
    ):
        raise ValueError(
            t("Die Lieferzeit muss eine positive ganze Zahl sein.")
        )

    quantiles = sorted(
        forecast["quantiles"],
        key=lambda quantile: quantile["level"],
    )

    lower_quantiles = [
        quantile
        for quantile in quantiles
        if quantile["level"] < 0.5
    ]
    upper_quantiles = [
        quantile
        for quantile in quantiles
        if quantile["level"] > 0.5
    ]

    if not lower_quantiles or not upper_quantiles:
        raise ValueError(
            t("Für die kontinuierliche Überprüfung wird mindestens ein Nachfragequantil unterhalb und eines oberhalb von 0,5 benötigt.")
        )

    lower_quantile = lower_quantiles[0]
    upper_quantile = upper_quantiles[-1]

    expected_value = prepared_forecast["expected_value"]
    lower_value = float(lower_quantile["value"])
    upper_value = float(upper_quantile["value"])

    if not lower_value <= expected_value <= upper_value:
        raise ValueError(
            t("Der Erwartungswert der Nachfrage muss zwischen dem verwendeten unteren und oberen Quantil liegen.")
        )

    normal_distribution = NormalDist()

    lower_z = normal_distribution.inv_cdf(
        float(lower_quantile["level"])
    )
    upper_z = normal_distribution.inv_cdf(
        float(upper_quantile["level"])
    )

    forecast_standard_deviation = (
        (upper_value - lower_value)
        / (upper_z - lower_z)
    )

    forecast_horizon_days = prepared_forecast[
        "horizon_days"
    ]
    time_factor = (
        lead_time_days / forecast_horizon_days
    )

    return {
        "annual_demand": (
            expected_value
            * 365.0
            / forecast_horizon_days
        ),
        "lead_time_demand_mean": (
            expected_value * time_factor
        ),
        "lead_time_demand_std": (
            forecast_standard_deviation
            * math.sqrt(time_factor)
        ),
        "forecast_horizon_days": forecast_horizon_days,
        "lead_time_days": lead_time_days,
    }
