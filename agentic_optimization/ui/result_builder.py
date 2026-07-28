from typing import Any, TypedDict

from agentic_optimization.models.registry import (
    get_model_family_definition,
    get_model_variant_definition,
)
from agentic_optimization.state import GraphState
from agentic_optimization.ui.review_builder import (
    ReviewItem,
    build_review_data,
)


class ResultMetric(TypedDict):
    label: str
    value: str


class ResultData(TypedDict):
    successful: bool
    status_label: str
    status_message: str
    metrics: list[ResultMetric]
    detail_title: str | None
    detail_rows: list[dict[str, str]]
    constraints: list[ReviewItem]
    family_name: str
    variant_name: str
    solver_status: str
    model_representation: str
    errors: list[str]


SOLVER_STATUS_LABELS = {
    "Optimal": "Optimale Lösung gefunden",
    "Infeasible": "Keine zulässige Lösung",
    "Unbounded": "Keine begrenzte optimale Lösung",
    "Not Solved": "Berechnung nicht abgeschlossen",
    "Undefined": "Unbestimmtes Berechnungsergebnis",
}


def _format_number(
    value: float | None,
) -> str:
    if value is None:
        return "–"

    numeric_value = float(value)

    if numeric_value.is_integer():
        return str(int(numeric_value))

    return f"{numeric_value:g}"


def _format_currency(
    value: float | None,
) -> str:
    if value is None:
        return "–"

    formatted = f"{float(value):,.2f}"

    formatted = (
        formatted
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )

    return f"{formatted} €"


def _format_percentage(
    value: float | None,
) -> str:
    if value is None:
        return "–"

    return f"{float(value) * 100:g} %"


def _build_newsvendor_result(
    optimization_result: dict[str, Any],
) -> tuple[
    list[ResultMetric],
    str,
    list[dict[str, str]],
]:
    metrics: list[ResultMetric] = [
        {
            "label": "Empfohlene Bestellmenge",
            "value": (
                f"{_format_number(
                    optimization_result.get(
                        'order_quantity'
                    )
                )} Einheiten"
            ),
        },
        {
            "label": "Erwarteter Gewinn",
            "value": _format_currency(
                optimization_result.get(
                    "expected_profit"
                )
            ),
        },
    ]

    detail_rows = [
        {
            "Nachfrage": _format_number(
                scenario.get("demand")
            ),
            "Wahrscheinlichkeit": _format_percentage(
                scenario.get("probability")
            ),
            "Verkauft": _format_number(
                scenario.get("sold")
            ),
            "Restbestand": _format_number(
                scenario.get("leftover")
            ),
        }
        for scenario in optimization_result.get(
            "scenario_results",
            [],
        )
        if isinstance(scenario, dict)
    ]

    return (
        metrics,
        "Ergebnis nach Nachfrageszenario",
        detail_rows,
    )


def build_result_data(
    state: GraphState,
) -> ResultData:
    result = state.get("result", {})

    optimization_result = (
        state.get("optimization_result")
        or result.get("optimization_result")
        or {}
    )

    optimization_errors = (
        state.get("optimization_errors")
        or result.get("optimization_errors")
        or []
    )

    family_id = (
        state.get("selected_model_family")
        or result.get(
            "selected_model",
            {},
        ).get("family_id")
    )

    variant_id = (
        state.get("selected_model_variant")
        or result.get(
            "selected_model",
            {},
        ).get("variant_id")
    )

    family_definition = get_model_family_definition(
        family_id
    )

    variant_definition = get_model_variant_definition(
        family_id,
        variant_id,
    )

    family_name = (
        family_definition["family_name"]
        if family_definition is not None
        else str(family_id or "–")
    )

    variant_name = (
        variant_definition["variant_name"]
        if variant_definition is not None
        else str(variant_id or "–")
    )

    solver_status = str(
        optimization_result.get(
            "solver_status",
            "Not Solved",
        )
    )

    status_label = SOLVER_STATUS_LABELS.get(
        solver_status,
        f"Solverstatus: {solver_status}",
    )

    successful = (
        not optimization_errors
        and solver_status == "Optimal"
    )

    metrics: list[ResultMetric] = []
    detail_title: str | None = None
    detail_rows: list[dict[str, str]] = []

    if (
        successful
        and family_id == "newsvendor"
        and variant_id
        == "single_product_single_period"
    ):
        (
            metrics,
            detail_title,
            detail_rows,
        ) = _build_newsvendor_result(
            optimization_result
        )

    if optimization_errors:
        status_label = "Berechnung fehlgeschlagen"
        status_message = (
            "Die Optimierung konnte nicht erfolgreich "
            "abgeschlossen werden."
        )

    elif solver_status == "Optimal":
        status_message = (
            "Für die geprüften Angaben wurde eine optimale "
            "Lösung gefunden."
        )

    elif solver_status == "Infeasible":
        status_message = (
            "Für die angegebenen Werte und Vorgaben existiert "
            "keine zulässige Lösung."
        )

    elif solver_status == "Unbounded":
        status_message = (
            "Das Optimierungsmodell besitzt keine begrenzte "
            "optimale Lösung."
        )

    else:
        status_message = (
            "Die Berechnung wurde nicht mit einer optimalen "
            "Lösung abgeschlossen."
        )

    constraints: list[ReviewItem] = []

    if variant_definition is not None:
        review_data = build_review_data(
            state,
            variant_definition,
        )
        constraints = review_data["constraints"]

    return {
        "successful": successful,
        "status_label": status_label,
        "status_message": status_message,
        "metrics": metrics,
        "detail_title": detail_title,
        "detail_rows": detail_rows,
        "constraints": constraints,
        "family_name": family_name,
        "variant_name": variant_name,
        "solver_status": solver_status,
        "model_representation": str(
            optimization_result.get(
                "model_representation",
                "",
            )
        ),
        "errors": [
            str(error)
            for error in optimization_errors
        ],
    }