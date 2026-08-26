"""
Bereitet das Optimierungsergebnis für die Darstellung in Formular 5 auf.
"""

from typing import TypedDict

from agentic_optimization.models.registry import (
    get_model_family_definition,
    get_model_variant_definition,
)
from agentic_optimization.models.types import (
    ResultMetric,
)
from agentic_optimization.state import GraphState
from agentic_optimization.ui.review_builder import (
    ReviewItem,
    build_review_data,
)


class ResultData(TypedDict):
    """UI-Datenvertrag für die finale Ergebnisansicht."""
    successful: bool
    status_label: str
    status_message: str
    metrics: list[ResultMetric]
    technical_metrics: list[ResultMetric]
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


def build_result_data(
    state: GraphState,
) -> ResultData:
    """
    Erstellt die Daten für die Ergebnisansicht aus dem Graph-State.

    Solverstatus, Fehler, aktive Constraints und technische Details werden
    zentral aufbereitet. Fachliche Kennzahlen und Szenariodetails liefert
    der result_builder der ausgewählten Modellvariante.
    """

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
    technical_metrics: list[ResultMetric] = []
    detail_title: str | None = None
    detail_rows: list[dict[str, str]] = []

    # Fachliche Kennzahlen und Detailzeilen sind variantenspezifisch.
    if successful and variant_definition is not None:
        variant_result = variant_definition[
            "result_builder"
        ](
            optimization_result
        )

        metrics = variant_result["metrics"]
        technical_metrics = variant_result[
            "technical_metrics"
        ]
        detail_title = variant_result[
            "detail_title"
        ]
        detail_rows = variant_result[
            "detail_rows"
        ]

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
            "Für die angegebenen Werte und betrieblichen "
            "Rahmenbedingungen existiert keine zulässige "
            "Lösung."
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

    # Aktive Constraints verwenden dieselben Labels und Einheiten wie im Review.
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
        "technical_metrics": technical_metrics,
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
