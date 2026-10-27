"""
Bereitet modulare Optimierungsergebnisse für die UI auf.
"""
from copy import deepcopy
from typing import TypedDict

from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)
from agentic_optimization.models.execution import (
    OptimizationExecutionResult,
)
from agentic_optimization.models.specification import (
    ExcludedRequirement,
    ModelSpecification,
)
from agentic_optimization.i18n import t


STATUS_LABELS = {
    "optimal": "Optimale Lösung gefunden",
    "feasible": "Zulässige Lösung gefunden",
    "infeasible": "Keine zulässige Lösung gefunden",
    "unbounded": "Optimierungsmodell ist unbeschränkt",
    "iteration_limit": "Berechnungslimit erreicht",
    "failed": "Berechnung fehlgeschlagen",
}

SOLUTION_METHOD_LABELS = {
    "linear_programming": "Lineare Optimierung",
    "analytical": "Analytische Berechnung",
    "numerical": "Numerische Optimierung",
}

METRIC_LABELS = {
    "annual_demand": "Jahresbedarf",
    "unconstrained_order_quantity": (
        "Unbeschränkte Bestellmenge"
    ),
    "orders_per_year": "Bestellungen pro Jahr",
    "average_inventory": "Durchschnittlicher Lagerbestand",
    "cycle_length_days": "Zyklusdauer in Tagen",
    "annual_ordering_cost": "Jährliche Bestellkosten",
    "annual_holding_cost": "Jährliche Lagerhaltungskosten",
    "annual_shortage_cost": "Jährliche Fehlmengenkosten",
    "annual_relevant_cost": "Jährliche relevante Gesamtkosten",
    "lead_time_demand_mean": (
        "Erwartete Nachfrage während der Lieferzeit"
    ),
    "lead_time_demand_std": (
        "Standardabweichung der Lieferzeitnachfrage"
    ),
    "safety_stock": "Sicherheitsbestand",
    "expected_shortage_per_cycle": (
        "Erwartete Fehlmenge pro Bestellzyklus"
    ),
    "annual_shortage_quantity": "Jährliche Fehlmenge",
    "forecast_expected_demand": "Angegebene erwartete Nachfrage",
    "model_expected_demand": (
        "Im Modell verwendete erwartete Nachfrage (aus den Quantilen)"
    ),
}


OBJECTIVE_LABELS_BY_CORE = {
    "single_period_newsvendor": {
        "maximize": "Erwarteter Gewinn",
        "minimize": "Erwartete Kosten",
    },
    "eoq": {
        "minimize": "Jährliche Bestell- und Lagerkosten",
    },
    "continuous_review_q_r": {
        "minimize": "Erwartete jährliche Gesamtkosten",
    },
}


def build_objective_label(specification: dict) -> str:
    """Verständliche Bezeichnung des optimierten Werts.

    Statt des Fachbegriffs „Zielfunktionswert“ wird benannt, was der
    Wert für den Nutzer bedeutet (z. B. „Erwarteter Gewinn“).
    """

    sense = (specification.get("objective") or {}).get("sense", "minimize")
    labels = OBJECTIVE_LABELS_BY_CORE.get(specification.get("core_id"), {})

    return t(labels.get(
        sense,
        "Erwarteter Gewinn" if sense == "maximize" else "Erwartete Kosten",
    ))


class ModularResultValue(TypedDict):
    """Ein benannter Ergebniswert für die Benutzeroberfläche."""

    value_id: str
    label: str
    value: int | float
    product_id: str | None
    aggregation: str | None


class ModularResultData(TypedDict):
    """UI-unabhängige Darstellung eines modularen Ergebnisses."""

    core_id: str
    core_name: str
    status: str
    status_label: str
    successful: bool
    solution_method: str
    solution_method_label: str
    objective_value: float | None
    objective_label: str
    decisions: list[ModularResultValue]
    metrics: list[ModularResultValue]
    excluded_requirements: list[ExcludedRequirement]
    technical_representation: str | None
    errors: list[str]
    warnings: list[str]


def build_default_result_explanation(
    result_data: ModularResultData,
) -> str:
    """Erklärt ein Ergebnis ohne Sprachmodell."""

    if not result_data["successful"]:
        return (
            t("Für dieses Modell konnte keine erfolgreiche Empfehlung berechnet werden. Prüfen Sie den Status und die angezeigten Fehlermeldungen.")
        )

    decisions = result_data["decisions"]

    if not decisions:
        decision_text = (
            t("Es wurde keine konkrete Entscheidungsgröße ausgegeben.")
        )
    elif len(decisions) == 1:
        decision = decisions[0]
        product_text = (
            t(" für {product_id}", product_id=decision['product_id'])
            if decision["product_id"]
            else ""
        )
        decision_text = (
            t("Die zentrale Empfehlung lautet{product_text}: {label} = {value5:g}.", product_text=product_text, label=decision['label'], value5=float(decision['value']))
        )
    else:
        product_ids = {
            decision["product_id"]
            for decision in decisions
            if decision["product_id"] is not None
        }

        if len(product_ids) > 1:
            decision_text = (
                t("Die Berechnung liefert für {value1} Produkte jeweils eine eigene Empfehlung.", value1=len(product_ids))
            )
        else:
            decision_text = (
                t("Die Berechnung liefert mehrere aufeinander abgestimmte Entscheidungswerte.")
            )

    explanation = (
        t("{status_label}. {decision_text} Die weiteren Kennzahlen zeigen die betrieblichen Auswirkungen dieser Empfehlung.", status_label=result_data['status_label'], decision_text=decision_text)
    )

    if result_data["excluded_requirements"]:
        explanation += (
            t(" Die ausgewiesenen, nicht berücksichtigten Anforderungen sind nicht Bestandteil der Empfehlung.")
        )

    return explanation


def build_modular_result_data(
    specification: ModelSpecification,
    result: OptimizationExecutionResult,
) -> ModularResultData:
    """Verknüpft ein Rohergebnis mit den Registry-Metadaten."""

    core_id = specification["core_id"]
    registered_core = CORE_REGISTRY.get(core_id)

    if registered_core is None:
        raise ValueError(
            t("Das Grundmodell '{core_id}' ist nicht registriert.", core_id=core_id)
        )

    core_definition = registered_core["definition"]
    quantity_definitions = core_definition[
        "provided_quantities"
    ]

    decisions: list[ModularResultValue] = []

    for decision in result["decisions"]:
        quantity_id = decision["quantity_id"]
        quantity_definition = quantity_definitions.get(
            quantity_id
        )

        decisions.append(
            {
                "value_id": quantity_id,
                "label": (
                    t(quantity_definition["name"])
                    if quantity_definition is not None
                    else quantity_id
                ),
                "value": decision["value"],
                "product_id": decision.get("product_id"),
                "aggregation": None,
            }
        )

    metrics: list[ModularResultValue] = []

    for metric in result["metrics"]:
        metric_id = metric["metric_id"]
        quantity_definition = quantity_definitions.get(
            metric_id
        )

        label = t(METRIC_LABELS.get(metric_id))

        if label is None and quantity_definition is not None:
            label = t(quantity_definition["name"])

        metrics.append(
            {
                "value_id": metric_id,
                "label": label or metric_id,
                "value": metric["value"],
                "product_id": metric.get("product_id"),
                "aggregation": metric.get("aggregation"),
            }
        )

    status = result["status"]
    solution_method = result["solution_method"]

    return {
        "core_id": core_id,
        "core_name": t(core_definition["name"]),
        "status": status,
        "status_label": t(STATUS_LABELS.get(
            status,
            status,
        )),
        "successful": (
            status in {"optimal", "feasible"}
            and not result["errors"]
        ),
        "solution_method": solution_method,
        "solution_method_label": (
            t(SOLUTION_METHOD_LABELS.get(
                solution_method,
                solution_method,
            ))
        ),
        "objective_value": result["objective_value"],
        "objective_label": build_objective_label(specification),
        "decisions": decisions,
        "metrics": metrics,
        "excluded_requirements": deepcopy(
            specification.get(
                "excluded_requirements",
                [],
            )
        ),
        "technical_representation": result[
            "technical_representation"
        ],
        "errors": list(result["errors"]),
        "warnings": list(result.get("warnings", [])),
    }