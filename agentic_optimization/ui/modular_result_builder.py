"""
Bereitet modulare Optimierungsergebnisse für die UI auf.
"""
from copy import deepcopy
from typing import TypedDict
from agentic_optimization.models.component_registry import OBJECTIVE_COMPONENTS
from agentic_optimization.models.product_names import product_name

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


class ModularResultValue(TypedDict):
    """Ein benannter Ergebniswert für die Benutzeroberfläche."""

    value_id: str
    label: str
    value: int | float
    product_id: str | None
    product_name: str
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
    objective_help: str
    decisions: list[ModularResultValue]
    metrics: list[ModularResultValue]
    excluded_requirements: list[ExcludedRequirement]
    technical_representation: str | None
    errors: list[str]


def build_default_result_explanation(
    result_data: ModularResultData,
) -> str:
    """Erklärt ein Ergebnis ohne Sprachmodell."""

    if not result_data["successful"]:
        return (
            "Für dieses Modell konnte keine erfolgreiche "
            "Empfehlung berechnet werden. Prüfen Sie den "
            "Status und die angezeigten Fehlermeldungen."
        )

    decisions = result_data["decisions"]

    if not decisions:
        decision_text = (
            "Es wurde keine konkrete Entscheidungsgröße "
            "ausgegeben."
        )
    elif len(decisions) == 1:
        decision = decisions[0]
        product_text = (
            f" für {decision.get('product_name', decision['product_id'])}"
            if decision["product_id"]
            else ""
        )
        decision_text = (
            f"Die zentrale Empfehlung lautet{product_text}: "
            f"{decision['label']} = "
            f"{float(decision['value']):g}."
        )
    else:
        product_ids = {
            decision["product_id"]
            for decision in decisions
            if decision["product_id"] is not None
        }

        if len(product_ids) > 1:
            decision_text = (
                "Die Berechnung liefert für "
                f"{len(product_ids)} Produkte jeweils "
                "eine eigene Empfehlung."
            )
        else:
            decision_text = (
                "Die Berechnung liefert mehrere "
                "aufeinander abgestimmte "
                "Entscheidungswerte."
            )

    explanation = (
        f"{result_data['status_label']}. "
        f"{decision_text} "
        "Die weiteren Kennzahlen zeigen die "
        "betrieblichen Auswirkungen dieser Empfehlung."
    )

    if result_data["excluded_requirements"]:
        explanation += (
            " Die ausgewiesenen, nicht berücksichtigten "
            "Anforderungen sind nicht Bestandteil der "
            "Empfehlung."
        )

    return explanation


def objective_presentation(specification: ModelSpecification) -> tuple[str, str]:
    """Verlässliche Beschriftung aus Grundmodell, Zielrichtung und Zielbausteinen."""
    ids = set(specification["objective"]["component_ids"])
    sense = specification["objective"]["sense"]
    core = specification["core_id"]
    label = "Modellierter Zielwert"
    if core == "eoq" and sense == "minimize" and ids == {"annual_ordering_cost", "annual_holding_cost"}:
        label = "Jährliche Bestell- und Lagerhaltungskosten (€)"
    elif core == "continuous_review_q_r" and sense == "minimize":
        if ids == {"annual_ordering_cost", "annual_holding_cost", "annual_shortage_cost"}:
            label = "Erwartete jährliche Bestell-, Lagerhaltungs- und Fehlmengenkosten (€)"
        elif ids == {"annual_ordering_cost", "annual_holding_cost"}:
            label = "Erwartete jährliche Bestell- und Lagerhaltungskosten (€)"
    elif core == "single_period_newsvendor":
        allowed = {"sales_revenue", "salvage_revenue", "purchase_cost", "disposal_cost", "shortage_cost"}
        if ids and ids <= allowed:
            if sense == "maximize" and {"sales_revenue", "purchase_cost"} <= ids:
                label = "Erwarteter Gewinn im Planungszeitraum (€)"
            elif sense == "maximize" and ids == {"sales_revenue"}:
                label = "Erwarteter Verkaufserlös im Planungszeitraum (€)"
            elif sense == "maximize" and ids <= {"sales_revenue", "salvage_revenue"}:
                label = "Erwartete Erlöse im Planungszeitraum (€)"
            elif sense == "minimize" and ids <= {"purchase_cost", "disposal_cost", "shortage_cost"}:
                label = "Erwartete Kosten im Planungszeitraum (€)"
    components = []
    for component_id in specification["objective"]["component_ids"]:
        definition = OBJECTIVE_COMPONENTS.get(component_id)
        if definition:
            positive = definition["effect"] == ("benefit" if sense == "maximize" else "cost")
            components.append(("+ " if positive else "− ") + definition["name"])
    help_text = "Berücksichtigte Bestandteile: " + "; ".join(components) + "."
    help_text += " Weitere, nicht im Modell erfasste Kosten und Erlöse sind darin nicht enthalten."
    return label, help_text


def build_modular_result_data(
    specification: ModelSpecification,
    result: OptimizationExecutionResult,
) -> ModularResultData:
    """Verknüpft ein Rohergebnis mit den Registry-Metadaten."""

    core_id = specification["core_id"]
    registered_core = CORE_REGISTRY.get(core_id)

    if registered_core is None:
        raise ValueError(
            f"Das Grundmodell '{core_id}' ist nicht registriert."
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
                    quantity_definition["name"]
                    if quantity_definition is not None
                    else quantity_id
                ),
                "value": decision["value"],
                "product_id": decision.get("product_id"),
                "product_name": product_name(specification, decision.get("product_id")),
                "aggregation": None,
            }
        )

    metrics: list[ModularResultValue] = []

    for metric in result["metrics"]:
        metric_id = metric["metric_id"]
        quantity_definition = quantity_definitions.get(
            metric_id
        )

        label = METRIC_LABELS.get(metric_id)

        if label is None and quantity_definition is not None:
            label = quantity_definition["name"]

        metrics.append(
            {
                "value_id": metric_id,
                "label": label or metric_id,
                "value": metric["value"],
                "product_id": metric.get("product_id"),
                "product_name": product_name(specification, metric.get("product_id")),
                "aggregation": metric.get("aggregation"),
            }
        )

    status = result["status"]
    solution_method = result["solution_method"]
    objective_label, objective_help = objective_presentation(specification)

    return {
        "core_id": core_id,
        "core_name": core_definition["name"],
        "status": status,
        "status_label": STATUS_LABELS.get(
            status,
            status,
        ),
        "successful": (
            status in {"optimal", "feasible"}
            and not result["errors"]
        ),
        "solution_method": solution_method,
        "solution_method_label": (
            SOLUTION_METHOD_LABELS.get(
                solution_method,
                solution_method,
            )
        ),
        "objective_value": result["objective_value"],
        "objective_label": objective_label,
        "objective_help": objective_help,
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
    }
