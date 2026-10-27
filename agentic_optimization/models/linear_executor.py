"""
Ausführung modular aufgebauter linearer Optimierungsmodelle.
"""

import pulp

from agentic_optimization.models.compiler import (
    compile_model_specification,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.execution import (
    DecisionValue,
    ExecutionStatus,
    MetricValue,
    OptimizationExecutionResult,
)
from agentic_optimization.i18n import t


_PULP_STATUS_MAPPING: dict[str, ExecutionStatus] = {
    "Optimal": "optimal",
    "Infeasible": "infeasible",
    "Unbounded": "unbounded",
    "Not Solved": "failed",
    "Undefined": "failed",
}


def _extract_decisions(
    specification: ModelSpecification,
    context,
) -> list[DecisionValue]:
    """Liest die Entscheidungsvariablen aus einem gelösten Modell."""

    decisions: list[DecisionValue] = []
    core_definition = CORE_REGISTRY[
        specification["core_id"]
    ]["definition"]

    for quantity_definition in core_definition[
        "provided_quantities"
    ].values():
        if quantity_definition["role"] != "decision":
            continue

        quantity_id = quantity_definition["quantity_id"]

        for product in specification["products"]:
            product_id = product["product_id"]

            if quantity_definition["scope"] == "per_product":
                value = context.get_quantity(
                    quantity_id,
                    product_id,
                ).value()

                if value is not None:
                    decisions.append(
                        {
                            "quantity_id": quantity_id,
                            "product_id": product_id,
                            "value": float(value),
                        }
                    )
                continue

            demand_points = context.forecasts[
                product_id
            ]["demand_points"]

            for point_index in range(len(demand_points)):
                value = context.get_quantity(
                    quantity_id,
                    product_id,
                    point_index,
                ).value()

                if value is not None:
                    decisions.append(
                        {
                            "quantity_id": quantity_id,
                            "product_id": product_id,
                            "forecast_point_index": point_index,
                            "value": float(value),
                        }
                    )

    return decisions


def _extract_metrics(
    specification: ModelSpecification,
    context,
) -> list[MetricValue]:
    """Berechnet Ergebniskennzahlen aus abgeleiteten Modellgrößen."""

    metrics: list[MetricValue] = []
    core_definition = CORE_REGISTRY[
        specification["core_id"]
    ]["definition"]

    for quantity_definition in core_definition[
        "provided_quantities"
    ].values():
        if quantity_definition["role"] != "derived":
            continue

        quantity_id = quantity_definition["quantity_id"]

        for product in specification["products"]:
            product_id = product["product_id"]

            if quantity_definition["scope"] == "per_product":
                value = context.get_quantity(
                    quantity_id,
                    product_id,
                ).value()

                if value is not None:
                    metrics.append(
                        {
                            "metric_id": quantity_id,
                            "product_id": product_id,
                            "value": float(value),
                        }
                    )
                continue

            expected_value = 0.0
            has_complete_values = True

            for point_index, demand_point in enumerate(
                context.forecasts[product_id]["demand_points"]
            ):
                value = context.get_quantity(
                    quantity_id,
                    product_id,
                    point_index,
                ).value()

                if value is None:
                    has_complete_values = False
                    break

                expected_value += (
                    demand_point["probability"]
                    * float(value)
                )

            if has_complete_values:
                metrics.append(
                    {
                        "metric_id": quantity_id,
                        "product_id": product_id,
                        "aggregation": "expected_value",
                        "value": expected_value,
                    }
                )

    return metrics


def execute_linear_model_specification(
    specification: ModelSpecification,
    solver: pulp.LpSolver | None = None,
) -> OptimizationExecutionResult:
    """Kompiliert und löst eine lineare Modellspezifikation."""

    context = compile_model_specification(specification)
    solver_to_use = (
        solver
        if solver is not None
        else pulp.PULP_CBC_CMD(msg=False)
    )

    try:
        context.model.solve(solver_to_use)
    except pulp.PulpSolverError as error:
        return {
            "status": "failed",
            "solution_method": "linear_programming",
            "objective_value": None,
            "decisions": [],
            "metrics": [],
            "technical_representation": str(context.model),
            "errors": [str(error)],
        }

    pulp_status = pulp.LpStatus[context.model.status]
    status = _PULP_STATUS_MAPPING.get(
        pulp_status,
        "failed",
    )

    objective_value: float | None = None
    decisions: list[DecisionValue] = []
    metrics: list[MetricValue] = []

    if status == "optimal":
        raw_objective_value = pulp.value(
            context.model.objective
        )

        if raw_objective_value is not None:
            objective_value = float(raw_objective_value)

        decisions = _extract_decisions(
            specification,
            context,
        )
        metrics = _extract_metrics(
            specification,
            context,
        )

    errors = []

    if status != "optimal":
        errors.append(
            t("Die lineare Optimierung endete mit dem Status '{pulp_status}'.", pulp_status=pulp_status)
        )

    return {
        "status": status,
        "solution_method": "linear_programming",
        "objective_value": objective_value,
        "decisions": decisions,
        "metrics": metrics,
        "technical_representation": str(context.model),
        "errors": errors,
    }