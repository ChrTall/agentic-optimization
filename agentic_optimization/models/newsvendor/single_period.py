"""
Grunddefinition des einperiodigen Newsvendor-Modells.
"""

from dataclasses import dataclass


import pulp

from agentic_optimization.models.forecast_adapter import (
    PreparedDemandForecast,
    prepare_demand_forecast,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.types import ModelInputField

from agentic_optimization.models.types import (
    ModelCoreDefinition,
)


CORE_DEFINITION: ModelCoreDefinition = {
    "core_id": "single_period_newsvendor",
    "name": "Einperiodiges Newsvendor-Modell",
    "minimum_products": 1,
    "required_forecast_fields": [
        {
            "name": "expected_value",
            "label": "Erwartete Nachfrage",
            "type": "number",
            "required": True,
            "help": (
                "Erwartete Nachfrage innerhalb des "
                "betrachteten Prognosezeitraums."
            ),
            "min_value": 0.0,
            "unit": "Einheiten",
        },
        {
            "name": "quantiles",
            "label": "Nachfragequantile",
            "type": "forecast_quantiles",
            "required": True,
            "help": (
                "Quantile der Nachfrageprognose mit "
                "Wahrscheinlichkeitsniveau und Nachfragewert."
            ),
            "unit": "Einheiten",
        },
        {
            "name": "horizon_days",
            "label": "Prognosezeitraum",
            "type": "integer",
            "required": True,
            "help": (
                "Anzahl der Tage, auf die sich die "
                "Nachfrageprognose bezieht."
            ),
            "min_value": 1,
            "step": 1,
            "unit": "Tage",
        },
    ],
    "provided_quantities": {
        "order_quantity": {
            "quantity_id": "order_quantity",
            "name": "Bestellmenge",
            "role": "decision",
            "scope": "per_product",
            "description": (
                "Vom Optimierungsmodell zu bestimmende "
                "Bestellmenge eines Produkts."
            ),
        },
        "sold_quantity": {
            "quantity_id": "sold_quantity",
            "name": "Verkaufte Menge",
            "role": "derived",
            "scope": "per_product_and_forecast_point",
            "description": (
                "Minimum aus Bestellmenge und Nachfrage."
            ),
        },
        "leftover_quantity": {
            "quantity_id": "leftover_quantity",
            "name": "Restbestand",
            "role": "derived",
            "scope": "per_product_and_forecast_point",
            "description": (
                "Positive Differenz zwischen Bestellmenge "
                "und Nachfrage."
            ),
        },
        "shortage_quantity": {
            "quantity_id": "shortage_quantity",
            "name": "Fehlmenge",
            "role": "derived",
            "scope": "per_product_and_forecast_point",
            "description": (
                "Positive Differenz zwischen Nachfrage "
                "und Bestellmenge."
            ),
        },
    },
}

@dataclass
class BuiltSinglePeriodNewsvendorCore:
    """Enthält das PuLP-Modell und die Größen des Newsvendor-Kerns."""

    model: pulp.LpProblem
    forecasts: dict[str, PreparedDemandForecast]
    order_quantities: dict[str, pulp.LpVariable]
    sold_quantities: dict[
        tuple[str, int],
        pulp.LpVariable,
    ]
    leftover_quantities: dict[
        tuple[str, int],
        pulp.LpVariable,
    ]
    shortage_quantities: dict[
        tuple[str, int],
        pulp.LpVariable,
    ]

    def get_quantity(
        self,
        quantity_id: str,
        product_id: str,
        forecast_point_index: int | None = None,
    ) -> pulp.LpVariable:
        """Gibt eine mathematische Größe über ihre fachliche ID zurück."""

        if quantity_id == "order_quantity":
            if forecast_point_index is not None:
                raise ValueError(
                    "Die Bestellmenge ist nicht "
                    "nach Forecast-Punkten indiziert."
                )

            return self.order_quantities[product_id]

        if forecast_point_index is None:
            raise ValueError(
                f"Für die Modellgröße '{quantity_id}' "
                "fehlt der Forecast-Punkt."
            )

        key = (
            product_id,
            forecast_point_index,
        )

        quantity_mappings = {
            "sold_quantity": self.sold_quantities,
            "leftover_quantity": self.leftover_quantities,
            "shortage_quantity": self.shortage_quantities,
        }

        quantity_mapping = quantity_mappings.get(quantity_id)

        if quantity_mapping is None:
            raise KeyError(
                f"Die Modellgröße '{quantity_id}' "
                "wird vom Newsvendor-Kern nicht bereitgestellt."
            )

        return quantity_mapping[key]


def build_single_period_newsvendor_core(
    specification: ModelSpecification,
) -> BuiltSinglePeriodNewsvendorCore:
    """
    Erzeugt Variablen und feste Beziehungen des Newsvendor-Kerns.

    Zielfunktion und optionale Nebenbedingungen werden anschließend
    durch die ausgewählten Bausteine ergänzt.
    """

    objective_sense = specification["objective"]["sense"]

    problem_sense = (
        pulp.LpMaximize
        if objective_sense == "maximize"
        else pulp.LpMinimize
    )

    model = pulp.LpProblem(
        "Modular_Single_Period_Newsvendor",
        problem_sense,
    )

    forecasts: dict[str, PreparedDemandForecast] = {}
    order_quantities: dict[str, pulp.LpVariable] = {}
    sold_quantities: dict[
        tuple[str, int],
        pulp.LpVariable,
    ] = {}
    leftover_quantities: dict[
        tuple[str, int],
        pulp.LpVariable,
    ] = {}
    shortage_quantities: dict[
        tuple[str, int],
        pulp.LpVariable,
    ] = {}

    for product_index, product in enumerate(
        specification["products"]
    ):
        product_id = product["product_id"]

        prepared_forecast = prepare_demand_forecast(
            product["forecast"]
        )
        forecasts[product_id] = prepared_forecast

        order_quantity = pulp.LpVariable(
            f"order_quantity_{product_index}",
            lowBound=0,
        )
        order_quantities[product_id] = order_quantity

        for point_index, demand_point in enumerate(
            prepared_forecast["demand_points"]
        ):
            key = (
                product_id,
                point_index,
            )
            variable_suffix = (
                f"{product_index}_{point_index}"
            )

            sold_quantity = pulp.LpVariable(
                f"sold_quantity_{variable_suffix}",
                lowBound=0,
            )
            leftover_quantity = pulp.LpVariable(
                f"leftover_quantity_{variable_suffix}",
                lowBound=0,
            )
            shortage_quantity = pulp.LpVariable(
                f"shortage_quantity_{variable_suffix}",
                lowBound=0,
            )

            sold_quantities[key] = sold_quantity
            leftover_quantities[key] = leftover_quantity
            shortage_quantities[key] = shortage_quantity

            demand = demand_point["value"]

            model += (
                sold_quantity + leftover_quantity
                == order_quantity,
                f"inventory_balance_{variable_suffix}",
            )

            model += (
                sold_quantity + shortage_quantity
                == demand,
                f"demand_balance_{variable_suffix}",
            )

    return BuiltSinglePeriodNewsvendorCore(
        model=model,
        forecasts=forecasts,
        order_quantities=order_quantities,
        sold_quantities=sold_quantities,
        leftover_quantities=leftover_quantities,
        shortage_quantities=shortage_quantities,
    )