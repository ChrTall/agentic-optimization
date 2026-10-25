"""
Registry für wiederverwendbare Optimierungsbausteine.
"""

from collections.abc import Callable
from typing import Literal, NotRequired, Protocol, TypedDict

import pulp

from agentic_optimization.models.forecast_adapter import (
    PreparedDemandForecast,
)
from agentic_optimization.models.specification import (
    ConstraintSpecification,
    ModelSpecification,
)
from agentic_optimization.models.types import (
    ConstraintParameterDefinition,
    ModelInputField,
)
from agentic_optimization.i18n import t


ComponentScope = Literal[
    "per_product",
    "global",
]

ObjectiveEffect = Literal[
    "benefit",
    "cost",
]


class ModelBuildContext(Protocol):
    """Schnittstelle zwischen Grundmodell und Bausteinen."""

    model: pulp.LpProblem
    forecasts: dict[str, PreparedDemandForecast]

    def get_quantity(
        self,
        quantity_id: str,
        product_id: str,
        forecast_point_index: int | None = None,
    ) -> pulp.LpVariable:
        """Gibt eine mathematische Größe des Grundmodells zurück."""


ObjectiveComponentBuilder = Callable[
    [
        ModelSpecification,
        ModelBuildContext,
    ],
    pulp.LpAffineExpression,
]

ConstraintComponentBuilder = Callable[
    [
        ModelSpecification,
        ModelBuildContext,
        ConstraintSpecification,
        int,
    ],
    None,
]


class ObjectiveComponentDefinition(TypedDict):
    """Beschreibt einen registrierten Zielfunktionsbaustein."""

    component_id: str
    name: str
    effect: ObjectiveEffect
    compatible_core_ids: list[str]
    required_model_quantities: list[str]
    required_product_facts: list[ModelInputField]
    required_global_facts: list[ModelInputField]
    linear_builder: NotRequired[ObjectiveComponentBuilder]
    selection_guidance: NotRequired[str]


class ConstraintComponentDefinition(TypedDict):
    """Beschreibt einen registrierten Nebenbedingungsbaustein."""

    component_id: str
    name: str
    scope: ComponentScope
    compatible_core_ids: list[str]
    required_model_quantities: list[str]
    required_product_facts: list[ModelInputField]
    required_global_facts: list[ModelInputField]
    parameters: list[ConstraintParameterDefinition]
    linear_builder: NotRequired[ConstraintComponentBuilder]
    selection_guidance: NotRequired[str]
    requires_explicit_request: NotRequired[bool]


def build_sales_revenue(
    specification: ModelSpecification,
    context: ModelBuildContext,
) -> pulp.LpAffineExpression:
    """Erzeugt den erwarteten Verkaufserlös."""

    terms = []

    for product in specification["products"]:
        product_id = product["product_id"]
        selling_price = float(
            product["facts"]["selling_price"]
        )

        demand_points = context.forecasts[
            product_id
        ]["demand_points"]

        for point_index, demand_point in enumerate(
            demand_points
        ):
            sold_quantity = context.get_quantity(
                "sold_quantity",
                product_id,
                point_index,
            )

            terms.append(
                demand_point["probability"]
                * selling_price
                * sold_quantity
            )

    return pulp.lpSum(terms)


def build_salvage_revenue(
    specification: ModelSpecification,
    context: ModelBuildContext,
) -> pulp.LpAffineExpression:
    """Erzeugt den erwarteten Restwerterlös."""

    terms = []

    for product in specification["products"]:
        product_id = product["product_id"]
        salvage_value = float(
            product["facts"]["salvage_value"]
        )

        demand_points = context.forecasts[
            product_id
        ]["demand_points"]

        for point_index, demand_point in enumerate(
            demand_points
        ):
            leftover_quantity = context.get_quantity(
                "leftover_quantity",
                product_id,
                point_index,
            )

            terms.append(
                demand_point["probability"]
                * salvage_value
                * leftover_quantity
            )

    return pulp.lpSum(terms)


def build_disposal_cost(
    specification: ModelSpecification,
    context: ModelBuildContext,
) -> pulp.LpAffineExpression:
    """Erzeugt die erwarteten Entsorgungskosten."""

    terms = []

    for product in specification["products"]:
        product_id = product["product_id"]
        disposal_cost = float(
            product["facts"]["disposal_cost"]
        )

        demand_points = context.forecasts[
            product_id
        ]["demand_points"]

        for point_index, demand_point in enumerate(
            demand_points
        ):
            leftover_quantity = context.get_quantity(
                "leftover_quantity",
                product_id,
                point_index,
            )

            terms.append(
                demand_point["probability"]
                * disposal_cost
                * leftover_quantity
            )

    return pulp.lpSum(terms)


def build_purchase_cost(
    specification: ModelSpecification,
    context: ModelBuildContext,
) -> pulp.LpAffineExpression:
    """Erzeugt die Beschaffungskosten aller Produkte."""

    return pulp.lpSum(
        float(product["facts"]["purchase_cost"])
        * context.get_quantity(
            "order_quantity",
            product["product_id"],
        )
        for product in specification["products"]
    )


def build_shortage_cost(
    specification: ModelSpecification,
    context: ModelBuildContext,
) -> pulp.LpAffineExpression:
    """Erzeugt die erwarteten Fehlmengenkosten."""

    terms = []

    for product in specification["products"]:
        product_id = product["product_id"]
        shortage_cost = float(
            product["facts"]["shortage_cost"]
        )

        demand_points = context.forecasts[
            product_id
        ]["demand_points"]

        for point_index, demand_point in enumerate(
            demand_points
        ):
            shortage_quantity = context.get_quantity(
                "shortage_quantity",
                product_id,
                point_index,
            )

            terms.append(
                demand_point["probability"]
                * shortage_cost
                * shortage_quantity
            )

    return pulp.lpSum(terms)


def build_minimum_order_quantity(
    specification: ModelSpecification,
    context: ModelBuildContext,
    constraint: ConstraintSpecification,
    constraint_index: int,
) -> None:
    """Fügt die Mindestbestellmenge eines Produkts hinzu."""

    del specification

    product_id = constraint.get("product_id")

    if not product_id:
        raise ValueError(
            t("Die Mindestbestellmenge muss einem Produkt zugeordnet sein.")
        )

    quantity = float(
        constraint["parameters"]["quantity"]
    )

    order_quantity = context.get_quantity(
        "order_quantity",
        product_id,
    )

    context.model += (
        order_quantity >= quantity,
        f"minimum_order_quantity_{constraint_index}",
    )


def build_maximum_order_quantity(
    specification: ModelSpecification,
    context: ModelBuildContext,
    constraint: ConstraintSpecification,
    constraint_index: int,
) -> None:
    """Fügt die Höchstbestellmenge eines Produkts hinzu."""

    del specification

    product_id = constraint.get("product_id")

    if not product_id:
        raise ValueError(
            t("Die Höchstbestellmenge muss einem Produkt zugeordnet sein.")
        )

    quantity = float(
        constraint["parameters"]["quantity"]
    )

    order_quantity = context.get_quantity(
        "order_quantity",
        product_id,
    )

    context.model += (
        order_quantity <= quantity,
        f"maximum_order_quantity_{constraint_index}",
    )


def build_order_multiple(
    specification: ModelSpecification,
    context: ModelBuildContext,
    constraint: ConstraintSpecification,
    constraint_index: int,
) -> None:
    """Beschränkt die Bestellmenge auf ein Vielfaches."""

    del specification

    product_id = constraint.get("product_id")

    if not product_id:
        raise ValueError(
            t("Das Bestellvielfache muss einem Produkt zugeordnet sein.")
        )

    multiple = int(
        constraint["parameters"]["multiple"]
    )

    # Produkt-IDs sind frei wählbare Namen (z. B. "Kühlschrank"); im
    # Variablennamen für den Solver wird daher nur der Index verwendet.
    package_count = context.model.add_variable(
        f"order_package_count_{constraint_index}",
        lowBound=0,
        cat=pulp.LpInteger,
    )

    order_quantity = context.get_quantity(
        "order_quantity",
        product_id,
    )

    context.model += (
        order_quantity == multiple * package_count,
        f"order_multiple_{constraint_index}",
    )


def build_minimum_cycle_service_level(
    specification: ModelSpecification,
    context: ModelBuildContext,
    constraint: ConstraintSpecification,
    constraint_index: int,
) -> None:
    """Sichert eine Mindestwahrscheinlichkeit ohne Fehlbestand."""

    product_id = constraint.get("product_id")

    if not product_id:
        raise ValueError(
            t("Das Mindest-Zyklusservicelevel muss einem Produkt zugeordnet sein.")
        )

    service_level = (
        float(
            constraint["parameters"][
                "service_level_percent"
            ]
        )
        / 100.0
    )

    product = next(
        (
            product
            for product in specification["products"]
            if product["product_id"] == product_id
        ),
        None,
    )

    if product is None:
        raise ValueError(
            t("Das Produkt '{product_id}' ist nicht vorhanden.", product_id=product_id)
        )

    quantiles = product.get(
        "forecast",
        {},
    ).get(
        "quantiles",
        [],
    )

    eligible_quantiles = [
        quantile
        for quantile in quantiles
        if float(quantile["level"]) >= service_level
    ]

    if not eligible_quantiles:
        raise ValueError(
            t("Die Nachfrageprognose enthält kein Quantil auf oder oberhalb des geforderten Servicelevels.")
        )

    demand_quantile = min(
        eligible_quantiles,
        key=lambda quantile: float(quantile["level"]),
    )

    order_quantity = context.get_quantity(
        "order_quantity",
        product_id,
    )

    context.model += (
        order_quantity >= float(demand_quantile["value"]),
        f"minimum_cycle_service_level_{constraint_index}",
    )


def build_budget_limit(
    specification: ModelSpecification,
    context: ModelBuildContext,
    constraint: ConstraintSpecification,
    constraint_index: int,
) -> None:
    """Begrenzt die gesamten Beschaffungsausgaben."""

    budget = float(
        constraint["parameters"]["budget"]
    )

    total_purchase_cost = pulp.lpSum(
        float(product["facts"]["purchase_cost"])
        * context.get_quantity(
            "order_quantity",
            product["product_id"],
        )
        for product in specification["products"]
    )

    context.model += (
        total_purchase_cost <= budget,
        f"budget_limit_{constraint_index}",
    )


def build_storage_capacity(
    specification: ModelSpecification,
    context: ModelBuildContext,
    constraint: ConstraintSpecification,
    constraint_index: int,
) -> None:
    """Begrenztlich die insgesamt benötigte Lagerfläche."""

    capacity = float(
        constraint["parameters"]["capacity"]
    )

    total_storage_requirement = pulp.lpSum(
        float(
            product["facts"]["storage_space_per_unit"]
        )
        * context.get_quantity(
            "order_quantity",
            product["product_id"],
        )
        for product in specification["products"]
    )

    context.model += (
        total_storage_requirement <= capacity,
        f"storage_capacity_{constraint_index}",
    )


OBJECTIVE_COMPONENTS: dict[
    str,
    ObjectiveComponentDefinition,
] = {
    "sales_revenue": {
        "component_id": "sales_revenue",
        "name": "Verkaufserlös",
        "effect": "benefit",
        "compatible_core_ids": [
            "single_period_newsvendor",
        ],
        "required_model_quantities": [
            "sold_quantity",
        ],
        "required_product_facts": [
            {
                "name": "selling_price",
                "label": "Verkaufspreis pro Einheit",
                "type": "number",
                "required": True,
                "help": "Erlös aus dem Verkauf einer Einheit.",
                "min_value": 0.0,
                "unit": "€ pro Einheit",
            }
        ],
        "required_global_facts": [],
        "linear_builder": build_sales_revenue,
    },
    "salvage_revenue": {
        "component_id": "salvage_revenue",
        "name": "Restwerterlös",
        "effect": "benefit",
        "compatible_core_ids": [
            "single_period_newsvendor",
        ],
        "required_model_quantities": [
            "leftover_quantity",
        ],
        "required_product_facts": [
            {
                "name": "salvage_value",
                "label": "Restwert pro unverkaufter Einheit",
                "type": "number",
                "required": True,
                "help": (
                    "Wert einer nach der Periode "
                    "unverkauften Einheit."
                ),
                "min_value": 0.0,
                "unit": "€ pro Einheit",
            }
        ],
        "required_global_facts": [],
        "linear_builder": build_salvage_revenue,
    },
    "disposal_cost": {
        "component_id": "disposal_cost",
        "name": "Entsorgungskosten",
        "effect": "cost",
        "compatible_core_ids": [
            "single_period_newsvendor",
        ],
        "required_model_quantities": [
            "leftover_quantity",
        ],
        "required_product_facts": [
            {
                "name": "disposal_cost",
                "label": "Entsorgungskosten pro unverkaufter Einheit",
                "type": "number",
                "required": True,
                "help": (
                    "Kosten für die Entsorgung einer "
                    "nach der Periode unverkauften Einheit."
                ),
                "min_value": 0.0,
                "unit": "€ pro Einheit",
            }
        ],
        "required_global_facts": [],
        "linear_builder": build_disposal_cost,
    },
    "purchase_cost": {
        "component_id": "purchase_cost",
        "name": "Einkaufskosten",
        "effect": "cost",
        "compatible_core_ids": [
            "single_period_newsvendor",
        ],
        "required_model_quantities": [
            "order_quantity",
        ],
        "required_product_facts": [
            {
                "name": "purchase_cost",
                "label": "Einkaufskosten pro Einheit",
                "type": "number",
                "required": True,
                "help": "Kosten für die Beschaffung einer Einheit.",
                "min_value": 0.0,
                "unit": "€ pro Einheit",
            }
        ],
        "required_global_facts": [],
        "linear_builder": build_purchase_cost,
    },
    "shortage_cost": {
        "component_id": "shortage_cost",
        "name": "Fehlmengenkosten",
        "effect": "cost",
        "selection_guidance": (
            "Wähle diesen Baustein nur, wenn ausdrücklich "
            "zusätzliche Kosten oder eine Vertragsstrafe je "
            "Einheit nicht gedeckter Nachfrage berücksichtigt "
            "werden sollen. Ein bloß entgangener Verkauf ist "
            "bei ausgewähltem Verkaufserlös bereits durch den "
            "ausbleibenden Erlös abgebildet und begründet keine "
            "separaten Fehlmengenkosten."
        ),
        "compatible_core_ids": [
            "single_period_newsvendor",
        ],
        "required_model_quantities": [
            "shortage_quantity",
        ],
        "required_product_facts": [
            {
                "name": "shortage_cost",
                "label": "Fehlmengenkosten pro Einheit",
                "type": "number",
                "required": True,
                "help": (
                    "Kosten je Einheit nicht gedeckter Nachfrage."
                ),
                "min_value": 0.0,
                "unit": "€ pro Einheit",
            }
        ],
        "required_global_facts": [],
        "linear_builder": build_shortage_cost,
    },
    "annual_ordering_cost": {
        "component_id": "annual_ordering_cost",
        "name": "Jährliche Bestellkosten",
        "effect": "cost",
        "compatible_core_ids": [
            "eoq",
            "continuous_review_q_r",
        ],
        "required_model_quantities": [
            "orders_per_year",
        ],
        "required_product_facts": [
            {
                "name": "ordering_cost",
                "label": "Bestellkosten pro Bestellung",
                "type": "number",
                "required": True,
                "help": (
                    "Fixe Kosten, die bei jeder "
                    "Bestellung entstehen."
                ),
                "min_value": 0.01,
                "unit": "€ pro Bestellung",
            }
        ],
        "required_global_facts": [],
    },
    "annual_holding_cost": {
        "component_id": "annual_holding_cost",
        "name": "Jährliche Lagerhaltungskosten",
        "effect": "cost",
        "compatible_core_ids": [
            "eoq",
            "continuous_review_q_r",
        ],
        "required_model_quantities": [
            "average_inventory",
        ],
        "required_product_facts": [
            {
                "name": "holding_cost",
                "label": "Lagerhaltungskosten",
                "type": "number",
                "required": True,
                "help": (
                    "Jährliche Lagerhaltungskosten "
                    "pro durchschnittlich gelagerter Einheit."
                ),
                "min_value": 0.01,
                "unit": "€ pro Einheit und Jahr",
            }
        ],
        "required_global_facts": [],
    },
    "annual_shortage_cost": {
        "component_id": "annual_shortage_cost",
        "name": "Jährliche Fehlmengenkosten",
        "effect": "cost",
        "compatible_core_ids": [
            "continuous_review_q_r",
        ],
        "required_model_quantities": [
            "annual_shortage_quantity",
        ],
        "required_product_facts": [
            {
                "name": "shortage_cost",
                "label": "Fehlmengenkosten pro Einheit",
                "type": "number",
                "required": True,
                "help": (
                    "Kosten je erwarteter Einheit "
                    "nicht gedeckter Nachfrage."
                ),
                "min_value": 0.01,
                "unit": "€ pro Einheit",
            }
        ],
        "required_global_facts": [],
    },
}


CONSTRAINT_COMPONENTS: dict[
    str,
    ConstraintComponentDefinition,
] = {
    "minimum_order_quantity": {
        "component_id": "minimum_order_quantity",
        "name": "Mindestbestellmenge",
        "scope": "per_product",
        "compatible_core_ids": [
            "single_period_newsvendor",
            "eoq",
            "continuous_review_q_r",
        ],
        "required_model_quantities": [
            "order_quantity",
        ],
        "required_product_facts": [],
        "required_global_facts": [],
        "selection_guidance": (
            "Wähle diesen Baustein nur, wenn ausdrücklich "
            "eine mengenmäßige Untergrenze für die "
            "Bestellmenge gefordert wird, zum Beispiel "
            "'mindestens 50 Einheiten'. Wähle ihn nicht "
            "für Liefertermine, Wochentage, "
            "Lieferhäufigkeiten oder Formulierungen wie "
            "'jede Lieferung'."
        ),
        "parameters": [
            {
                "name": "quantity",
                "label": "Mindestbestellmenge",
                "type": "number",
                "required": True,
                "help": "Die mindestens zu bestellende Menge.",
                "min_value": 0.0,
                "step": 1.0,
                "unit": "Einheiten",
            }
        ],
        "linear_builder": build_minimum_order_quantity,
    },
    "maximum_order_quantity": {
        "component_id": "maximum_order_quantity",
        "name": "Höchstbestellmenge",
        "scope": "per_product",
        "compatible_core_ids": [
            "single_period_newsvendor",
            "eoq",
            "continuous_review_q_r",
        ],
        "required_model_quantities": [
            "order_quantity",
        ],
        "required_product_facts": [],
        "required_global_facts": [],
        "parameters": [
            {
                "name": "quantity",
                "label": "Höchstbestellmenge",
                "type": "number",
                "required": True,
                "help": "Die maximal zulässige Bestellmenge.",
                "min_value": 0.0,
                "step": 1.0,
                "unit": "Einheiten",
            }
        ],
        "linear_builder": build_maximum_order_quantity,
    },
    "order_multiple": {
        "component_id": "order_multiple",
        "name": "Verpackungseinheit",
        "scope": "per_product",
        "compatible_core_ids": [
            "single_period_newsvendor",
            "eoq",
            "continuous_review_q_r",
        ],
        "required_model_quantities": [
            "order_quantity",
        ],
        "required_product_facts": [],
        "required_global_facts": [],
        "parameters": [
            {
                "name": "multiple",
                "label": "Bestellvielfaches",
                "type": "integer",
                "required": True,
                "help": (
                    "Die Bestellmenge muss ein Vielfaches "
                    "dieser Verpackungseinheit sein."
                ),
                "min_value": 1,
                "step": 1,
                "unit": "Einheiten",
            }
        ],
        "linear_builder": build_order_multiple,
    },
    "minimum_cycle_service_level": {
        "component_id": "minimum_cycle_service_level",
        "name": "Mindestwahrscheinlichkeit ohne Fehlbestand",
        "selection_guidance": (
            "Wähle diesen Baustein, wenn eine minimale "
            "Wahrscheinlichkeit, Lieferfähigkeit oder "
            "Bedarfsdeckung ohne Fehlbestand gefordert wird."
        ),
        "scope": "per_product",
        "compatible_core_ids": [
            "single_period_newsvendor",
            "continuous_review_q_r",
        ],
        "required_model_quantities": [],
        "required_product_facts": [],
        "required_global_facts": [],
        "parameters": [
            {
                "name": "service_level_percent",
                "label": (
                    "Mindestwahrscheinlichkeit ohne "
                    "Fehlbestand"
                ),
                "type": "number",
                "required": True,
                "help": (
                    "Gewünschte Mindestwahrscheinlichkeit, "
                    "dass die Nachfrage vollständig aus "
                    "dem verfügbaren Bestand bedient "
                    "werden kann."
                ),
                "min_value": 0.01,
                "max_value": 99.99,
                "step": 0.1,
                "unit": "%",
            }
        ],
        "linear_builder": (
            build_minimum_cycle_service_level
        ),
    },
    "budget_limit": {
        "component_id": "budget_limit",
        "name": "Gemeinsames Beschaffungsbudget",
        "scope": "global",
        "compatible_core_ids": [
            "single_period_newsvendor",
        ],
        "required_model_quantities": [
            "order_quantity",
        ],
        "required_product_facts": [
            {
                "name": "purchase_cost",
                "label": "Einkaufskosten pro Einheit",
                "type": "number",
                "required": True,
                "help": "Kosten für die Beschaffung einer Einheit.",
                "min_value": 0.0,
                "unit": "€ pro Einheit",
            }
        ],
        "required_global_facts": [],
        "parameters": [
            {
                "name": "budget",
                "label": "Verfügbares Gesamtbudget",
                "type": "number",
                "required": True,
                "help": (
                    "Maximal verfügbare Beschaffungsausgaben "
                    "für alle Produkte."
                ),
                "min_value": 0.0,
                "unit": "€",
            }
        ],
        "linear_builder": build_budget_limit,
    },
    "storage_capacity": {
        "component_id": "storage_capacity",
        "name": "Gemeinsame Lagerkapazität",
        "scope": "global",
        "compatible_core_ids": [
            "single_period_newsvendor",
        ],
        "required_model_quantities": [
            "order_quantity",
        ],
        "required_product_facts": [
            {
                "name": "storage_space_per_unit",
                "label": "Platzbedarf pro Einheit",
                "type": "number",
                "required": True,
                "help": (
                    "Benötigte Lagerfläche für eine "
                    "Einheit des Produkts."
                ),
                "min_value": 0.0,
                "unit": "m³ pro Einheit",
            }
        ],
        "required_global_facts": [],
        "parameters": [
            {
                "name": "capacity",
                "label": "Verfügbare Lagerkapazität",
                "type": "number",
                "required": True,
                "help": (
                    "Insgesamt verfügbare Lagerfläche "
                    "für alle Produkte."
                ),
                "min_value": 0.0,
                "unit": "m³",
            }
        ],
        "linear_builder": build_storage_capacity,
    },
    "minimum_reorder_point": {
        "component_id": "minimum_reorder_point",
        "name": "Mindestmeldebestand",
        "scope": "per_product",
        "compatible_core_ids": [
            "continuous_review_q_r",
        ],
        "required_model_quantities": [
            "reorder_point",
        ],
        "required_product_facts": [],
        "required_global_facts": [],
        "requires_explicit_request": True,
        "selection_guidance": (
            "Nur bei ausdrücklich gewünschter zusätzlicher Untergrenze "
            "für den Meldebestand auswählen. Die Bitte, den Meldebestand "
            "zu bestimmen, ist keine solche Untergrenze."
        ),
        "parameters": [
            {
                "name": "reorder_point",
                "label": "Mindestmeldebestand",
                "type": "number",
                "required": True,
                "help": (
                    "Untere Grenze für den Meldebestand."
                ),
                "min_value": 0.0,
                "step": 1.0,
                "unit": "Einheiten",
            }
        ],
    },
    "maximum_reorder_point": {
        "component_id": "maximum_reorder_point",
        "name": "Höchstmeldebestand",
        "scope": "per_product",
        "compatible_core_ids": [
            "continuous_review_q_r",
        ],
        "required_model_quantities": [
            "reorder_point",
        ],
        "required_product_facts": [],
        "required_global_facts": [],
        "requires_explicit_request": True,
        "selection_guidance": (
            "Nur bei ausdrücklich gewünschter zusätzlicher Obergrenze "
            "für den Meldebestand auswählen. Die Bitte, den Meldebestand "
            "zu bestimmen, ist keine solche Obergrenze."
        ),
        "parameters": [
            {
                "name": "reorder_point",
                "label": "Höchstmeldebestand",
                "type": "number",
                "required": True,
                "help": (
                    "Obere Grenze für den Meldebestand."
                ),
                "min_value": 0.0,
                "step": 1.0,
                "unit": "Einheiten",
            }
        ],
    },
    "maximum_inventory_position": {
        "component_id": "maximum_inventory_position",
        "name": "Maximale Bestandsposition (Lagerkapazität)",
        "scope": "per_product",
        "compatible_core_ids": [
            "continuous_review_q_r",
        ],
        "required_model_quantities": [
            "order_quantity",
            "reorder_point",
        ],
        "required_product_facts": [],
        "required_global_facts": [],
        "requires_explicit_request": True,
        "selection_guidance": (
            "Auswählen, wenn eine Lager- oder Bestandsobergrenze für das "
            "Produkt genannt wird. Sie wird konservativ als "
            "Bestellmenge Q plus Meldebestand R höchstens gleich der "
            "Kapazität abgebildet."
        ),
        "parameters": [
            {
                "name": "capacity",
                "label": "Maximale Bestandsposition Q + R",
                "type": "number",
                "required": True,
                "help": (
                    "Obergrenze für Bestellmenge plus Meldebestand. "
                    "Konservative Abbildung der Lagerkapazität."
                ),
                "min_value": 1.0,
                "step": 1.0,
                "unit": "Einheiten",
            }
        ],
    },
}
