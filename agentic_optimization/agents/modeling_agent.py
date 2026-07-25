from agentic_optimization.state import (
    WORKFLOW_MODEL_SELECTED,
    GraphState,
    ModelConfig,
    with_control,
)


def modeling_agent(state: GraphState) -> GraphState:
    """
    Wählt eine vollständig unterstützte Modellfamilie und Modellvariante aus.
    """

    requirements = state.get("requirements", {})

    decision = str(requirements.get("decision", "")).lower()
    objective = str(requirements.get("objective", "")).lower()
    demand_type = str(requirements.get("demand_type", "")).lower()
    constraints = str(requirements.get("constraints", "")).lower()
    raw_description = str(
        requirements.get("raw_description", "")
    ).lower()

    combined_text = (
        f"{decision} {objective} {demand_type} "
        f"{constraints} {raw_description}"
    )

    model_family: str | None = None
    model_variant: str | None = None
    family_name: str | None = None
    variant_name: str | None = None

    has_order_decision = any(
        keyword in combined_text
        for keyword in [
            "bestellmenge",
            "bestellen",
            "einkaufsmenge",
        ]
    )

    has_uncertain_demand = any(
        keyword in combined_text
        for keyword in [
            "unsicher",
            "saisonal",
            "nachfrageszenarien",
            "wahrscheinlichkeiten",
        ]
    )

    if has_order_decision and has_uncertain_demand:
        model_family = "newsvendor"
        model_variant = "single_product_single_period"
        family_name = "Newsvendor"
        variant_name = "Single Product, Single Period"

        explanation = (
            "Das Problem beschreibt eine Bestellmengenentscheidung "
            "für ein einzelnes Produkt und eine einzelne Periode "
            "unter unsicherer Nachfrage. Dafür eignet sich die "
            "Newsvendor-Variante 'Single Product, Single Period'."
        )
    else:
        explanation = (
            "Es konnte derzeit keine vollständig unterstützte "
            "Modellfamilie mit konkreter Variante ausgewählt werden."
        )

    model_config: ModelConfig = {
        "explanation": explanation,
        "objective": requirements.get("objective"),
        "decision": requirements.get("decision"),
    }

    if model_family is not None and family_name is not None:
        model_config["family_id"] = model_family
        model_config["family_name"] = family_name

    if model_variant is not None and variant_name is not None:
        model_config["variant_id"] = model_variant
        model_config["variant_name"] = variant_name

    return with_control(
        {
            **state,
            "selected_model_family": model_family,
            "selected_model_variant": model_variant,
            "model_config": model_config,
        },
        event=None,
        workflow_status=WORKFLOW_MODEL_SELECTED,
    )
