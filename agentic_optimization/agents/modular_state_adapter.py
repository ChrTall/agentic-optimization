"""
Überführt die bisherige Modellauswahl in den modularen Zustand.
"""

from agentic_optimization.state import (
    GraphState,
    ModularModelState,
)


LEGACY_SELECTION_TO_CORE = {
    (
        "newsvendor",
        "single_product_single_period",
    ): "single_period_newsvendor",
    (
        "eoq",
        "basic_eoq",
    ): "eoq",
    (
        "reorder_point",
        "q_r_policy",
    ): "continuous_review_q_r",
}


def initialize_modular_model_state(
    state: GraphState,
) -> GraphState:
    """Initialisiert den modularen Zustand aus der alten Auswahl."""

    selection = (
        state.get("selected_model_family"),
        state.get("selected_model_variant"),
    )
    core_id = LEGACY_SELECTION_TO_CORE.get(selection)

    if core_id is None:
        raise ValueError(
            "Die bisherige Modellauswahl kann keinem "
            "modularen Grundmodell zugeordnet werden."
        )

    existing_modular_state = state.get(
        "modular_model",
        {},
    )
    products = list(
        existing_modular_state.get("products", [])
    )

    product_scope = state.get(
        "requirements",
        {},
    ).get("product_scope")

    if not products:
        if product_scope == "multiple_products":
            raise ValueError(
                "Für mehrere Produkte muss vor der modularen "
                "Modellvorbereitung eine Produktliste vorliegen."
            )

        products = [
            {
                "product_id": "product_1",
                "forecast": {},
                "facts": {},
            }
        ]

    evidence = list(
        existing_modular_state.get("evidence", [])
    )
    evidence_ids = {
        item["source_id"]
        for item in evidence
    }

    raw_input = state.get("raw_input", "").strip()

    if raw_input and "raw_input" not in evidence_ids:
        evidence.append(
            {
                "source_id": "raw_input",
                "content": raw_input,
            }
        )
        evidence_ids.add("raw_input")

    constraints = state.get(
        "requirements",
        {},
    ).get("constraints", [])

    for index, constraint in enumerate(constraints):
        source_id = f"requirement_constraint_{index}"

        if source_id not in evidence_ids:
            evidence.append(
                {
                    "source_id": source_id,
                    "content": constraint,
                }
            )
            evidence_ids.add(source_id)

    modular_state: ModularModelState = {
        "core_id": core_id,
        "products": products,
        "global_facts": dict(
            existing_modular_state.get(
                "global_facts",
                {},
            )
        ),
        "evidence": evidence,
        "review_confirmed": False,
        "optimization_result": None,
        "preparation_error": None,
        "pending_input_submissions": [],
    }

    return {
        **state,
        "modular_model": modular_state,
    }