from typing import Any

from agentic_optimization.state import (
    WORKFLOW_READY_FOR_RESULT,
    GraphState,
    with_control,
)


def result_node(state: GraphState) -> GraphState:
    """
    Bereitet das Optimierungsergebnis für nachgelagerte
    Ausgabekanäle auf.
    """

    requirements = state.get("requirements", {})
    model_config = state.get("model_config", {})

    family = state.get("selected_model_family")
    variant = state.get("selected_model_variant")

    selected_model_inputs = _get_selected_model_inputs(
        state=state,
        family=family,
        variant=variant,
    )

    optimization_result = state.get(
        "optimization_result"
    )

    optimization_errors = state.get(
        "optimization_errors",
        [],
    )

    result = {
        "summary": _build_summary(
            requirements=requirements,
            model_config=model_config,
            family=family,
            variant=variant,
        ),
        "requirements": requirements,
        "selected_model": {
            "family_id": family,
            "family_name": model_config.get("family_name"),
            "variant_id": variant,
            "variant_name": model_config.get("variant_name"),
            "explanation": model_config.get("explanation"),
        },
        "model_config": model_config,
        "model_specific_inputs": state.get(
            "model_specific_inputs",
            {},
        ),
        "selected_model_inputs": selected_model_inputs,
        "structured_constraints": state.get(
            "structured_constraints",
            [],
        ),
        "unsupported_constraints": state.get(
            "unsupported_constraints",
            [],
        ),
        "constraint_validation_errors": state.get(
            "constraint_validation_errors",
            [],
        ),
        "optimization_result": optimization_result,
        "optimization_errors": optimization_errors,
        "next_steps": [
            "Optimierungsergebnis fachlich prüfen.",
            "Modellannahmen bei Bedarf anpassen.",
            "Bestellentscheidung dokumentieren.",
        ],
    }

    return with_control(
        {
            **state,
            "result": result,
        },
        event=None,
        workflow_status=WORKFLOW_READY_FOR_RESULT,
    )


def _get_selected_model_inputs(
    state: GraphState,
    family: str | None,
    variant: str | None,
) -> dict[str, Any]:
    if not family or not variant:
        return {}

    return (
        state.get("model_specific_inputs", {})
        .get(family, {})
        .get(variant, {})
    )


def _build_summary(
    requirements: dict[str, Any],
    model_config: dict[str, Any],
    family: str | None,
    variant: str | None,
) -> str:
    decision = requirements.get(
        "decision",
        "nicht angegeben",
    )
    objective = requirements.get(
        "objective",
        "nicht angegeben",
    )

    family_name = (
        model_config.get("family_name")
        or family
        or "keine Modellfamilie"
    )
    variant_name = (
        model_config.get("variant_name")
        or variant
        or "keine Modellvariante"
    )

    return (
        f"Für die Entscheidung '{decision}' mit dem Ziel "
        f"'{objective}' wurde die Modellfamilie "
        f"'{family_name}' mit der Variante "
        f"'{variant_name}' ausgewählt."
    )