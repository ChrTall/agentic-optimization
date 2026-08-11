from typing import Any

from agentic_optimization.models.registry import (
    get_model_family_definition,
    get_model_variant_definition,
)
from agentic_optimization.state import (
    WORKFLOW_READY_FOR_RESULT,
    GraphState,
    with_control,
)


def result_node(state: GraphState) -> GraphState:
    """
    Bündelt den finalen Workflow-Zustand zu einem ausgabekanalunabhängigen Ergebnis.

    Sowohl erfolgreiche Optimierungen als auch Solverfehler und unzulässige
    Modelle werden in einem einheitlichen Ergebnisobjekt zusammengeführt.
    """

    requirements = state.get("requirements", {})

    family = state.get("selected_model_family")
    variant = state.get("selected_model_variant")

    family_definition = get_model_family_definition(family)
    variant_definition = get_model_variant_definition(family, variant)

    family_name = (
        family_definition["family_name"]
        if family_definition is not None
        else family
    )

    variant_name = (
        variant_definition["variant_name"]
        if variant_definition is not None
        else variant
    )

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

    # Das Ergebnis enthält einen abgeschlossenen Snapshot der für die
    # Darstellung relevanten Daten und kann unabhängig vom Graph-State
    # an unterschiedliche Ausgabekanäle übergeben werden.
    result = {
        "summary": _build_summary(
            requirements=requirements,
            family_name=family_name,
            variant_name=variant_name,
        ),
        "requirements": requirements,
        "selected_model": {
            "family_id": family,
            "family_name": family_name,
            "variant_id": variant,
            "variant_name": variant_name,
        },
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
    """Liest die Eingaben der aktuell ausgewählten Modellvariante."""

    if not family or not variant:
        return {}

    return (
        state.get("model_specific_inputs", {})
        .get(family, {})
        .get(variant, {})
    )


def _build_summary(
    requirements: dict[str, Any],
    family_name: str | None,
    variant_name: str | None,
) -> str:
    """Erstellt eine kurze Zusammenfassung der Modellauswahl."""

    decision = requirements.get(
        "decision",
        "nicht angegeben",
    )
    objective = requirements.get(
        "objective",
        "nicht angegeben",
    )

    family_label = (
        family_name or "keine Modellfamilie"
    )
    variant_label = (
        variant_name or "keine Modellvariante"
    )

    return (
        f"Für die Entscheidung '{decision}' mit dem Ziel "
        f"'{objective}' wurde die Modellfamilie "
        f"'{family_label}' mit der Variante "
        f"'{variant_label}' ausgewählt."
    )
