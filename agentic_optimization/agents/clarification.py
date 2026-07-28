from agentic_optimization.state import (
    WORKFLOW_GENERAL_INPUTS_VALID,
    WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    GraphState,
    with_control,
)

FIELD_QUESTIONS = {
    "decision": "Welche konkrete Entscheidung soll optimiert werden? Zum Beispiel Bestellmenge, Produktionsmenge oder Lagerbestand.",
    "objective": "Was ist das Ziel der Optimierung? Zum Beispiel Kosten minimieren, Gewinn maximieren oder Servicelevel erreichen.",
    "demand_type": "Wie ist die Nachfrage beschaffen? Ist sie bekannt, unsicher, saisonal oder historisch geschätzt?",
    "constraints": "Welche Nebenbedingungen müssen berücksichtigt werden? Zum Beispiel Budget, Kapazität, Lagerplatz, Mindestmengen oder Lieferzeiten.",
    "product_scope": "Für wie viele Produkte soll eine Entscheidung getroffen werden: für ein Produkt oder für mehrere Produkte?",
    "planning_horizon": "Geht es um eine einmalige Bestellentscheidung oder um mehrere aufeinanderfolgende Zeiträume?",
}


def clarification_node(state: GraphState) -> GraphState:
    """
    Erstellt eine Rückfrage, falls wichtige Informationen fehlen.

    Diese Version beantwortet noch nicht automatisch die Rückfrage,
    sondern formuliert nur die Frage, die später im Chat angezeigt werden kann.
    """

    missing_fields = state.get("missing_fields", [])
    requirement_conflicts = state.get("requirement_conflicts", [])

    if not missing_fields and not requirement_conflicts:
        return with_control(
            {
                **state,
                "clarification_question": "",
            },
            event=None,
            workflow_status=WORKFLOW_GENERAL_INPUTS_VALID,
        )

    questions = [
        FIELD_QUESTIONS.get(field, f"Bitte ergänze die Angabe: {field}.")
        for field in missing_fields
    ]
    questions.extend(
    conflict.get(
        "message",
        "Eine Formularangabe widerspricht der freien Beschreibung.",
    )
    for conflict in requirement_conflicts
)

    clarification_question = "Ich brauche noch folgende Informationen:\n\n"
    clarification_question += "\n".join(f"- {question}" for question in questions)

    return with_control(
        {
            **state,
            "clarification_question": clarification_question,
        },
        event=None,
        workflow_status=WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    )