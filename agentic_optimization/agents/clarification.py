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
    "constraints": "Welche betrieblichen Rahmenbedingungen oder Limits müssen berücksichtigt werden? Zum Beispiel Budget, Kapazität, Lagerplatz, Mindestmengen oder Lieferzeiten.",
    "product_scope": "Für wie viele Produkte soll eine Entscheidung getroffen werden: für ein Produkt oder für mehrere Produkte?",
    "planning_horizon": "Geht es um eine einmalige Bestellentscheidung oder um mehrere aufeinanderfolgende Zeiträume?",
}

FIELD_LABELS = {
    "decision": "Art der Entscheidung",
    "objective": "Optimierungsziel",
    "demand_type": "Nachfrage",
    "product_scope": "Produktumfang",
    "planning_horizon": "Planungszeitraum",
}

VALUE_LABELS = {
    "order_quantity": "Bestellmenge",
    "production_quantity": "Produktionsmenge",
    "inventory_level": "Lagerbestand",
    "maximize_profit": "Gewinn maximieren",
    "minimize_cost": "Kosten minimieren",
    "reach_service_level": "Servicelevel erreichen",
    "known": "sichere oder bekannte Nachfrage",
    "uncertain": "unsichere Nachfrage",
    "historical_estimate": (
        "historisch geschätzte Nachfrage"
    ),
    "single_product": "ein Produkt",
    "multiple_products": "mehrere Produkte",
    "single_period": "eine einmalige Entscheidung",
    "multiple_periods": "mehrere Zeiträume",
}

def clarification_node(state: GraphState) -> GraphState:
    """
    Erstellt die in Formular 2 angezeigte Clarification-Meldung.

    Fehlende oder ungültige Angaben, Widersprüche und unsupported
    Requirements werden zu einer gemeinsamen Meldung zusammengeführt.
    """

    missing_fields = state.get("missing_fields", [])
    requirement_conflicts = state.get("requirement_conflicts", [])
    unsupported_requirements = state.get("unsupported_requirements", [])
    general_validation_errors = state.get("general_validation_errors", [])

    if (
        not missing_fields
        and not requirement_conflicts
        and not unsupported_requirements
        and not general_validation_errors
    ):
        return with_control(
            {
                **state,
                "clarification_question": "",
            },
            event=None,
            workflow_status=WORKFLOW_GENERAL_INPUTS_VALID,
        )

    questions = [
        FIELD_QUESTIONS.get(
            field,
            f"Bitte ergänzen Sie die Angabe: {field}.",
        )
        for field in missing_fields
    ]
    questions.extend(general_validation_errors)

    for conflict in requirement_conflicts:
        field_name = conflict.get(
            "field",
            "",
        )

        field_label = FIELD_LABELS.get(
            field_name,
            field_name or "Angabe",
        )

        form_value = conflict.get(
            "form_value",
            "",
        )

        raw_input_value = conflict.get(
            "raw_input_value",
            "",
        )

        form_label = VALUE_LABELS.get(
            form_value,
            form_value,
        )

        raw_input_label = VALUE_LABELS.get(
            raw_input_value,
            raw_input_value,
        )

        questions.append(
            f"Die Angaben zu „{field_label}“ "
            "widersprechen sich. "
            f"In Formular 1 wurde „{form_label}“ "
            "ausgewählt. Im Beschreibungstext wurde "
            f"„{raw_input_label}“ erkannt. Bitte "
            "korrigieren Sie entweder die Auswahl oder "
            "den Beschreibungstext."
        )

    for unsupported in unsupported_requirements:
        message = unsupported.get("message", "").strip()

        if message:
            questions.append(message)
            continue

        user_value = unsupported.get(
            "user_value",
            "Diese Anforderung",
        )

        questions.append(
            f"Die Anforderung „{user_value}“ wird derzeit "
            "nicht unterstützt. Bitte passen Sie die "
            "Problembeschreibung an."
        )

    clarification_question = "Bitte prüfen Sie folgende Angaben:\n\n"
    clarification_question += "\n".join(f"- {question}" for question in questions)

    return with_control(
        {
            **state,
            "clarification_question": clarification_question,
        },
        event=None,
        workflow_status=WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    )
