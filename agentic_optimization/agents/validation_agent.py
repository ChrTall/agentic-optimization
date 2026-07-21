from agentic_optimization.models.registry import get_model_definition
from agentic_optimization.state import (
    GraphState,
    WORKFLOW_GENERAL_INPUTS_VALID,
    WORKFLOW_MODEL_INPUTS_INVALID,
    WORKFLOW_MODEL_SPECIFIC_INPUTS_COMPLETE,
    WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_READY_FOR_RESULT,
    with_control,
)


def validate_general_requirements(state: GraphState) -> GraphState:
    """
    Validiert allgemeine Anforderungen vor der Modellauswahl.

    Aktuell wird nur geprüft, ob Pflichtangaben fehlen.
    Fachliche Konsistenzprüfungen können später hier ergänzt werden,
    z.B. Widersprüche zwischen Formular und Freitext.
    """

    missing_fields = state.get("missing_fields", [])

    return with_control(
        {
            **state,
            "general_validation_errors": [],
            "consistency_warnings": [],
        },
        event=None,
        workflow_status=(
            WORKFLOW_NEEDS_GENERAL_CLARIFICATION
            if missing_fields
            else WORKFLOW_GENERAL_INPUTS_VALID
        ),
    )


def detect_model_specific_inputs(state: GraphState) -> GraphState:
    """
    Prüft nach der Modellauswahl, ob modellspezifische Eingaben fehlen.

    Dafür wird die Model Registry als Bauplan genutzt:
    Sie definiert, welche Pflichtfelder das ausgewählte Modell benötigt.
    """

    model_config = state.get("model_config", {})
    selected_model = (
        state.get("selected_model")
        or model_config.get("model_type")
    )

    model_definition = get_model_definition(selected_model)

    if model_definition is None:
        return with_control(
            {
                **state,
                "selected_model": selected_model,
                "required_model_inputs": [],
                "missing_model_input_fields": [],
            },
            event=None,
            workflow_status=WORKFLOW_MODEL_SPECIFIC_INPUTS_COMPLETE,
        )

    model_specific_inputs = state.get("model_specific_inputs", {})
    selected_model_inputs = model_specific_inputs.get(selected_model, {})

    required_fields = model_definition["required_fields"]

    missing_fields = [
        field
        for field in required_fields
        if selected_model_inputs.get(field) in (None, "", [])
    ]

    if missing_fields:
        return with_control(
            {
                **state,
                "selected_model": selected_model,
                "required_model_inputs": [
                    {
                        "model": selected_model,
                        "form_id": f"{selected_model}_parameters",
                        "fields": model_definition["input_fields"],
                    }
                ],
                "missing_model_input_fields": missing_fields,
            },
            event=None,
            workflow_status=WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
        )

    return with_control(
        {
            **state,
            "selected_model": selected_model,
            "required_model_inputs": [],
            "missing_model_input_fields": [],
        },
        event=None,
        workflow_status=WORKFLOW_MODEL_SPECIFIC_INPUTS_COMPLETE,
    )


def validate_model_specific_inputs(state: GraphState) -> GraphState:
    """
    Validiert die eingegebenen Modellparameter.

    Der passende Validator wird aus der Model Registry geholt.
    """

    model_config = state.get("model_config", {})
    selected_model = (
        state.get("selected_model")
        or model_config.get("model_type")
    )

    model_specific_inputs = state.get("model_specific_inputs", {})
    selected_model_inputs = (
        model_specific_inputs.get(selected_model, {})
        if selected_model
        else {}
    )

    model_definition = get_model_definition(selected_model)

    if selected_model is None:
        errors = ["Es wurde kein Modell ausgewählt."]
        missing_fields = []
        required_model_inputs = []
    elif model_definition is None:
        errors = []
        missing_fields = []
        required_model_inputs = []
    else:
        required_fields = model_definition["required_fields"]

        missing_fields = [
            field
            for field in required_fields
            if selected_model_inputs.get(field) in (None, "", [])
        ]

        validator = model_definition["validator"]
        errors = validator(selected_model_inputs)

        required_model_inputs = (
            [
                {
                    "model": selected_model,
                    "form_id": f"{selected_model}_parameters",
                    "fields": model_definition["input_fields"],
                }
            ]
            if errors or missing_fields
            else []
        )

    validation_errors = errors

    return with_control(
        {
            **state,
            "missing_model_input_fields": missing_fields,
            "required_model_inputs": required_model_inputs,
            "model_input_validation_errors": validation_errors,
            "ready_for_generation": not (validation_errors or missing_fields),
        },
        event=None,
        workflow_status=(
            WORKFLOW_MODEL_INPUTS_INVALID
            if validation_errors or missing_fields
            else WORKFLOW_READY_FOR_RESULT
        ),
    )
