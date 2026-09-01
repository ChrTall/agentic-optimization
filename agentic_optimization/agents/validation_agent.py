import math
from typing import Any, get_args

from agentic_optimization.models.registry import (
    get_model_variant_definition,
)
from agentic_optimization.state import (
    WORKFLOW_GENERAL_INPUTS_VALID,
    WORKFLOW_MODEL_INPUTS_INVALID,
    WORKFLOW_MODEL_INPUTS_VALID,
    WORKFLOW_MODEL_SPECIFIC_INPUTS_COMPLETE,
    WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_READY_FOR_REVIEW,
    DecisionType,
    DemandType,
    GraphState,
    ObjectiveType,
    PlanningHorizon,
    ReviewType,
    with_control,
)

ALLOWED_DECISION_TYPES = set(get_args(DecisionType))
ALLOWED_OBJECTIVE_TYPES = set(get_args(ObjectiveType))
ALLOWED_DEMAND_TYPES = set(get_args(DemandType))
ALLOWED_PLANNING_HORIZONS = set(get_args(PlanningHorizon))
ALLOWED_REVIEW_TYPES = set(get_args(ReviewType))


def validate_general_requirements(
    state: GraphState,
) -> GraphState:
    """
    Prüft das Ergebnis des Requirement Agents nach Formular 1 und 2.

    Fehlende Felder, ungültige Werte, Widersprüche zwischen Formular und
    Freitext sowie unsupported Requirements verhindern die Modellauswahl.
    """

    requirements = {
        **state.get("requirements", {}),
        # Der Produktumfang ist keine Nutzerauswahl mehr. Alle aktuell
        # registrierten Modellvarianten gelten für genau ein Produkt.
        "product_scope": "single_product",
    }

    errors: list[str] = []
    missing_fields = [
        field
        for field in state.get("missing_fields", [])
        if field != "product_scope"
    ]
    requirement_conflicts = state.get("requirement_conflicts", [])
    unsupported_requirements = state.get("unsupported_requirements",[],)

    fields_to_validate = {
        "decision": (
            requirements.get("decision"),
            ALLOWED_DECISION_TYPES,
            "Die Art der Entscheidung besitzt einen ungültigen Wert.",
        ),
        "objective": (
            requirements.get("objective"),
            ALLOWED_OBJECTIVE_TYPES,
            "Das Optimierungsziel besitzt einen ungültigen Wert.",
        ),
        "demand_type": (
            requirements.get("demand_type"),
            ALLOWED_DEMAND_TYPES,
            "Die Nachfrageart besitzt einen ungültigen Wert.",
        ),
        "planning_horizon": (
            requirements.get("planning_horizon"),
            ALLOWED_PLANNING_HORIZONS,
            "Der Planungshorizont besitzt einen ungültigen Wert.",
        ),
    }

    if requirements.get("planning_horizon") == "multiple_periods":
        fields_to_validate["review_type"] = (
            requirements.get("review_type"),
            ALLOWED_REVIEW_TYPES,
            "Die Art der Bestandsüberprüfung besitzt einen ungültigen Wert.",
        )

    for field_name, (
        value,
        allowed_values,
        error_message,
    ) in fields_to_validate.items():
        if value is None or value == "":
            if field_name not in missing_fields:
                missing_fields.append(field_name)
            continue

        if value not in allowed_values:
            errors.append(error_message)
            continue

    missing_fields = list(dict.fromkeys(missing_fields))
    needs_clarification = bool(
        missing_fields
        or errors
        or requirement_conflicts
        or unsupported_requirements
    )

    return with_control(
        {
            **state,
            "requirements": requirements,
            "missing_fields": missing_fields,
            "general_validation_errors": errors,
        },
        event=None,
        workflow_status=(
            WORKFLOW_NEEDS_GENERAL_CLARIFICATION
            if needs_clarification
            else WORKFLOW_GENERAL_INPUTS_VALID
        ),
    )


def _get_selected_model_inputs(
    state: GraphState,
) -> dict[str, Any]:
    """Liest die gespeicherten Formular-3-Werte der ausgewählten Variante."""

    family = state.get("selected_model_family")
    variant = state.get("selected_model_variant")

    if not family or not variant:
        return {}

    return (
        state.get("model_specific_inputs", {})
        .get(family, {})
        .get(variant, {})
    )

def detect_model_specific_inputs(
    state: GraphState,
) -> GraphState:
    """
    Prüft, ob für die ausgewählte Modellfamilie und Variante
    alle Pflichtinputs vorhanden sind.
    """

    family = state.get("selected_model_family")
    variant = state.get("selected_model_variant")

    if not family or not variant:
        return with_control(
            {
                **state,
                "required_model_inputs": [],
                "missing_model_input_fields": [],
                "model_input_validation_errors": [
                    "Es konnte keine vollständig unterstützte Modellfamilie und Modellvariante ausgewählt werden."
                ],
                "ready_for_optimization": False,
            },
            event=None,
            workflow_status=WORKFLOW_MODEL_INPUTS_INVALID,
        )

    model_definition = get_model_variant_definition(
        family,
        variant,
    )

    if model_definition is None:
        return with_control(
            {
                **state,
                "required_model_inputs": [],
                "missing_model_input_fields": [],
                "model_input_validation_errors": [
                    f"Die Modellvariante '{family}/{variant}' ist nicht registriert."
                ],
                "ready_for_optimization": False,
            },
            event=None,
            workflow_status=WORKFLOW_MODEL_INPUTS_INVALID,
        )

    selected_inputs = _get_selected_model_inputs(state)
    required_fields = model_definition["required_fields"]

    missing_fields = [
        field
        for field in required_fields
        if selected_inputs.get(field) in (None, "", [])
    ]

    required_model_inputs = (
        [
            {
                "family": family,
                "variant": variant,
                "form_id": f"{family}_{variant}_parameters",
                "fields": model_definition["input_fields"],
            }
        ]
        if missing_fields
        else []
    )

    return with_control(
        {
            **state,
            "required_model_inputs": required_model_inputs,
            "missing_model_input_fields": missing_fields,
            "model_input_validation_errors": [],
            "ready_for_optimization": False,
        },
        event=None,
        workflow_status=(
            WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS
            if missing_fields
            else WORKFLOW_MODEL_SPECIFIC_INPUTS_COMPLETE
        ),
    )


def validate_model_specific_inputs(
    state: GraphState,
) -> GraphState:
    """
    Prüft die Modellparameter aus Formular 3.

    Neben fehlenden Pflichtwerten werden die Werte durch den input_validator
    der ausgewählten Modellvariante geprüft.
    """

    family = state.get("selected_model_family")
    variant = state.get("selected_model_variant")

    errors: list[str] = []
    missing_fields: list[str] = []
    required_model_inputs: list[dict[str, Any]] = []

    model_definition = None

    if not family or not variant:
        errors.append(
            "Es wurde keine vollständig unterstützte "
            "Modellfamilie und Modellvariante ausgewählt."
        )
    else:
        model_definition = get_model_variant_definition(
            family,
            variant,
        )

        if model_definition is None:
            errors.append(
                f"Die Modellvariante '{family}/{variant}' "
                "ist nicht registriert."
            )

    if model_definition is not None:
        selected_inputs = _get_selected_model_inputs(state)
        required_fields = model_definition["required_fields"]

        missing_fields = [
            field
            for field in required_fields
            if selected_inputs.get(field) in (None, "", [])
        ]

        input_validator = model_definition["input_validator"]
        errors.extend(input_validator(selected_inputs))

        if errors or missing_fields:
            required_model_inputs = [
                {
                    "family": family,
                    "variant": variant,
                    "form_id": f"{family}_{variant}_parameters",
                    "fields": model_definition["input_fields"],
                }
            ]

    validation_failed = bool(errors or missing_fields)

    return with_control(
        {
            **state,
            "missing_model_input_fields": missing_fields,
            "required_model_inputs": required_model_inputs,
            "model_input_validation_errors": errors,
            "ready_for_optimization": False,
        },
        event=None,
        workflow_status=(
            WORKFLOW_MODEL_INPUTS_INVALID
            if validation_failed
            else WORKFLOW_MODEL_INPUTS_VALID
        ),
    )

def validate_structured_constraints(
    state: GraphState,
) -> GraphState:
    """
    Prüft die in Formular 3 bestätigten Constraints.

    Geprüft werden Constraint-Typen, Parameter und Pflichtwerte. Danach wird
    der constraint_validator der ausgewählten Modellvariante ausgeführt.
    """

    family = state.get("selected_model_family")
    variant = state.get("selected_model_variant")
    constraints = state.get("structured_constraints", [])
    unsupported_constraints = state.get(
        "unsupported_constraints",
        [],
    )

    errors: list[str] = []
    required_constraint_inputs: list[dict[str, Any]] = []

    # ---------------------------------------------------------------
    # 1. Ausgewählte Modellvariante prüfen
    # ---------------------------------------------------------------

    model_definition = None

    if not family or not variant:
        errors.append(
            "Für die Constraint-Validierung fehlt eine vollständig "
            "ausgewählte Modellfamilie und Modellvariante."
        )
    else:
        model_definition = get_model_variant_definition(
            family,
            variant,
        )

        if model_definition is None:
            errors.append(
                f"Die Modellvariante '{family}/{variant}' "
                "ist nicht registriert."
            )

    # ---------------------------------------------------------------
    # 2. Grundstruktur der State-Felder prüfen
    # ---------------------------------------------------------------

    if not isinstance(constraints, list):
        errors.append(
            "'structured_constraints' muss eine Liste sein."
        )
        constraints = []

    if not isinstance(unsupported_constraints, list):
        errors.append(
            "'unsupported_constraints' muss eine Liste sein."
        )
        unsupported_constraints = []

    # ---------------------------------------------------------------
    # 3. Unsupported Constraints als nicht blockierende Hinweise prüfen
    # ---------------------------------------------------------------

    for unsupported in unsupported_constraints:
        # Inhaltlich nicht unterstützte Freitext-Constraints werden vom
        # Solver ignoriert. Nur ein technisch ungültiges State-Format ist
        # weiterhin ein Validierungsfehler.
        if not isinstance(unsupported, str) or not unsupported.strip():
            errors.append(
                "Eine nicht unterstützte Nebenbedingung besitzt "
                "kein gültiges Textformat."
            )

    # ---------------------------------------------------------------
    # 4. Allgemeine Constraintstruktur validieren
    # ---------------------------------------------------------------

    seen_types: set[str] = set()

    for constraint_index, constraint in enumerate(constraints):
        if not isinstance(constraint, dict):
            errors.append(
                "Ein strukturierter Constraint muss als Dictionary "
                "angegeben werden."
            )
            continue

        constraint_type = constraint.get("type")
        parameters = constraint.get("parameters", {})

        if (
            not isinstance(constraint_type, str)
            or not constraint_type.strip()
        ):
            errors.append(
                "Ein strukturierter Constraint besitzt keinen "
                "gültigen Typ."
            )
            continue

        constraint_type = constraint_type.strip()

        if constraint_type in seen_types:
            errors.append(
                f"Der Constraint-Typ '{constraint_type}' "
                "wurde mehrfach angegeben."
            )
            continue

        seen_types.add(constraint_type)

        if model_definition is None:
            continue

        definition = model_definition[
            "supported_constraints"
        ].get(constraint_type)

        if definition is None:
            errors.append(
                f"Der Constraint-Typ '{constraint_type}' wird für "
                f"'{family}/{variant}' nicht unterstützt."
            )
            continue

        if not isinstance(parameters, dict):
            errors.append(
                f"Die Parameter für '{constraint_type}' müssen "
                "als Dictionary angegeben werden."
            )
            continue

        expected_parameter_names = {
            parameter_definition["name"]
            for parameter_definition in definition["parameters"]
        }

        submitted_parameter_names = set(parameters)

        unknown_parameter_names = (
            submitted_parameter_names
            - expected_parameter_names
        )

        for parameter_name in sorted(
            unknown_parameter_names
        ):
            errors.append(
                f"Der Parameter '{parameter_name}' wird für den "
                f"Constraint '{constraint_type}' nicht unterstützt."
            )

        missing_parameter_fields: list[
            dict[str, Any]
        ] = []

        for parameter_definition in definition["parameters"]:
            parameter_name = parameter_definition["name"]
            parameter_type = parameter_definition["type"]
            required = parameter_definition["required"]

            parameter_is_missing = (
                parameter_name not in parameters
                or parameters.get(parameter_name) is None
                or parameters.get(parameter_name) == ""
            )

            if required and parameter_is_missing:
                errors.append(
                    f"Für den Constraint '{constraint_type}' "
                    f"fehlt der Parameter '{parameter_name}'."
                )

                missing_parameter_fields.append(
                    {
                        "name": parameter_name,
                        "label": parameter_definition["label"],
                        "type": parameter_type,
                        "required": required,
                        "help": parameter_definition.get("help"),
                    }
                )

                continue

            if parameter_is_missing:
                continue

            value = parameters[parameter_name]

            if parameter_type == "number" and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                errors.append(
                    f"Der Parameter '{parameter_name}' des "
                    f"Constraints '{constraint_type}' muss "
                    "eine endliche Zahl sein."
                )

        if missing_parameter_fields:
            required_constraint_inputs.append(
                {
                    "constraint_index": constraint_index,
                    "constraint_type": constraint_type,
                    "source_text": constraint.get(
                        "source_text"
                    ),
                    "fields": missing_parameter_fields,
                }
            )

    # ---------------------------------------------------------------
    # 5. Variantenspezifischen Constraintvalidator ausführen
    # ---------------------------------------------------------------

    if not errors and model_definition is not None:
        selected_inputs = _get_selected_model_inputs(state)

        constraint_validator = model_definition[
            "constraint_validator"
        ]

        variant_errors = constraint_validator(
            selected_inputs,
            constraints,
        )

        errors.extend(variant_errors)

    # ---------------------------------------------------------------
    # 6. Workflowstatus setzen
    # ---------------------------------------------------------------

    validation_failed = bool(errors)

    return with_control(
        {
            **state,
            "constraint_validation_errors": errors,
            "required_constraint_inputs": (
                required_constraint_inputs
            ),
            "ready_for_optimization": not validation_failed,
            "review_confirmed": False,
        },
        event=None,
        workflow_status=(
            WORKFLOW_MODEL_INPUTS_INVALID
            if validation_failed
            else WORKFLOW_READY_FOR_REVIEW
        ),
    )
