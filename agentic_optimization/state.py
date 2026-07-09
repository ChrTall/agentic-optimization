from typing import Any, Final, Literal, TypedDict


WORKFLOW_GENERAL_REQUIREMENTS_EXTRACTED: Final = "general_requirements_extracted"
WORKFLOW_NEEDS_GENERAL_CLARIFICATION: Final = "needs_general_clarification"
WORKFLOW_GENERAL_INPUTS_VALID: Final = "general_inputs_valid"
WORKFLOW_GENERAL_INPUTS_INVALID: Final = "general_inputs_invalid"
WORKFLOW_MODEL_SELECTED: Final = "model_selected"
WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS: Final = "needs_model_specific_inputs"
WORKFLOW_MODEL_SPECIFIC_INPUTS_SUBMITTED: Final = "model_specific_inputs_submitted"
WORKFLOW_MODEL_SPECIFIC_INPUTS_COMPLETE: Final = "model_specific_inputs_complete"
WORKFLOW_MODEL_INPUTS_INVALID: Final = "model_inputs_invalid"
WORKFLOW_READY_FOR_RESULT: Final = "ready_for_result"

EVENT_NEW_PROBLEM: Final = "new_problem"
EVENT_GENERAL_CLARIFICATION_SUBMITTED: Final = "general_clarification_submitted"
EVENT_MODEL_INPUTS_SUBMITTED: Final = "model_inputs_submitted"

_CLEAR = object()

WorkflowStatus = Literal[
    "general_requirements_extracted",
    "needs_general_clarification",
    "general_inputs_valid",
    "general_inputs_invalid",
    "model_selected",
    "needs_model_specific_inputs",
    "model_specific_inputs_submitted",
    "model_specific_inputs_complete",
    "model_inputs_invalid",
    "ready_for_result",
]

WorkflowEvent = Literal[
    "new_problem",
    "general_clarification_submitted",
    "model_inputs_submitted",
]

class ControlState(TypedDict, total=False):
    event: WorkflowEvent
    workflow_status: WorkflowStatus

class GraphState(TypedDict, total=False):
    """
    Gemeinsamer Zustand des LangGraph-Workflows.

    Dieser State wird von Knoten zu Knoten weitergegeben.
    Jeder Agent liest relevante Felder aus und ergänzt neue Ergebnisse.
    """

    # Workflow control
    control: ControlState

    # Eingaben aus UI / Formular / Chat
    raw_input: str
    form_data: dict[str, Any]
    messages: list[dict[str, str]]


    # Allgemeine Requirement-Extraktion
    requirements: dict[str, Any]
    missing_fields: list[str]
    clarification_question: str

    # Allgemeine Validierung
    general_validation_errors: list[str]

    # Für spätere Konsistenzprüfung zwischen Formular, Freitext und Modellformularen
    consistency_warnings: list[dict[str, Any]]

    # Modell-Auswahl
    selected_model: str
    model_config: dict[str, Any]

    # Modellspezifische Eingaben
    required_model_inputs: list[dict[str, Any]]
    model_specific_inputs: dict[str, Any]
    missing_model_input_fields: list[str]

    # Modellspezifische Validierung
    model_input_validation_errors: list[str]

    # Spätere Generierung / Ergebnisaufbereitung
    ready_for_generation: bool

    # Ergebnis des Code Agents, später optional
    pulp_code: str

    # Ergebnis des Result Nodes
    result: dict[str, Any]


def get_event(state: GraphState) -> WorkflowEvent:
    control = state.get("control", {})
    return control.get("event") or EVENT_NEW_PROBLEM


def get_workflow_status(state: GraphState) -> WorkflowStatus | None:
    control = state.get("control", {})
    return control.get("workflow_status")


def with_control(
    state: GraphState,
    *,
    event: WorkflowEvent | None | object = _CLEAR,
    workflow_status: WorkflowStatus | None = None,
) -> GraphState:
    control = {
        **state.get("control", {}),
    }

    if event is _CLEAR:
        pass
    elif event is None:
        control.pop("event", None)
    else:
        control["event"] = event

    if workflow_status is not None:
        control["workflow_status"] = workflow_status

    return {
        **state,
        "control": control,
    }