from typing import Any, Final, Literal, TypedDict

from agentic_optimization.models.types import (
    StructuredConstraint,
)
from agentic_optimization.models.execution import (
    OptimizationExecutionResult,
)
from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ModelSpecification,
    ProductSpecification,
    SelectionEvidence,
)
from agentic_optimization.models.validator import (
    MissingInput,
    SpecificationValidationResult,
)

# ---------------------------------------------------------------------------
# Workflow-Status
# ---------------------------------------------------------------------------
# Diese Konstanten beschreiben, in welchem fachlichen Zustand sich der Workflow
# gerade befindet. Sie werden nach einzelnen Graph-Nodes gesetzt und später
# für Routing-Entscheidungen im Graph oder in der Streamlit-UI verwendet.

WORKFLOW_GENERAL_REQUIREMENTS_EXTRACTED: Final = "general_requirements_extracted"
WORKFLOW_NEEDS_GENERAL_CLARIFICATION: Final = "needs_general_clarification"
WORKFLOW_GENERAL_INPUTS_VALID: Final = "general_inputs_valid"
WORKFLOW_GENERAL_INPUTS_INVALID: Final = "general_inputs_invalid"
WORKFLOW_MODEL_SELECTED: Final = "model_selected"
WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS: Final = "needs_model_specific_inputs"
WORKFLOW_MODEL_SPECIFIC_INPUTS_SUBMITTED: Final = "model_specific_inputs_submitted"
WORKFLOW_MODEL_SPECIFIC_INPUTS_COMPLETE: Final = "model_specific_inputs_complete"
WORKFLOW_MODEL_INPUTS_VALID: Final = "model_inputs_valid"
WORKFLOW_MODEL_INPUTS_INVALID: Final = "model_inputs_invalid"
WORKFLOW_READY_FOR_RESULT: Final = "ready_for_result"
WORKFLOW_READY_FOR_OPTIMIZATION: Final = "ready_for_optimization"
WORKFLOW_OPTIMIZATION_COMPLETED: Final = "optimization_completed"
WORKFLOW_OPTIMIZATION_FAILED: Final = "optimization_failed"
WORKFLOW_READY_FOR_REVIEW: Final = "ready_for_review"
WORKFLOW_MODEL_UNSUPPORTED: Final = "model_unsupported"


# ---------------------------------------------------------------------------
# Workflow-Events
# ---------------------------------------------------------------------------
# Events beschreiben, wodurch ein Graph-Durchlauf ausgelöst wurde.
# Beispiel:
# - neues Problem wurde eingegeben
# - eine Rückfrage wurde beantwortet
# - modellspezifische Eingaben wurden abgeschickt

EVENT_NEW_PROBLEM: Final = "new_problem"
EVENT_GENERAL_CLARIFICATION_SUBMITTED: Final = "general_clarification_submitted"
EVENT_MODEL_INPUTS_SUBMITTED: Final = "model_inputs_submitted"
EVENT_OPTIMIZATION_REQUESTED: Final = "optimization_requested"
EVENT_MODULAR_MODEL_PREPARATION_REQUESTED: Final = "modular_model_preparation_requested"
EVENT_MODULAR_INPUTS_SUBMITTED: Final = "modular_inputs_submitted"
EVENT_MODULAR_OPTIMIZATION_REQUESTED: Final = "modular_optimization_requested"



# Interner Platzhalter für with_control().
# Damit können wir unterscheiden zwischen:
# - Event unverändert lassen
# - Event bewusst löschen
# - Event neu setzen
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
    "model_inputs_valid",
    "model_inputs_invalid",
    "ready_for_result",
    "ready_for_optimization",
    "optimization_completed",
    "optimization_failed",
    "ready_for_review",
    "model_unsupported",
]

WorkflowEvent = Literal[
    "new_problem",
    "general_clarification_submitted",
    "model_inputs_submitted",
    "optimization_requested",
    "modular_model_preparation_requested",
    "modular_inputs_submitted",
    "modular_optimization_requested",
]

DecisionType = Literal[
    "order_quantity",
    "production_quantity",
    "inventory_level",
]

ObjectiveType = Literal[
    "maximize_profit",
    "minimize_cost",
    "reach_service_level",
]

DemandType = Literal[
    "known",
    "uncertain",
    "historical_estimate",
]

ProductScope = Literal[
    "single_product",
    "multiple_products",
]

PlanningHorizon = Literal[
    "single_period",
    "multiple_periods",
]

ReviewType = Literal[
    "continuous_review",
    "periodic_review",
]


# Konflikte entstehen, wenn Formular und Freitext zwei
# unterschiedliche, grundsätzlich unterstützte Werte liefern.
class RequirementConflict(TypedDict):
    field: str
    form_value: str
    raw_input_value: str
    message: str


# Unsupported Requirements beschreiben ausdrücklich genannte
# Problemmerkmale, für die das System noch keinen erlaubten
# strukturierten Wert beziehungsweise kein Modell unterstützt.
class UnsupportedRequirement(TypedDict):
    field: str
    user_value: str
    message: str


class Requirements(TypedDict, total=False):
    decision: DecisionType | None
    objective: ObjectiveType | None
    demand_type: DemandType | None
    product_scope: ProductScope | None
    planning_horizon: PlanningHorizon | None
    review_type: ReviewType | None
    constraints: list[str]
    raw_description: str


class ControlState(TypedDict, total=False):
    """
    Steuerbereich des GraphState.

    event:
        Beschreibt, wodurch der aktuelle Graph-Durchlauf gestartet wurde.

    workflow_status:
        Beschreibt, in welchem Zustand sich der Workflow nach einem Node befindet.
    """

    event: WorkflowEvent
    workflow_status: WorkflowStatus


# Beschreibende Metadaten der Modellauswahl.
# Die verbindlichen technischen Schlüssel sind
# selected_model_family und selected_model_variant.
class ModelConfig(TypedDict, total=False):
    family_id: str
    family_name: str
    variant_id: str
    variant_name: str
    explanation: str
    objective: Any
    decision: Any


class ModularInputSubmission(TypedDict):
    """Ein vom Nutzer ergänzter Wert für ein fehlendes Feld."""

    missing_input: MissingInput
    value: Any


class ModularModelState(TypedDict, total=False):
    """Zustand der neuen modularen Modellpipeline."""

    core_id: str
    products: list[ProductSpecification]
    global_facts: dict[str, Any]
    evidence: list[SelectionEvidence]
    proposal: ComponentSelectionProposal
    specification: ModelSpecification
    validation: SpecificationValidationResult
    review_confirmed: bool
    optimization_result: OptimizationExecutionResult | None
    pending_input_submissions: list[ModularInputSubmission]

class GraphState(TypedDict, total=False):
    """
    Gemeinsamer Zustand des LangGraph-Workflows.

    Dieser State wird von Knoten zu Knoten weitergegeben.
    Jeder Agent liest relevante Felder aus und ergänzt neue Ergebnisse.
    """

    modular_model: ModularModelState

    # Steuerinformationen für Routing im Graph und in der UI
    control: ControlState

    # Eingaben aus UI / Formular / Chat
    raw_input: str
    form_data: dict[str, Any]
    messages: list[dict[str, str]]

    # Ergebnis der allgemeinen Requirement-Extraktion
    requirements: Requirements
    missing_fields: list[str]
    clarification_question: str

    # Ergebnis der allgemeinen Validierung
    general_validation_errors: list[str]

    # Vorgesehene Felder für zukünftige Hinweise zu auffälligen
    # oder möglicherweise inkonsistenten Eingaben.
    consistency_warnings: list[dict[str, Any]]
    requirement_warnings: list[str]

    # Probleme in den erkannten Anforderungen, die geklärt
    # werden müssen, bevor der Workflow fortgesetzt werden kann.
    requirement_conflicts: list[RequirementConflict]
    unsupported_requirements: list[UnsupportedRequirement]

    # Modellauswahl
    selected_model_family: str | None
    selected_model_variant: str | None
    model_config: ModelConfig
    model_selection_error: str | None

    # Modellspezifische Eingaben:
    # family -> variant -> inputs
    model_specific_inputs: dict[str, dict[str, dict[str, Any]]]

    # Aus dem Freitext erkannte, noch nicht vom Nutzer bestätigte Werte:
    # family -> variant -> input candidates
    model_input_candidates: dict[str, dict[str, dict[str, Any]]]
    model_input_extraction_errors: list[str]

    required_model_inputs: list[dict[str, Any]]
    missing_model_input_fields: list[str]
    model_input_validation_errors: list[str]

    # Das LLM liefert zunächst unbestätigte Kandidaten.
    # Nur die vom Nutzer bestätigten structured_constraints werden
    # validiert und später an den deterministischen Modell-Builder übergeben.
    constraint_candidates: list[StructuredConstraint]
    structured_constraints: list[StructuredConstraint]

    # Im Freitext erkannte Rahmenbedingungen, die von der ausgewählten
    # Modellvariante nicht unterstützt und daher nur informativ angezeigt
    # werden. Sie werden nicht an den Solver übergeben.
    unsupported_constraints: list[str]

    # Fehler bei der Zuordnung von Freitext-Constraints.
    constraint_mapping_errors: list[str]

    # Fehler bei der Validierung aktiver Constraints.
    constraint_validation_errors: list[str]

    # Noch fehlende Parameter aktiver Constraints.
    required_constraint_inputs: list[dict[str, Any]]

    # Optimierungssteuerung
    # Wird erst nach erfolgreicher Input- und Constraintvalidierung gesetzt.
    ready_for_optimization: bool

    # Technische Rohausgabe des variantenspezifischen Optimierers.
    optimization_result: dict[str, Any] | None
    optimization_errors: list[str]

    # Schutz vor einer Optimierung ohne ausdrückliche
    # Bestätigung des Nutzers im Review-Formular.
    review_confirmed: bool

    # Vom Result-Node gebündeltes Ergebnis.
    result: dict[str, Any]


def get_event(state: GraphState) -> WorkflowEvent:
    """
    Gibt das aktuelle Workflow-Event zurück.

    Falls kein Event gesetzt ist, wird standardmäßig
    EVENT_NEW_PROBLEM angenommen.
    """
    control = state.get("control", {})
    return control.get("event") or EVENT_NEW_PROBLEM


def get_workflow_status(state: GraphState) -> WorkflowStatus | None:
    """
    Gibt den aktuellen Workflow-Status zurück.

    Wenn noch kein Status gesetzt wurde, wird None zurückgegeben.
    """
    control = state.get("control", {})
    return control.get("workflow_status")


def with_control(
    state: GraphState,
    *,
    event: WorkflowEvent | None | object = _CLEAR,
    workflow_status: WorkflowStatus | None = None,
) -> GraphState:
    """
    Aktualisiert den control-Bereich des GraphState.

    Der restliche State bleibt unverändert erhalten.
    """

    control = {
        **state.get("control", {}),
    }

    # Event unverändert lassen, löschen oder neu setzen.
    if event is _CLEAR:
        pass
    elif event is None:
        control.pop("event", None)
    else:
        control["event"] = event

    # Workflow-Status nur aktualisieren, wenn ein neuer Status übergeben wurde.
    if workflow_status is not None:
        control["workflow_status"] = workflow_status

    # Gesamten State kopieren und nur den control-Block ersetzen.
    return {
        **state,
        "control": control,
    }


def invalidate_optimization_state(
    state: GraphState,
) -> GraphState:
    """
    Verwirft alle nachgelagerten Zustände, die von Modellinputs
    oder aktiven Constraints abhängen.

    Diese Funktion muss verwendet werden, sobald der Nutzer
    Modellparameter oder strukturierte Constraints ändert.
    """
    updated_state: GraphState = {
        **state,
        "review_confirmed": False,
        "ready_for_optimization": False,
        "optimization_result": None,
        "optimization_errors": [],
        "result": {},
    }

    if "modular_model" in state:
        updated_state["modular_model"] = {
            **state["modular_model"],
            "review_confirmed": False,
            "optimization_result": None,
        }

    return updated_state
