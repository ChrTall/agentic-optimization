from typing import Any, Final, Literal, TypedDict

from agentic_optimization.models.execution import (
    OptimizationExecutionResult,
)
from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ModelSpecification,
    ProductSpecification,
    SelectionEvidence,
    ExcludedRequirement,
)
from agentic_optimization.models.validator import (
    MissingInput,
    SpecificationValidationResult,
)
from agentic_optimization.models.core_selector import (
    CoreSelectionDecision,
)
from agentic_optimization.models.core_applicability import (
    ApplicabilityPlan,
    Finding,
    UserConditionAnswer,
)
from agentic_optimization.models.core_candidate_selection import (
    CoreCandidateProposal,
)
from agentic_optimization.models.core_question_planner import (
    CoreQuestionPlan,
)
from agentic_optimization.models.problem_profile import (
    BusinessProblemProfile,
)

# ---------------------------------------------------------------------------
# Workflow-Status
# ---------------------------------------------------------------------------
# Diese Konstanten beschreiben, in welchem fachlichen Zustand sich der Workflow
# gerade befindet. Sie werden nach einzelnen Graph-Nodes gesetzt und später
# für Routing-Entscheidungen im Graph oder in der Streamlit-UI verwendet.

WORKFLOW_NEEDS_GENERAL_CLARIFICATION: Final = "needs_general_clarification"
WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS: Final = "needs_model_specific_inputs"
WORKFLOW_MODEL_INPUTS_INVALID: Final = "model_inputs_invalid"
WORKFLOW_OPTIMIZATION_COMPLETED: Final = "optimization_completed"
WORKFLOW_OPTIMIZATION_FAILED: Final = "optimization_failed"
WORKFLOW_READY_FOR_REVIEW: Final = "ready_for_review"
WORKFLOW_MODEL_UNSUPPORTED: Final = "model_unsupported"
WORKFLOW_MODEL_PREPARATION_FAILED: Final = "model_preparation_failed"
WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT: Final = "needs_business_profile_input"
WORKFLOW_BUSINESS_CORE_SELECTED: Final = "business_core_selected"
WORKFLOW_PRODUCT_TABLE_READY: Final = "product_table_ready"
WORKFLOW_BUSINESS_CASE_UNSUPPORTED: Final = "business_case_unsupported"
WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED: Final = "business_profile_analysis_failed"



# ---------------------------------------------------------------------------
# Workflow-Events
# ---------------------------------------------------------------------------
# Events beschreiben, wodurch ein Graph-Durchlauf ausgelöst wurde.
# Beispiel:
# - neues Problem wurde eingegeben
# - eine Rückfrage wurde beantwortet
# - modellspezifische Eingaben wurden abgeschickt

EVENT_MODULAR_MODEL_PREPARATION_REQUESTED: Final = "modular_model_preparation_requested"
EVENT_MODULAR_PRODUCT_SCHEMA_REQUESTED: Final = "modular_product_schema_requested"
EVENT_MODULAR_REVIEW_EXPLANATION_REQUESTED: Final = "modular_review_explanation_requested"
EVENT_MODULAR_RESULT_EXPLANATION_REQUESTED: Final = "modular_result_explanation_requested"
EVENT_MODULAR_INPUTS_SUBMITTED: Final = "modular_inputs_submitted"
EVENT_MODULAR_OPTIMIZATION_REQUESTED: Final = "modular_optimization_requested"
EVENT_BUSINESS_INTAKE_REQUESTED: Final = "business_intake_requested"
EVENT_MODULAR_REQUIREMENT_EXCLUSIONS_CONFIRMED: Final = "modular_requirement_exclusions_confirmed"





# Interner Platzhalter für with_control().
# Damit können wir unterscheiden zwischen:
# - Event unverändert lassen
# - Event bewusst löschen
# - Event neu setzen
_CLEAR = object()

WorkflowStatus = Literal[
    "needs_general_clarification",
    "needs_model_specific_inputs",
    "model_inputs_invalid",
    "optimization_completed",
    "optimization_failed",
    "ready_for_review",
    "model_unsupported",
    "model_preparation_failed",
    "needs_business_profile_input",
    "business_core_selected",
    "product_table_ready",
    "business_case_unsupported",
    "business_profile_analysis_failed",
    "modular_requirement_exclusions_confirmed",
]

WorkflowEvent = Literal[
    "modular_model_preparation_requested",
    "modular_product_schema_requested",
    "modular_review_explanation_requested",
    "modular_result_explanation_requested",
    "modular_inputs_submitted",
    "modular_optimization_requested",
    "business_intake_requested",
    "modular_requirement_exclusions_confirmed",
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
    "repeated_cycles",
    "multiple_periods",
]

ReviewType = Literal[
    "continuous_review",
    "periodic_review",
]


class Requirements(TypedDict, total=False):
    decision: DecisionType | None
    objective: ObjectiveType | None
    demand_type: DemandType | None
    product_scope: ProductScope | None
    planning_horizon: PlanningHorizon | None
    review_type: ReviewType | None
    constraints: list[str]
    raw_description: str
    explicit_model_conditions: dict[str, str]


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


class BusinessQuestionAnswer(TypedDict):
    """Antwort auf eine aktuell offene Geschäftsfrage."""

    question_id: str
    option_id: str
    source_id: str

class BusinessIntakeState(TypedDict, total=False):
    """Zustand des modellneutralen Geschäftseinstiegs."""

    profile: BusinessProblemProfile
    asked_question_ids: list[str]
    pending_question_ids: list[str]
    core_selection: CoreSelectionDecision
    analysis_error: str | None
    pending_answers: list[BusinessQuestionAnswer]
    analyzed_raw_input: str
    applicability_plan: ApplicabilityPlan
    applicability_text_findings: dict[str, Finding]
    applicability_text_evidence: list[dict[str, str]]
    applicability_analyzed_raw_input: str
    applicability_core_id: str
    applicability_user_answers: dict[str, UserConditionAnswer]
    candidate_proposal: CoreCandidateProposal
    candidate_analyzed_raw_input: str | None
    candidate_analyzed_answer_context: str
    chosen_core_id: str | None
    core_question_plan: CoreQuestionPlan | None
    core_question_prompt: str | None


class ModularInputSubmission(TypedDict):
    """Ein vom Nutzer ergänzter Wert für ein fehlendes Feld."""

    missing_input: MissingInput
    value: Any


class ModularModelState(TypedDict, total=False):
    """Zustand der neuen modularen Modellpipeline."""

    core_id: str
    pending_core_assumptions: list[str]
    core_assumption_evidence: list[dict[str, str]]
    core_assumptions_confirmed: bool
    products: list[ProductSpecification]
    global_facts: dict[str, Any]
    evidence: list[SelectionEvidence]
    proposal: ComponentSelectionProposal
    preliminary_proposal: ComponentSelectionProposal
    specification: ModelSpecification
    validation: SpecificationValidationResult
    review_confirmed: bool
    optimization_result: OptimizationExecutionResult | None
    pending_input_submissions: list[ModularInputSubmission]
    preparation_error: str | None
    excluded_requirements: list[ExcludedRequirement]
    review_explanation: str
    review_explanation_source: str
    result_explanation: str
    result_explanation_source: str

class GraphState(TypedDict, total=False):
    """
    Gemeinsamer Zustand des LangGraph-Workflows.

    Dieser State wird von Knoten zu Knoten weitergegeben.
    Jeder Agent liest relevante Felder aus und ergänzt neue Ergebnisse.
    """
    business_intake: BusinessIntakeState
    modular_model: ModularModelState
    control: ControlState
    raw_input: str
    requirements: Requirements


def get_event(state: GraphState) -> WorkflowEvent:
    """
    Gibt das aktuelle Workflow-Event zurück.

    Ein Graph-Durchlauf muss immer durch ein explizites
    Ereignis ausgelöst werden.
    """
    control = state.get("control", {})
    event = control.get("event")

    if event is None:
        raise ValueError(
            "Für den Workflow wurde kein Event angegeben."
        )

    return event


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
    updated_state: GraphState = {**state}

    if "modular_model" in state:
        updated_state["modular_model"] = {
            **state["modular_model"],
            "review_confirmed": False,
            "optimization_result": None,
            "review_explanation": "",
            "review_explanation_source": "",
            "result_explanation": "",
            "result_explanation_source": "",
        }
    return updated_state
