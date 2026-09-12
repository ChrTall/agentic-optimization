"""
LangGraph-kompatible Knoten der modularen Modellpipeline.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from agentic_optimization.agents.modular_model_agent import (
    ModularModelDraft,
    create_modular_model_agent,
    execute_modular_model_draft,
    update_modular_model_draft,
)
from agentic_optimization.state import (
    WORKFLOW_MODEL_INPUTS_INVALID,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_READY_FOR_REVIEW,
    WORKFLOW_OPTIMIZATION_COMPLETED,
    WORKFLOW_OPTIMIZATION_FAILED,
    GraphState,
    with_control,
)
from agentic_optimization.models.validator import (
    validate_model_specification,
)


def create_modular_model_preparation_node(
    llm: BaseChatModel,
) -> Callable[[GraphState], GraphState]:
    """Erzeugt den Knoten für die agentische Modellvorbereitung."""

    prepare_model = create_modular_model_agent(llm)

    def prepare_modular_model_node(
        state: GraphState,
    ) -> GraphState:
        modular_state = state.get("modular_model")

        if modular_state is None:
            raise ValueError(
                "Der modulare Modellzustand fehlt."
            )

        core_id = modular_state.get("core_id")
        products = modular_state.get("products")

        if not core_id:
            raise ValueError(
                "Das ausgewählte Grundmodell fehlt."
            )

        if products is None:
            raise ValueError(
                "Die Produktdaten für das modulare "
                "Modell fehlen."
            )

        draft = prepare_model(
            core_id=core_id,
            requirements=dict(
                state.get("requirements", {})
            ),
            products=products,
            global_facts=modular_state.get(
                "global_facts",
                {},
            ),
            evidence=modular_state.get(
                "evidence",
                [],
            ),
        )

        validation = draft["validation"]

        if validation["is_valid"]:
            workflow_status = WORKFLOW_READY_FOR_REVIEW
        elif (
            validation["missing_inputs"]
            and not validation["errors"]
        ):
            workflow_status = (
                WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS
            )
        else:
            workflow_status = WORKFLOW_MODEL_INPUTS_INVALID

        updated_state: GraphState = {
            **state,
            "modular_model": {
                **modular_state,
                "proposal": draft["proposal"],
                "specification": draft["specification"],
                "validation": validation,
                "review_confirmed": False,
                "optimization_result": None,
            },
        }

        return with_control(
            updated_state,
            event=None,
            workflow_status=workflow_status,
        )

    return prepare_modular_model_node


def apply_modular_model_inputs_node(
    state: GraphState,
) -> GraphState:
    """Übernimmt eingereichte Werte in den modularen Entwurf."""

    modular_state = state.get("modular_model")

    if modular_state is None:
        raise ValueError(
            "Der modulare Modellzustand fehlt."
        )

    proposal = modular_state.get("proposal")
    specification = modular_state.get("specification")

    if proposal is None or specification is None:
        raise ValueError(
            "Es liegt noch kein modularer Modellentwurf vor."
        )

    draft: ModularModelDraft = {
        "proposal": proposal,
        "specification": specification,
        "validation": validate_model_specification(
            specification
        ),
    }

    submissions = modular_state.get(
        "pending_input_submissions",
        [],
    )

    for submission in submissions:
        submitted_missing_input = submission[
            "missing_input"
        ]

        current_missing_input = next(
            (
                candidate
                for candidate
                in draft["validation"]["missing_inputs"]
                if candidate == submitted_missing_input
            ),
            None,
        )

        if current_missing_input is None:
            field_name = submitted_missing_input[
                "field"
            ]["name"]

            raise ValueError(
                f"Die Eingabe für '{field_name}' ist "
                "nicht mehr offen oder gehört nicht zum "
                "aktuellen Modellentwurf."
            )

        draft = update_modular_model_draft(
            draft,
            current_missing_input,
            submission["value"],
        )

    validation = draft["validation"]

    if validation["is_valid"]:
        workflow_status = WORKFLOW_READY_FOR_REVIEW
    elif (
        validation["missing_inputs"]
        and not validation["errors"]
    ):
        workflow_status = (
            WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS
        )
    else:
        workflow_status = WORKFLOW_MODEL_INPUTS_INVALID

    updated_state: GraphState = {
        **state,
        "modular_model": {
            **modular_state,
            "specification": draft["specification"],
            "validation": validation,
            "pending_input_submissions": [],
            "review_confirmed": False,
            "optimization_result": None,
        },
    }

    return with_control(
        updated_state,
        event=None,
        workflow_status=workflow_status,
    )


def execute_modular_model_node(
    state: GraphState,
) -> GraphState:
    """Führt einen bestätigten modularen Modellentwurf aus."""

    modular_state = state.get("modular_model")

    if modular_state is None:
        raise ValueError(
            "Der modulare Modellzustand fehlt."
        )

    proposal = modular_state.get("proposal")
    specification = modular_state.get("specification")

    if proposal is None or specification is None:
        raise ValueError(
            "Es liegt noch kein modularer Modellentwurf vor."
        )

    validation = validate_model_specification(
        specification
    )

    draft: ModularModelDraft = {
        "proposal": proposal,
        "specification": specification,
        "validation": validation,
    }

    optimization_result = execute_modular_model_draft(
        draft,
        confirmed=modular_state.get(
            "review_confirmed",
            False,
        ),
    )

    if optimization_result["status"] == "failed":
        workflow_status = WORKFLOW_OPTIMIZATION_FAILED
    else:
        workflow_status = WORKFLOW_OPTIMIZATION_COMPLETED

    updated_state: GraphState = {
        **state,
        "modular_model": {
            **modular_state,
            "validation": validation,
            "optimization_result": optimization_result,
        },
    }

    return with_control(
        updated_state,
        event=None,
        workflow_status=workflow_status,
    )