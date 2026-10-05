"""
LangGraph-kompatible Knoten der modularen Modellpipeline.
"""

import logging
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from agentic_optimization.agents.modular_model_agent import (
    ModularModelDraft,
    create_modular_model_agent,
    create_preliminary_component_selection_agent,
    execute_modular_model_draft,
    update_modular_model_draft,
)
from agentic_optimization.agents.modular_result_explanation_agent import (
    create_modular_result_explanation_agent,
)
from agentic_optimization.agents.modular_review_explanation_agent import (
    create_modular_review_explanation_agent,
)
from agentic_optimization.models.core_applicability import (
    CONDITION_LABELS,
)
from agentic_optimization.models.proposal_mapper import (
    build_specification_from_proposal,
    exclude_unsupported_requirements,
)
from agentic_optimization.models.validator import (
    validate_model_specification,
)
from agentic_optimization.state import (
    WORKFLOW_MODEL_INPUTS_INVALID,
    WORKFLOW_MODEL_PREPARATION_FAILED,
    WORKFLOW_MODEL_UNSUPPORTED,
    WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_OPTIMIZATION_COMPLETED,
    WORKFLOW_OPTIMIZATION_FAILED,
    WORKFLOW_PRODUCT_TABLE_READY,
    WORKFLOW_READY_FOR_REVIEW,
    GraphState,
    with_control,
)
from agentic_optimization.ui.modular_input_builder import (
    build_modular_input_layout,
)
from agentic_optimization.ui.modular_result_builder import (
    build_default_result_explanation,
    build_modular_result_data,
)
from agentic_optimization.ui.modular_review_builder import (
    build_default_review_explanation,
    build_modular_review_data,
)

LOGGER = logging.getLogger(__name__)


def create_modular_product_schema_node(
    llm: BaseChatModel,
) -> Callable[[GraphState], GraphState]:
    """
    Wählt vorläufige Bausteine zur Bestimmung der Tabellenspalten.
    """

    select_components = (
        create_preliminary_component_selection_agent(llm)
    )

    def prepare_product_schema_node(
        state: GraphState,
    ) -> GraphState:
        modular_state = state.get("modular_model")

        if modular_state is None:
            raise ValueError(
                "Der modulare Modellzustand fehlt."
            )

        core_id = modular_state.get("core_id")

        if not core_id:
            raise ValueError(
                "Das ausgewählte Grundmodell fehlt."
            )

        try:
            proposal = select_components(
                core_id=core_id,
                requirements=dict(
                    state.get("requirements", {})
                ),
                global_facts=modular_state.get(
                    "global_facts",
                    {},
                ),
                evidence=modular_state.get(
                    "evidence",
                    [],
                ),
            )
        except (TypeError, ValueError) as error:
            LOGGER.exception(
                "Die vorläufige Bausteinauswahl ist "
                "fehlgeschlagen: %s",
                error,
            )

            return with_control(
                {
                    **state,
                    "modular_model": {
                        **modular_state,
                        "preparation_error": (
                            "Die benötigten Tabellenspalten "
                            "konnten nicht zuverlässig bestimmt "
                            "werden. Bitte starten Sie die "
                            "Analyse erneut."
                        ),
                    },
                },
                event=None,
                workflow_status=(
                    WORKFLOW_MODEL_PREPARATION_FAILED
                ),
            )

        return with_control(
            {
                **state,
                "modular_model": {
                    **modular_state,
                    "preliminary_proposal": proposal,
                    "preparation_error": None,
                },
            },
            event=None,
            workflow_status=WORKFLOW_PRODUCT_TABLE_READY,
        )

    return prepare_product_schema_node


def create_modular_review_explanation_node(
    llm: BaseChatModel,
) -> Callable[[GraphState], GraphState]:
    """
    Erzeugt und speichert die Erklärung für Formular 4.

    Bei einem Fehler des Sprachmodells wird ein
    deterministischer Standardtext gespeichert.
    """

    explanation_agent = (
        create_modular_review_explanation_agent(llm)
    )

    def explain_modular_review_node(
        state: GraphState,
    ) -> GraphState:
        modular_state = state.get("modular_model")

        if modular_state is None:
            raise ValueError(
                "Der modulare Modellzustand fehlt."
            )

        existing_explanation = modular_state.get(
            "review_explanation"
        )

        if (
            isinstance(existing_explanation, str)
            and existing_explanation.strip()
        ):
            return with_control(
                state,
                event=None,
                workflow_status=WORKFLOW_READY_FOR_REVIEW,
            )

        proposal = modular_state.get("proposal")
        specification = modular_state.get(
            "specification"
        )

        if proposal is None or specification is None:
            raise ValueError(
                "Für die Erklärung fehlt der vollständige "
                "modulare Modellentwurf."
            )

        review_data = build_modular_review_data(
            proposal,
            specification,
        )
        pending_assumptions = [
            CONDITION_LABELS[condition_id]
            for condition_id in modular_state.get(
                "pending_core_assumptions", []
            )
        ]
        review_data["pending_core_assumptions"] = (
            pending_assumptions
        )
        fallback_explanation = (
            build_default_review_explanation(
                review_data,
                pending_core_assumptions=(
                    pending_assumptions
                ),
            )
        )

        try:
            explanation = explanation_agent(
                review_data=dict(review_data)
            )
            explanation_source = "llm"
        except Exception as error:  # noqa: BLE001
            LOGGER.exception(
                "Die automatische Review-Erklärung ist "
                "fehlgeschlagen. Der Standardtext wird "
                "verwendet: %s",
                error,
            )
            explanation = fallback_explanation
            explanation_source = "fallback"

        return with_control(
            {
                **state,
                "modular_model": {
                    **modular_state,
                    "review_explanation": explanation,
                    "review_explanation_source": (
                        explanation_source
                    ),
                },
            },
            event=None,
            workflow_status=WORKFLOW_READY_FOR_REVIEW,
        )

    return explain_modular_review_node


def create_modular_result_explanation_node(
    llm: BaseChatModel,
) -> Callable[[GraphState], GraphState]:
    """
    Erzeugt und speichert die Erklärung für Formular 5.

    Bei einem Fehler des Sprachmodells wird ein
    deterministischer Standardtext gespeichert.
    """

    explanation_agent = (
        create_modular_result_explanation_agent(llm)
    )

    def explain_modular_result_node(
        state: GraphState,
    ) -> GraphState:
        modular_state = state.get("modular_model")

        if modular_state is None:
            raise ValueError(
                "Der modulare Modellzustand fehlt."
            )

        existing_explanation = modular_state.get(
            "result_explanation"
        )

        if (
            isinstance(existing_explanation, str)
            and existing_explanation.strip()
        ):
            return with_control(
                state,
                event=None,
            )

        proposal = modular_state.get("proposal")
        specification = modular_state.get(
            "specification"
        )
        optimization_result = modular_state.get(
            "optimization_result"
        )

        if (
            proposal is None
            or specification is None
            or optimization_result is None
        ):
            raise ValueError(
                "Für die Ergebnis-Erklärung fehlen "
                "Modell oder Optimierungsergebnis."
            )

        review_data = build_modular_review_data(
            proposal,
            specification,
        )
        result_data = build_modular_result_data(
            specification,
            optimization_result,
        )
        fallback_explanation = (
            build_default_result_explanation(
                result_data
            )
        )

        try:
            explanation = explanation_agent(
                review_data=dict(review_data),
                result_data=dict(result_data),
            )
            explanation_source = "llm"
        except Exception as error:  # noqa: BLE001
            LOGGER.exception(
                "Die automatische Ergebnis-Erklärung ist "
                "fehlgeschlagen. Der Standardtext wird "
                "verwendet: %s",
                error,
            )
            explanation = fallback_explanation
            explanation_source = "fallback"

        return with_control(
            {
                **state,
                "modular_model": {
                    **modular_state,
                    "result_explanation": explanation,
                    "result_explanation_source": (
                        explanation_source
                    ),
                },
            },
            event=None,
        )

    return explain_modular_result_node


def create_modular_model_preparation_node(
    llm: BaseChatModel,
) -> Callable[[GraphState], GraphState]:
    """Erzeugt den Knoten für die agentische Modellvorbereitung."""

    prepare_model = create_modular_model_agent(
        llm,
        enable_coverage_audit=False,
    )

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

        try:
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
                preliminary_proposal=(
                    modular_state.get(
                        "preliminary_proposal"
                    )
                ),
            )
        except (TypeError, ValueError) as error:
            LOGGER.exception(
                "Die modulare Modellvorbereitung ist "
                "fehlgeschlagen: %s",
                error,
            )
            preparation_error = (
                "Die automatische Modellzusammenstellung "
                "konnte nicht zuverlässig abgeschlossen "
                "werden. Bitte starten Sie die Analyse "
                "erneut."
            )

            updated_modular_state = {
                **modular_state,
                "preparation_error": preparation_error,
                "review_confirmed": False,
                "optimization_result": None,
            }

            for field_name in (
                "proposal",
                "specification",
                "validation",
            ):
                updated_modular_state.pop(
                    field_name,
                    None,
                )

            return with_control(
                {
                    **state,
                    "modular_model": (
                        updated_modular_state
                    ),
                },
                event=None,
                workflow_status=(
                    WORKFLOW_MODEL_PREPARATION_FAILED
                ),
            )

        proposal = draft["proposal"]
        extracted_products = draft["products"]
        extracted_global_facts = draft["global_facts"]
        selection_status = proposal[
            "selection_status"
        ]

        if selection_status != "complete":
            if (
                selection_status
                == "needs_clarification"
            ):
                workflow_status = (
                    WORKFLOW_NEEDS_GENERAL_CLARIFICATION
                )
            else:
                workflow_status = (
                    WORKFLOW_MODEL_UNSUPPORTED
                )

            updated_modular_state = {
                **modular_state,
                "products": extracted_products,
                "global_facts": extracted_global_facts,
                "proposal": proposal,
                "review_confirmed": False,
                "optimization_result": None,
                "preparation_error": None,
            }

            updated_modular_state.pop(
                "specification",
                None,
            )
            updated_modular_state.pop(
                "validation",
                None,
            )

            return with_control(
                {
                    **state,
                    "modular_model": (
                        updated_modular_state
                    ),
                },
                event=None,
                workflow_status=workflow_status,
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
                "products": extracted_products,
                "global_facts": extracted_global_facts,
                "proposal": proposal,
                "specification": draft["specification"],
                "validation": validation,
                "review_confirmed": False,
                "optimization_result": None,
                "review_explanation": "",
                "review_explanation_source": "",
                "preparation_error": None,
            },
        }

        return with_control(
            updated_state,
            event=None,
            workflow_status=workflow_status,
        )

    return prepare_modular_model_node


def confirm_modular_requirement_exclusions_node(
    state: GraphState,
) -> GraphState:
    """
    Setzt die Modellvorbereitung nach einer ausdrücklichen
    Bestätigung nicht unterstützter Anforderungen fort.
    """

    modular_state = state.get("modular_model")

    if modular_state is None:
        raise ValueError(
            "Der modulare Modellzustand fehlt."
        )

    proposal = modular_state.get("proposal")

    if proposal is None:
        raise ValueError(
            "Der zu prüfende Bausteinvorschlag fehlt."
        )

    updated_proposal, new_exclusions = (
        exclude_unsupported_requirements(
            proposal,
            confirmation_source=(
                "modular_unsupported_review"
            ),
        )
    )

    excluded_requirements = [
        *modular_state.get(
            "excluded_requirements",
            [],
        ),
        *new_exclusions,
    ]

    if (
        updated_proposal["selection_status"]
        != "complete"
    ):
        updated_modular_state = {
            **modular_state,
            "proposal": updated_proposal,
            "excluded_requirements": (
                excluded_requirements
            ),
            "review_confirmed": False,
            "optimization_result": None,
        }
        updated_modular_state.pop(
            "specification",
            None,
        )
        updated_modular_state.pop(
            "validation",
            None,
        )

        return with_control(
            {
                **state,
                "modular_model": updated_modular_state,
            },
            event=None,
            workflow_status=(
                WORKFLOW_NEEDS_GENERAL_CLARIFICATION
            ),
        )

    specification = build_specification_from_proposal(
        updated_proposal,
        modular_state.get("products", []),
        modular_state.get("global_facts", {}),
        excluded_requirements=excluded_requirements,
    )
    validation = validate_model_specification(
        specification
    )

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

    return with_control(
        {
            **state,
            "modular_model": {
                **modular_state,
                "proposal": updated_proposal,
                "specification": specification,
                "validation": validation,
                "excluded_requirements": (
                    excluded_requirements
                ),
                "review_confirmed": False,
                "optimization_result": None,
                "review_explanation": "",
                "review_explanation_source": "",
                "preparation_error": None,
            },
        },
        event=None,
        workflow_status=workflow_status,
    )


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

    input_layout = build_modular_input_layout(specification)
    allowed_inputs = [
        *(
            item
            for group in input_layout["product_groups"]
            for item in [
                *group["forecast_inputs"],
                *group["fact_inputs"],
            ]
        ),
        *input_layout["global_inputs"],
        *(
            item
            for group in input_layout["constraint_groups"]
            for item in group["inputs"]
        ),
    ]

    for submission in submissions:
        submitted_missing_input = submission[
            "missing_input"
        ]

        current_missing_input = next(
            (
                candidate
                for candidate
                in allowed_inputs
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
            "review_explanation": "",
            "review_explanation_source": "",
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

    if (
        modular_state.get("pending_core_assumptions")
        and not modular_state.get("core_assumptions_confirmed", False)
    ):
        raise ValueError(
            "Offene Voraussetzungen des Grundmodells müssen "
            "vor der Berechnung ausdrücklich bestätigt werden."
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
            "result_explanation": "",
            "result_explanation_source": "",
        },
    }

    return with_control(
        updated_state,
        event=None,
        workflow_status=workflow_status,
    )
