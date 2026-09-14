"""
Agentische Vorbereitung modularer Optimierungsmodelle.
"""
from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel

from agentic_optimization.agents.component_selection_agent import (
    create_component_selection_agent,
)
from agentic_optimization.agents.modular_fact_extraction_agent import (
    create_modular_fact_extraction_agent,
)
from agentic_optimization.models.execution import (
    OptimizationExecutionResult,
    execute_model_specification,
)
from agentic_optimization.models.proposal_mapper import (
    build_specification_from_proposal,
)
from agentic_optimization.models.specification import (
    ComponentSelectionProposal,
    ModelSpecification,
    ProductSpecification,
    SelectionEvidence,
)
from agentic_optimization.models.specification_updates import (
    apply_modular_input_value,
)
from agentic_optimization.models.validator import (
    MissingInput,
    SpecificationValidationResult,
    validate_model_specification,
)


class ModularModelDraft(TypedDict):
    """Ergebnis der agentischen Modellvorbereitung."""

    proposal: ComponentSelectionProposal
    specification: NotRequired[ModelSpecification]
    validation: NotRequired[
        SpecificationValidationResult
    ]


ModularModelAgent = Callable[..., ModularModelDraft]


def create_modular_model_agent(
    llm: BaseChatModel,
) -> ModularModelAgent:
    """Erzeugt einen Agenten zur Vorbereitung eines Modellentwurfs."""

    fact_extraction_agent = (
        create_modular_fact_extraction_agent(llm)
    )

    component_selection_agent = (
        create_component_selection_agent(llm)
    )

    def prepare_model(
        *,
        core_id: str,
        requirements: dict[str, Any],
        products: list[ProductSpecification],
        global_facts: dict[str, Any],
        evidence: list[SelectionEvidence],
    ) -> ModularModelDraft:
        extracted_products, extracted_global_facts = (
            fact_extraction_agent(
                core_id=core_id,
                requirements=requirements,
                products=products,
                global_facts=global_facts,
                evidence=evidence,
            )
        )

        proposal = component_selection_agent(
            core_id=core_id,
            requirements=requirements,
            products=extracted_products,
            global_facts=extracted_global_facts,
            evidence=evidence,
        )

        if proposal["selection_status"] != "complete":
            return {
                "proposal": proposal,
            }

        specification = build_specification_from_proposal(
            proposal,
            extracted_products,
            extracted_global_facts,
        )

        validation = validate_model_specification(
            specification
        )

        return {
            "proposal": proposal,
            "specification": specification,
            "validation": validation,
        }

    return prepare_model


def update_modular_model_draft(
    draft: ModularModelDraft,
    missing_input: MissingInput,
    value: Any,
) -> ModularModelDraft:
    """Ergänzt einen Wert und validiert den Modellentwurf erneut."""

    updated_specification = apply_modular_input_value(
        draft["specification"],
        missing_input,
        value,
    )

    return {
        "proposal": draft["proposal"],
        "specification": updated_specification,
        "validation": validate_model_specification(
            updated_specification
        ),
    }


def execute_modular_model_draft(
    draft: ModularModelDraft,
    *,
    confirmed: bool,
) -> OptimizationExecutionResult:
    """Führt ausschließlich einen bestätigten, gültigen Entwurf aus."""

    if not confirmed:
        raise ValueError(
            "Der Modellentwurf muss vor der Ausführung "
            "bestätigt werden."
        )

    validation = validate_model_specification(
        draft["specification"]
    )

    if not validation["is_valid"]:
        messages = list(validation["errors"])

        messages.extend(
            "Fehlende Eingabe: "
            f"{missing_input['field']['name']}"
            for missing_input in validation["missing_inputs"]
        )

        raise ValueError(
            "Der Modellentwurf ist nicht vollständig: "
            + "; ".join(messages)
        )

    return execute_model_specification(
        draft["specification"]
    )