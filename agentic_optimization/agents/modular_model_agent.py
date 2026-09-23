"""
Agentische Vorbereitung modularer Optimierungsmodelle.
"""
from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel

from agentic_optimization.agents.component_coverage_agent import (
    create_component_coverage_agent,
)
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
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
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
    products: list[ProductSpecification]
    global_facts: dict[str, Any]
    specification: NotRequired[ModelSpecification]
    validation: NotRequired[
        SpecificationValidationResult
    ]

ModularModelAgent = Callable[..., ModularModelDraft]


PreliminaryComponentSelectionAgent = Callable[
    ...,
    ComponentSelectionProposal,
]


def _build_preliminary_fallback_proposal(
    *,
    core_id: str,
    requirements: dict[str, Any],
    evidence: list[SelectionEvidence],
) -> ComponentSelectionProposal:
    """Baut nur die für eine Produkttabelle sicheren Pflichtspalten."""

    registered_core = CORE_REGISTRY.get(core_id)

    if registered_core is None:
        raise ValueError(
            f"Das Grundmodell '{core_id}' ist nicht registriert."
        )

    objective = requirements.get("objective")
    objective_sense = "minimize"
    component_ids = list(
        registered_core["definition"][
            "required_objective_component_ids"
        ]
    )

    if (
        core_id == "single_period_newsvendor"
        and objective
        in {
            "maximize_profit",
            "Gewinn maximieren",
        }
    ):
        objective_sense = "maximize"
        component_ids = [
            "sales_revenue",
            "purchase_cost",
        ]

    if not component_ids:
        raise ValueError(
            "Für die Produkttabelle konnte keine sichere "
            "Zielfunktionsauswahl abgeleitet werden."
        )

    if not evidence:
        raise ValueError(
            "Für die vorläufige Bausteinauswahl fehlt ein "
            "Quellenbeleg."
        )

    source_id = evidence[0]["source_id"]

    return {
        "core_id": core_id,
        "selection_status": "complete",
        "objective_sense": objective_sense,
        "objective_components": [
            {
                "component_id": component_id,
                "reason": (
                    "Deterministische Rückfallebene für die "
                    "Spalten der Produkttabelle."
                ),
                "sources": [source_id],
            }
            for component_id in component_ids
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }


def create_preliminary_component_selection_agent(
    llm: BaseChatModel,
) -> PreliminaryComponentSelectionAgent:
    """
    Wählt Bausteine vorläufig aus, bevor Produkte existieren.

    Diese Auswahl dient ausschließlich zur Bestimmung der
    Tabellenspalten. Nach der Produkteingabe wird die vollständige
    Modellvorbereitung erneut ausgeführt.
    """

    component_selection_agent = (
        create_component_selection_agent(llm)
    )

    def select_preliminary_components(
        *,
        core_id: str,
        requirements: dict[str, Any],
        global_facts: dict[str, Any],
        evidence: list[SelectionEvidence],
    ) -> ComponentSelectionProposal:
        try:
            return component_selection_agent(
                core_id=core_id,
                requirements=requirements,
                products=[],
                global_facts=global_facts,
                evidence=evidence,
            )
        except (TypeError, ValueError):
            return _build_preliminary_fallback_proposal(
                core_id=core_id,
                requirements=requirements,
                evidence=evidence,
            )

    return select_preliminary_components


def create_modular_model_agent(
    llm: BaseChatModel,
    *,
    enable_coverage_audit: bool = True,
) -> ModularModelAgent:
    """Erzeugt einen Agenten zur Vorbereitung eines Modellentwurfs."""

    fact_extraction_agent = (
        create_modular_fact_extraction_agent(llm)
    )

    component_selection_agent = (
        create_component_selection_agent(llm)
    )

    component_coverage_agent = (
        create_component_coverage_agent(llm)
    )

    def prepare_model(
        *,
        core_id: str,
        requirements: dict[str, Any],
        products: list[ProductSpecification],
        global_facts: dict[str, Any],
        evidence: list[SelectionEvidence],
        preliminary_proposal: (
            ComponentSelectionProposal | None
        ) = None,
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
            locked_objective_proposal=(
                preliminary_proposal
            ),
        )

        if enable_coverage_audit:
            proposal = component_coverage_agent(
                core_id=core_id,
                requirements=requirements,
                products=extracted_products,
                evidence=evidence,
                proposal=proposal,
            )

        if proposal["selection_status"] != "complete":
            return {
                "proposal": proposal,
                "products": extracted_products,
                "global_facts": extracted_global_facts,
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
            "products": extracted_products,
            "global_facts": extracted_global_facts,
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
        "products": updated_specification["products"],
        "global_facts": updated_specification[
            "global_facts"
        ],
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
