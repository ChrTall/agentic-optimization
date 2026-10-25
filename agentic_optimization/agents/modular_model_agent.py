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
from agentic_optimization.agents.component_selection_integrity import (
    create_component_selection_integrity_agent,
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
from agentic_optimization.i18n import t


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


def _build_preliminary_proposal(
    *,
    core_id: str,
    requirements: dict[str, Any],
) -> ComponentSelectionProposal:
    """Baut Formularspalten ausschließlich aus registrierten Pflichtteilen."""

    registered_core = CORE_REGISTRY.get(core_id)

    if registered_core is None:
        raise ValueError(
            t("Das Grundmodell '{core_id}' ist nicht registriert.", core_id=core_id)
        )

    objective = requirements.get("objective")
    objective_sense = registered_core["definition"].get(
        "required_objective_sense"
    )
    component_ids = list(
        registered_core["definition"][
            "required_objective_component_ids"
        ]
    )

    components_by_goal = registered_core["definition"].get(
        "required_objective_components_by_goal", {}
    )
    goal_components = components_by_goal.get(objective, [])
    if goal_components:
        component_ids.extend(goal_components)
        component_ids = list(dict.fromkeys(component_ids))
        if objective_sense is None:
            objective_sense = (
                "maximize" if objective == "maximize_profit" else "minimize"
            )

    if not component_ids:
        raise ValueError(
            t("Für die Produkttabelle konnte keine sichere Zielfunktionsauswahl abgeleitet werden.")
        )

    return {
        "core_id": core_id,
        "selection_status": "complete",
        "objective_sense": objective_sense,
        "objective_components": [
            {
                "component_id": component_id,
                "reason": (
                    t("Vom registrierten Grundmodell für die Produkttabelle benötigt.")
                ),
                "sources": ["core_definition"],
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
    Ermittelt Pflichtbausteine vor der Produkteingabe deterministisch.

    Diese Auswahl dient ausschließlich zur Bestimmung der
    Tabellenspalten. Nach der Produkteingabe wird die vollständige
    Modellvorbereitung erneut ausgeführt.
    """

    def select_preliminary_components(
        *,
        core_id: str,
        requirements: dict[str, Any],
        global_facts: dict[str, Any],
        evidence: list[SelectionEvidence],
    ) -> ComponentSelectionProposal:
        return _build_preliminary_proposal(
            core_id=core_id,
            requirements=requirements,
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
        create_component_selection_agent(
            llm,
            require_explicit_requirement_evidence=True,
        )
    )
    integrity_agent = create_component_selection_integrity_agent(llm)

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
            # Die vorläufige Auswahl dient nur den Eingabefeldern.
            # Nach der Faktenextraktion muss die endgültige Auswahl
            # fehlende Zielfunktionsbausteine ergänzen dürfen.
        )
        proposal = integrity_agent(
            proposal=proposal,
            evidence=evidence,
            objective_goal=requirements.get("objective"),
        )

        if enable_coverage_audit:
            before_coverage = proposal
            proposal = component_coverage_agent(
                core_id=core_id,
                requirements=requirements,
                products=extracted_products,
                evidence=evidence,
                proposal=proposal,
                normalize_missing_inputs=lambda candidate: integrity_agent(
                    proposal=candidate,
                    evidence=evidence,
                    objective_goal=requirements.get("objective"),
                ),
            )
            if (
                proposal["unresolved_requirements"]
                != before_coverage["unresolved_requirements"]
            ):
                proposal = integrity_agent(
                    proposal=proposal,
                    evidence=evidence,
                    objective_goal=requirements.get("objective"),
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
            t("Der Modellentwurf muss vor der Ausführung bestätigt werden.")
        )

    validation = validate_model_specification(
        draft["specification"]
    )

    if not validation["is_valid"]:
        messages = list(validation["errors"])

        messages.extend(
            t("Fehlende Eingabe: {name}", name=missing_input['field']['name'])
            for missing_input in validation["missing_inputs"]
        )

        raise ValueError(
            t("Der Modellentwurf ist nicht vollständig: ")
            + "; ".join(messages)
        )

    return execute_model_specification(
        draft["specification"]
    )
