import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentic_optimization.agents.component_coverage_agent import (
    create_component_coverage_agent,
)
from agentic_optimization.agents.component_selection_integrity import (
    create_component_selection_integrity_agent,
)


def create_proposal() -> dict:
    return {
        "core_id": "eoq",
        "selection_status": "complete",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten wurden genannt.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten wurden genannt.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }


def create_agent_response(
    uncovered_requirements: list[dict],
) -> Mock:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "uncovered_requirements": (
                    uncovered_requirements
                ),
            }
        )
    )

    return llm


def call_agent(
    llm: Mock,
    proposal: dict | None = None,
):
    agent = create_component_coverage_agent(llm)

    return agent(
        core_id="eoq",
        requirements={
            "objective": "Gesamtkosten minimieren",
        },
        products=[
            {
                "product_id": "Artikel-A",
                "forecast": {},
                "facts": {},
            },
            {
                "product_id": "Artikel-B",
                "forecast": {},
                "facts": {},
            },
        ],
        evidence=[
            {
                "source_id": "raw_input",
                "content": (
                    "Bestell- und Lagerhaltungskosten sollen "
                    "minimiert werden. Jede Lieferung muss "
                    "an einem Dienstag eintreffen."
                ),
            }
        ],
        proposal=(
            proposal
            if proposal is not None
            else create_proposal()
        ),
    )


def test_adds_overlooked_unsupported_requirement() -> None:
    llm = create_agent_response(
        [
            {
                "description": (
                    "Jede Lieferung muss an einem "
                    "Dienstag eintreffen."
                ),
                "resolution": "unsupported",
                "candidate_component_ids": [],
                "reason": (
                    "Es ist kein Baustein für einen "
                    "bestimmten Liefertag vorhanden."
                ),
                "sources": ["raw_input"],
            }
        ]
    )

    result = call_agent(llm)

    assert result["selection_status"] == "unsupported"
    assert result["constraints"] == []
    assert result["unresolved_requirements"] == [
        {
            "description": (
                "Jede Lieferung muss an einem "
                "Dienstag eintreffen."
            ),
            "resolution": "unsupported",
            "candidate_component_ids": [],
            "reason": (
                "Es ist kein Baustein für einen "
                "bestimmten Liefertag vorhanden."
            ),
            "sources": ["raw_input"],
        }
    ]


def test_preserves_complete_proposal_when_nothing_is_missing(
) -> None:
    proposal = create_proposal()
    original_proposal = deepcopy(proposal)
    llm = create_agent_response([])

    result = call_agent(
        llm,
        proposal,
    )

    assert result == original_proposal
    assert proposal == original_proposal


def test_normalizes_missing_input_before_selected_component_check() -> None:
    finding = {
        "description": "Die Bestellkosten wurden noch nicht eingegeben.",
        "resolution": "clarification",
        "candidate_component_ids": ["annual_ordering_cost"],
        "reason": "Der Zahlenwert fehlt noch.",
        "sources": ["raw_input"],
    }
    llm = create_agent_response([finding])
    normalized = []

    def remove_input_only(proposal):
        normalized.append(proposal)
        result = deepcopy(proposal)
        result["unresolved_requirements"] = []
        result["selection_status"] = "complete"
        return result

    agent = create_component_coverage_agent(llm)
    result = agent(
        core_id="eoq",
        requirements={"objective": "Kosten minimieren"},
        products=[],
        evidence=[{"source_id": "raw_input", "content": "Bestellkosten fehlen."}],
        proposal=create_proposal(),
        normalize_missing_inputs=remove_input_only,
    )
    assert normalized[0]["unresolved_requirements"] == [finding]
    assert result["selection_status"] == "complete"


def test_invalid_coverage_source_is_rejected_before_normalization() -> None:
    finding = {
        "description": "Ein Eingabewert fehlt.",
        "resolution": "clarification",
        "candidate_component_ids": ["annual_ordering_cost"],
        "reason": "Wert fehlt.",
        "sources": ["invented"],
    }
    agent = create_component_coverage_agent(create_agent_response([finding]))
    normalizer = Mock()
    with pytest.raises(ValueError, match="unbekannte Quellen"):
        agent(
            core_id="eoq",
            requirements={},
            products=[],
            evidence=[{"source_id": "raw_input", "content": "Text"}],
            proposal=create_proposal(),
            normalize_missing_inputs=normalizer,
        )
    normalizer.assert_not_called()


def test_registered_missing_input_is_removed_by_integrity_audit() -> None:
    coverage_llm = create_agent_response([{
        "description": "Die Bestellkosten pro Bestellung sind nicht angegeben.",
        "resolution": "clarification",
        "candidate_component_ids": ["annual_ordering_cost"],
        "reason": "Bestellkosten pro Bestellung sind nicht angegeben.",
        "sources": ["raw_input"],
    }])
    integrity_llm = Mock()
    integrity_llm.invoke.return_value = AIMessage(content=json.dumps({
        "rejected_constraints": [],
        "input_only_requirements": [{
            "index": 0,
            "field_names": ["ordering_cost"],
        }],
    }))
    integrity = create_component_selection_integrity_agent(integrity_llm)
    result = create_component_coverage_agent(coverage_llm)(
        core_id="eoq",
        requirements={"objective": "Kosten minimieren"},
        products=[],
        evidence=[{"source_id": "raw_input", "content": "Bestellkosten fehlen."}],
        proposal=create_proposal(),
        normalize_missing_inputs=lambda proposal: integrity(
            proposal=proposal,
            evidence=[{"source_id": "raw_input", "content": "Bestellkosten fehlen."}],
            objective_goal="Kosten minimieren",
        ),
    )
    assert result["selection_status"] == "complete"
    assert result["unresolved_requirements"] == []
    assert integrity_llm.invoke.call_count == 1


@pytest.mark.parametrize(
    "reason",
    [
        (
            "Der fachliche Restwert ist bereits durch den "
            "ausgewählten Baustein 'salvage_revenue' "
            "abgedeckt; keine offene Anforderung."
        ),
        (
            "Die Einperiodigkeit ist bereits durch das "
            "ausgewählte Grundmodell abgedeckt."
        ),
        (
            "Der Vorschlag modelliert dafür bereits einen "
            "Restwerterlös. Ein weiterer fachlicher Bedarf "
            "ist nicht offen."
        ),
    ],
)
def test_ignores_requirement_declared_as_already_covered(
    reason: str,
) -> None:
    proposal = create_proposal()

    llm = create_agent_response(
        [
            {
                "description": (
                    "Eine bereits abgedeckte Anforderung."
                ),
                "resolution": "unsupported",
                "candidate_component_ids": [],
                "reason": reason,
                "sources": ["raw_input"],
            }
        ]
    )

    result = call_agent(
        llm,
        proposal,
    )

    assert result["selection_status"] == "complete"
    assert result["unresolved_requirements"] == []


def test_preserves_requirement_declared_as_not_covered(
) -> None:
    llm = create_agent_response(
        [
            {
                "description": (
                    "Jede Lieferung muss dienstags erfolgen."
                ),
                "resolution": "unsupported",
                "candidate_component_ids": [],
                "reason": (
                    "Die Anforderung ist durch die "
                    "ausgewählten Bausteine nicht abgedeckt."
                ),
                "sources": ["raw_input"],
            }
        ]
    )

    result = call_agent(llm)

    assert result["selection_status"] == "unsupported"
    assert len(result["unresolved_requirements"]) == 1


def test_rejects_unknown_evidence_source() -> None:
    llm = create_agent_response(
        [
            {
                "description": "Unbelegte Anforderung",
                "resolution": "unsupported",
                "candidate_component_ids": [],
                "reason": "Kein Baustein vorhanden.",
                "sources": ["invented_source"],
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="invented_source",
    ):
        call_agent(llm)


def test_sends_catalog_evidence_and_proposal_to_auditor(
) -> None:
    llm = create_agent_response([])

    call_agent(llm)

    messages = llm.invoke.call_args.args[0]
    system_message = messages[0].content
    user_message = messages[1].content

    assert isinstance(system_message, str)
    assert isinstance(user_message, str)
    normalized_system_message = " ".join(
        system_message.split()
    )
    assert (
        "fachliche Vollständigkeit"
        in normalized_system_message
    )
    assert (
        "Grundmodell" in normalized_system_message
    )
    assert (
        "Produktumfang" in normalized_system_message
    )
    assert (
        "nicht lediglich dem ähnlichsten"
        in system_message
    )
    assert (
        "Erzeuge niemals die Kombination"
        in system_message
    )
    assert (
        "candidate_component_ids mindestens eine"
        in normalized_system_message
    )
    assert (
        "Fehlende Eingabewerte oder Constraint-Parameter"
        in system_message
    )
    assert (
        "Der Validator erhebt quantity später in Formular 3"
        in normalized_system_message
    )
    payload = json.loads(user_message)

    assert [
        product["product_id"]
        for product in payload["products"]
    ] == [
        "Artikel-A",
        "Artikel-B",
    ]
    assert (
        "gleichmäßig und planbar"
        in " ".join(
            payload["component_catalog"]["core"][
                "assumptions"
            ]
        )
    )

    assert payload["core_id"] == "eoq"
    assert payload["component_catalog"]["core"][
        "core_id"
    ] == "eoq"
    assert payload["evidence"][0]["source_id"] == (
        "raw_input"
    )
    assert payload["component_selection_proposal"] == (
        create_proposal()
    )
    assert (
        "theoretisch weitere optionale"
        in normalized_system_message
    )
    assert (
        "Gewinn oder Deckungsbeitrag"
        in normalized_system_message
    )
    assert (
        "entgangenen Verkäufen oder entgangenem Umsatz"
        in normalized_system_message
    )
    assert (
        "begründet weder shortage_cost noch eine clarification"
        in normalized_system_message
    )
