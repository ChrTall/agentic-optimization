import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agentic_optimization.agents.component_selection_integrity import (
    create_component_selection_integrity_agent,
)


def _proposal():
    return {
        "core_id": "continuous_review_q_r",
        "selection_status": "needs_clarification",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": component_id,
                "reason": "Vom Grundmodell benötigt.",
                "sources": ["core_definition"],
            }
            for component_id in (
                "annual_ordering_cost",
                "annual_holding_cost",
                "annual_shortage_cost",
            )
        ],
        "constraints": [{
            "component_id": "minimum_reorder_point",
            "product_id": "product_1",
            "parameters": {},
            "reason": "Meldebestand soll bestimmt werden.",
            "sources": ["raw_input"],
        }],
        "unresolved_requirements": [{
            "description": "Nachfragespezifikation fehlt.",
            "resolution": "clarification",
            "candidate_component_ids": ["minimum_cycle_service_level"],
            "reason": "Erwartungswert und Quantile fehlen.",
            "sources": ["raw_input"],
        }],
    }


def test_moves_forecast_values_to_form_and_drops_unrequested_bound():
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({
        "rejected_constraints": [{
            "index": 0,
            "reason": "Keine zusätzliche Mindestgrenze verlangt.",
        }],
        "input_only_requirements": [{
            "index": 0,
            "field_names": ["expected_value", "quantiles"],
        }],
    }))

    result = create_component_selection_integrity_agent(llm)(
        proposal=_proposal(),
        evidence=[{
            "source_id": "raw_input",
            "content": (
                "Wir möchten Bestellmenge und Meldebestand bestimmen. "
                "Die Nachfrage ist unsicher."
            ),
        }],
        objective_goal="minimize_cost",
    )

    assert result["selection_status"] == "complete"
    assert result["constraints"] == []
    assert result["unresolved_requirements"] == []
    task = json.loads(llm.invoke.call_args.args[0][1].content)
    assert {field["name"] for field in task[
        "registered_input_fields"
    ]} >= {"expected_value", "quantiles", "horizon_days"}


def test_preserves_explicit_minimum_bound_without_numeric_value():
    proposal = _proposal()
    proposal["unresolved_requirements"] = []
    proposal["selection_status"] = "complete"
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({
        "rejected_constraints": [],
        "input_only_requirements": [],
    }))

    result = create_component_selection_integrity_agent(llm)(
        proposal=proposal,
        evidence=[{
            "source_id": "raw_input",
            "content": "Der Meldebestand muss eine Mindestgrenze haben.",
        }],
        objective_goal="minimize_cost",
    )

    assert result["constraints"] == proposal["constraints"]
    assert result["selection_status"] == "complete"


def test_cannot_remove_unsupported_business_requirement():
    proposal = _proposal()
    proposal["constraints"] = []
    proposal["unresolved_requirements"] = [{
        "description": "Lieferung nur dienstags",
        "resolution": "unsupported",
        "candidate_component_ids": [],
        "reason": "Kein Baustein.",
        "sources": ["raw_input"],
    }]
    proposal["selection_status"] = "unsupported"
    llm = Mock()

    result = create_component_selection_integrity_agent(llm)(
        proposal=proposal,
        evidence=[{
            "source_id": "raw_input",
            "content": "Lieferung nur dienstags.",
        }],
        objective_goal="minimize_cost",
    )

    assert result == proposal
    llm.invoke.assert_not_called()
