import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentic_optimization.agents.modular_model_agent import (
    create_modular_model_agent,
    execute_modular_model_draft,
    update_modular_model_draft,
)


def test_prepares_modular_q_r_model_draft() -> None:
    response = {
        "core_id": "continuous_review_q_r",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten wurden genannt.",
                "sources": ["source_1"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten wurden genannt.",
                "sources": ["source_1"],
            },
            {
                "component_id": "annual_shortage_cost",
                "reason": "Fehlmengenkosten wurden genannt.",
                "sources": ["source_1"],
            },
        ],
        "constraints": [],
    }

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(response)
    )

    agent = create_modular_model_agent(llm)

    draft = agent(
        core_id="continuous_review_q_r",
        requirements={
            "objective": "Jährliche Kosten minimieren",
        },
        products=[
            {
                "product_id": "product_1",
                "forecast": {
                    "expected_value": 100.0,
                    "quantiles": [
                        {
                            "level": 0.1,
                            "value": 70.0,
                        },
                        {
                            "level": 0.5,
                            "value": 100.0,
                        },
                        {
                            "level": 0.9,
                            "value": 140.0,
                        },
                    ],
                    "horizon_days": 30,
                },
                "facts": {
                    "lead_time_days": 10,
                },
            }
        ],
        global_facts={},
        evidence=[
            {
                "source_id": "source_1",
                "content": (
                    "Bestellkosten, Lagerkosten und "
                    "Fehlmengenkosten sollen berücksichtigt "
                    "werden."
                ),
            }
        ],
    )

    assert draft["proposal"] == response
    assert draft["specification"]["core_id"] == (
        "continuous_review_q_r"
    )
    assert draft["specification"]["objective"] == {
        "sense": "minimize",
        "component_ids": [
            "annual_ordering_cost",
            "annual_holding_cost",
            "annual_shortage_cost",
        ],
    }

    assert draft["validation"]["is_valid"] is False
    assert draft["validation"]["errors"] == []

    assert {
        missing_input["field"]["name"]
        for missing_input
        in draft["validation"]["missing_inputs"]
    } == {
        "ordering_cost",
        "holding_cost",
        "shortage_cost",
    }

    with pytest.raises(
        ValueError,
        match="nicht vollständig",
    ):
        execute_modular_model_draft(
            draft,
            confirmed=True,
        )

    values = {
        "ordering_cost": 50.0,
        "holding_cost": 2.0,
        "shortage_cost": 20.0,
    }

    for missing_input in draft[
        "validation"
    ]["missing_inputs"]:
        field_name = missing_input["field"]["name"]

        draft = update_modular_model_draft(
            draft,
            missing_input,
            values[field_name],
        )

    assert draft["validation"]["is_valid"] is True
    assert draft["validation"]["errors"] == []
    assert draft["validation"]["missing_inputs"] == []

    with pytest.raises(
        ValueError,
        match="bestätigt",
    ):
        execute_modular_model_draft(
            draft,
            confirmed=False,
        )

    result = execute_modular_model_draft(
        draft,
        confirmed=True,
    )

    assert result["status"] == "optimal"
    assert result["solution_method"] == "numerical"

    llm.invoke.assert_called_once()