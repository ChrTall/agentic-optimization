import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agentic_optimization.graph import build_graph
from agentic_optimization.state import (
    EVENT_NEW_PROBLEM,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    get_workflow_status,
)


Q_R_DESCRIPTION = """
Ein Campusladen minimiert mit laufender Bestandsüberwachung seine Kosten.
Die jährliche Nachfrage beträgt 7.300 T-Shirts. Jede Bestellung kostet
250 Euro, die Lagerhaltung 50 Euro je T-Shirt und Jahr und eine Fehlmenge
10 Euro je T-Shirt. Die mittlere Lieferzeitnachfrage beträgt 60 T-Shirts
mit einer Standardabweichung von 6,928 T-Shirts. Die Bestellmenge liegt
zwischen 250 und 300, der Meldebestand zwischen 60 und 80 T-Shirts.
Q plus R darf höchstens 400 T-Shirts betragen.
""".strip()


def test_new_problem_prefills_q_r_model_inputs_and_constraints() -> None:
    requirement_response = {
        "decision": "order_quantity",
        "objective": "minimize_cost",
        "demand_type": "uncertain",
        "product_scope": "single_product",
        "planning_horizon": "multiple_periods",
        "review_type": "continuous_review",
        "constraints": [
            "Die Bestellmenge liegt zwischen 250 und 300.",
            "Der Meldebestand liegt zwischen 60 und 80.",
            "Q plus R darf höchstens 400 betragen.",
        ],
        "unsupported_requirements": [],
    }
    constraint_response = {
        "structured_constraints": [
            {
                "type": "minimum_order_quantity",
                "parameters": {"quantity": 250},
                "source_ids": ["constraint_0"],
            },
            {
                "type": "maximum_order_quantity",
                "parameters": {"quantity": 300},
                "source_ids": ["constraint_0"],
            },
            {
                "type": "minimum_reorder_point",
                "parameters": {"reorder_point": 60},
                "source_ids": ["constraint_1"],
            },
            {
                "type": "maximum_reorder_point",
                "parameters": {"reorder_point": 80},
                "source_ids": ["constraint_1"],
            },
            {
                "type": "storage_capacity",
                "parameters": {"capacity": 400},
                "source_ids": ["constraint_2"],
            },
        ],
        "unsupported_constraints": [],
    }
    model_input_response = {
        "model_inputs": {
            "annual_demand": 7300,
            "fixed_order_cost": 250,
            "holding_cost": 50,
            "penalty_cost": 10,
            "lead_time_demand_mean": 60,
            "lead_time_demand_std": 6.928,
        }
    }

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(requirement_response)),
        AIMessage(content=json.dumps(constraint_response)),
        AIMessage(content=json.dumps(model_input_response)),
    ]
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "control": {"event": EVENT_NEW_PROBLEM},
            "raw_input": Q_R_DESCRIPTION,
            "form_data": {
                "decision": "order_quantity",
                "objective": "minimize_cost",
                "demand_type": "uncertain",
                "planning_horizon": "multiple_periods",
                "review_type": "continuous_review",
            },
            "messages": [],
        }
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS
    )
    assert result["model_input_candidates"][
        "reorder_point"
    ]["q_r_policy"] == {
        "annual_demand": 7300.0,
        "fixed_order_cost": 250.0,
        "holding_cost": 50.0,
        "penalty_cost": 10.0,
        "lead_time_demand_mean": 60.0,
        "lead_time_demand_std": 6.928,
    }
    assert result.get("model_specific_inputs", {}) == {}
    assert result["model_input_extraction_errors"] == []
    assert {
        item["type"]
        for item in result["constraint_candidates"]
    } == {
        "minimum_order_quantity",
        "maximum_order_quantity",
        "minimum_reorder_point",
        "maximum_reorder_point",
        "storage_capacity",
    }
    assert result["required_model_inputs"][0]["form_id"] == (
        "reorder_point_q_r_policy_parameters"
    )
    assert llm.invoke.call_count == 3
