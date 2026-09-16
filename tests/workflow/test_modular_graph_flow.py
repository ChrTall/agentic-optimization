import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agentic_optimization.graph import build_graph
from agentic_optimization.state import (
    EVENT_MODULAR_INPUTS_SUBMITTED,
    EVENT_MODULAR_MODEL_PREPARATION_REQUESTED,
    EVENT_MODULAR_OPTIMIZATION_REQUESTED,
    EVENT_NEW_PROBLEM,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_OPTIMIZATION_COMPLETED,
    WORKFLOW_READY_FOR_REVIEW,
    EVENT_BUSINESS_INTAKE_REQUESTED,
    EVENT_MODULAR_REQUIREMENT_EXCLUSIONS_CONFIRMED,
    WORKFLOW_BUSINESS_CORE_SELECTED,
    EVENT_MODULAR_PRODUCT_SCHEMA_REQUESTED,
    EVENT_MODULAR_REVIEW_EXPLANATION_REQUESTED,
    EVENT_MODULAR_RESULT_EXPLANATION_REQUESTED,
    WORKFLOW_PRODUCT_TABLE_READY,
    get_workflow_status,
)
from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
    create_profile_field,
)
from agentic_optimization.models.proposal_mapper import (
    build_specification_from_proposal,
)


def test_runs_modular_model_through_graph() -> None:
    response = {
        "core_id": "continuous_review_q_r",
        "selection_status": "complete",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten berücksichtigen.",
                "sources": ["source_1"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten berücksichtigen.",
                "sources": ["source_1"],
            },
            {
                "component_id": "annual_shortage_cost",
                "reason": "Fehlmengenkosten berücksichtigen.",
                "sources": ["source_1"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content=json.dumps(
                {
                    "products": [],
                    "global_facts": {},
                }
            )
        ),
        AIMessage(
            content=json.dumps(response)
        ),
        AIMessage(
            content=json.dumps(
                {
                    "uncovered_requirements": [],
                }
            )
        ),
    ]

    graph = build_graph(llm)

    initial_state = {
        "control": {
            "event": (
                EVENT_MODULAR_MODEL_PREPARATION_REQUESTED
            ),
        },
        "requirements": {
            "objective": "minimize_cost",
        },
        "modular_model": {
            "core_id": "continuous_review_q_r",
            "products": [
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
            "global_facts": {},
            "evidence": [
                {
                    "source_id": "source_1",
                    "content": (
                        "Bestellkosten, Lagerkosten und "
                        "Fehlmengenkosten berücksichtigen."
                    ),
                }
            ],
        },
    }

    prepared_state = graph.invoke(initial_state)
    prepared_model = prepared_state["modular_model"]

    assert get_workflow_status(prepared_state) == (
        WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS
    )
    assert "event" not in prepared_state["control"]

    values = {
        "ordering_cost": 50.0,
        "holding_cost": 2.0,
        "shortage_cost": 20.0,
    }

    input_state = {
        **prepared_state,
        "control": {
            **prepared_state["control"],
            "event": EVENT_MODULAR_INPUTS_SUBMITTED,
        },
        "modular_model": {
            **prepared_model,
            "pending_input_submissions": [
                {
                    "missing_input": missing_input,
                    "value": values[
                        missing_input["field"]["name"]
                    ],
                }
                for missing_input
                in prepared_model["validation"][
                    "missing_inputs"
                ]
            ],
        },
    }

    completed_state = graph.invoke(input_state)
    completed_model = completed_state["modular_model"]

    assert completed_model["validation"]["is_valid"] is True
    assert get_workflow_status(completed_state) == (
        WORKFLOW_READY_FOR_REVIEW
    )
    assert "event" not in completed_state["control"]

    optimization_state = {
        **completed_state,
        "control": {
            **completed_state["control"],
            "event": EVENT_MODULAR_OPTIMIZATION_REQUESTED,
        },
        "modular_model": {
            **completed_model,
            "review_confirmed": True,
        },
    }

    optimized_state = graph.invoke(optimization_state)
    optimization_result = optimized_state[
        "modular_model"
    ]["optimization_result"]

    assert optimization_result is not None
    assert optimization_result["status"] == "optimal"
    assert optimization_result["solution_method"] == (
        "numerical"
    )
    assert {
        decision["quantity_id"]
        for decision in optimization_result["decisions"]
    } == {
        "order_quantity",
        "reorder_point",
    }

    assert get_workflow_status(optimized_state) == (
        WORKFLOW_OPTIMIZATION_COMPLETED
    )
    assert "event" not in optimized_state["control"]

    assert llm.invoke.call_count == 3


def test_enters_modular_pipeline_after_model_selection() -> None:
    requirement_response = {
        "decision": None,
        "objective": None,
        "demand_type": None,
        "product_scope": None,
        "planning_horizon": None,
        "review_type": None,
        "constraints": [],
        "unsupported_requirements": [],
    }

    fact_extraction_response = {
        "products": [],
        "global_facts": {},
    }

    component_response = {
        "core_id": "single_period_newsvendor",
        "selection_status": "complete",
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "reason": "Verkaufserlöse berücksichtigen.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "purchase_cost",
                "reason": "Einkaufskosten berücksichtigen.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content=json.dumps(requirement_response)
        ),
        AIMessage(
            content=json.dumps(
                fact_extraction_response
            )
        ),
        AIMessage(
            content=json.dumps(component_response)
        ),
        AIMessage(
            content=json.dumps(
                {
                    "uncovered_requirements": [],
                }
            )
        ),
    ]

    graph = build_graph(llm)

    result = graph.invoke(
        {
            "control": {
                "event": EVENT_NEW_PROBLEM,
            },
            "use_modular_pipeline": True,
            "form_data": {
                "decision": "order_quantity",
                "objective": "maximize_profit",
                "demand_type": "uncertain",
                "planning_horizon": "single_period",
            },
            "raw_input": (
                "Verkaufserlöse und Einkaufskosten "
                "sollen berücksichtigt werden."
            ),
        }
    )

    modular_state = result["modular_model"]

    assert result["selected_model_family"] == "newsvendor"
    assert result["selected_model_variant"] == (
        "single_product_single_period"
    )
    assert modular_state["core_id"] == (
        "single_period_newsvendor"
    )
    assert modular_state["proposal"] == component_response
    assert modular_state["validation"]["is_valid"] is False
    assert modular_state["validation"]["missing_inputs"]
    assert get_workflow_status(result) == (
        WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS
    )
    assert "event" not in result["control"]

    assert llm.invoke.call_count == 4


def test_enters_graph_through_business_intake() -> None:
    profile = create_empty_business_problem_profile()

    evidence = [
        {
            "source_id": "form_1",
            "source_type": "user_selection",
        }
    ]

    profile["recommendation"] = create_profile_field(
        status="confirmed",
        value="order_quantity",
        evidence=evidence,
    )
    profile["procurement_pattern"] = (
        create_profile_field(
            status="confirmed",
            value="recurring",
            evidence=evidence,
        )
    )
    profile["product_scope"] = create_profile_field(
        status="confirmed",
        value="single_product",
        evidence=evidence,
    )
    profile["primary_goal"] = create_profile_field(
        status="confirmed",
        value="minimize_cost",
        evidence=evidence,
    )

    llm = Mock()
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "control": {
                "event": (
                    EVENT_BUSINESS_INTAKE_REQUESTED
                ),
            },
            "raw_input": "",
            "business_intake": {
                "profile": profile,
                "asked_question_ids": [],
                "pending_question_ids": [],
                "pending_answers": [],
            },
        }
    )

    assert get_workflow_status(result) == (
        WORKFLOW_BUSINESS_CORE_SELECTED
    )
    assert "event" not in result["control"]

    assert result["business_intake"][
        "core_selection"
    ]["selected_core_id"] == "eoq"

    assert result["modular_model"]["core_id"] == "eoq"
    assert result["modular_model"][
        "review_confirmed"
    ] is False
    assert result["modular_model"][
        "optimization_result"
    ] is None

    llm.invoke.assert_not_called()


def test_routes_confirmed_requirement_exclusions(
) -> None:
    graph = build_graph(Mock())

    state = {
        "control": {
            "event": (
                EVENT_MODULAR_REQUIREMENT_EXCLUSIONS_CONFIRMED
            ),
        },
        "modular_model": {
            "core_id": "eoq",
            "products": [
                {
                    "product_id": "product_1",
                    "forecast": {
                        "expected_value": 3500.0,
                        "horizon_days": 365,
                    },
                    "facts": {
                        "ordering_cost": 120.0,
                        "holding_cost": 12.0,
                    },
                }
            ],
            "global_facts": {},
            "proposal": {
                "core_id": "eoq",
                "selection_status": "unsupported",
                "objective_sense": "minimize",
                "objective_components": [
                    {
                        "component_id": (
                            "annual_ordering_cost"
                        ),
                        "reason": (
                            "Bestellkosten wurden genannt."
                        ),
                        "sources": ["raw_input"],
                    },
                    {
                        "component_id": (
                            "annual_holding_cost"
                        ),
                        "reason": (
                            "Lagerkosten wurden genannt."
                        ),
                        "sources": ["raw_input"],
                    },
                ],
                "constraints": [],
                "unresolved_requirements": [
                    {
                        "description": (
                            "Die Lieferung muss an einem "
                            "Dienstag erfolgen."
                        ),
                        "resolution": "unsupported",
                        "candidate_component_ids": [],
                        "reason": (
                            "Für feste Liefertage ist kein "
                            "Baustein verfügbar."
                        ),
                        "sources": ["raw_input"],
                    }
                ],
            },
            "excluded_requirements": [],
            "review_confirmed": False,
            "optimization_result": None,
        },
    }

    result = graph.invoke(state)

    assert get_workflow_status(result) == (
        WORKFLOW_READY_FOR_REVIEW
    )
    assert "event" not in result["control"]

    modular_result = result["modular_model"]

    assert modular_result["proposal"][
        "selection_status"
    ] == "complete"
    assert modular_result["specification"][
        "excluded_requirements"
    ][0]["description"] == (
        "Die Lieferung muss an einem Dienstag erfolgen."
    )


def test_prepares_dynamic_product_table_through_graph(
) -> None:
    proposal = {
        "core_id": "single_period_newsvendor",
        "selection_status": "complete",
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "reason": "Verkaufserlöse berücksichtigen.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "purchase_cost",
                "reason": "Einkaufskosten berücksichtigen.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(proposal)
    )
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "control": {
                "event": (
                    EVENT_MODULAR_PRODUCT_SCHEMA_REQUESTED
                ),
            },
            "requirements": {
                "objective": "Gewinn maximieren",
            },
            "modular_model": {
                "core_id": "single_period_newsvendor",
                "products": [],
                "global_facts": {},
                "evidence": [
                    {
                        "source_id": "raw_input",
                        "content": (
                            "Der erwartete Gewinn soll "
                            "maximiert werden."
                        ),
                    }
                ],
            },
        }
    )

    assert get_workflow_status(result) == (
        WORKFLOW_PRODUCT_TABLE_READY
    )
    assert "event" not in result["control"]
    assert result["modular_model"][
        "preliminary_proposal"
    ] == proposal


def test_generates_review_explanation_through_graph(
) -> None:
    proposal = {
        "core_id": "single_period_newsvendor",
        "selection_status": "complete",
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "reason": "Verkaufserlöse berücksichtigen.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "purchase_cost",
                "reason": "Einkaufskosten berücksichtigen.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }
    products = [
        {
            "product_id": "product_1",
            "forecast": {
                "expected_value": 100.0,
                "quantiles": [
                    {
                        "level": 0.5,
                        "value": 100.0,
                    }
                ],
                "horizon_days": 30,
            },
            "facts": {
                "selling_price": 12.0,
                "purchase_cost": 5.0,
            },
        }
    ]
    specification = build_specification_from_proposal(
        proposal,
        products,
        {},
    )

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "explanation": (
                    "Das Newsvendor-Modell berechnet die "
                    "einmalige Bestellmenge bei unsicherer "
                    "Nachfrage."
                )
            }
        )
    )
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "control": {
                "event": (
                    EVENT_MODULAR_REVIEW_EXPLANATION_REQUESTED
                ),
            },
            "modular_model": {
                "core_id": "single_period_newsvendor",
                "products": products,
                "global_facts": {},
                "proposal": proposal,
                "specification": specification,
            },
        }
    )

    assert get_workflow_status(result) == (
        WORKFLOW_READY_FOR_REVIEW
    )
    assert "event" not in result["control"]
    assert result["modular_model"][
        "review_explanation"
    ].startswith("Das Newsvendor-Modell")
    assert result["modular_model"][
        "review_explanation_source"
    ] == "llm"
    assert llm.invoke.call_count == 1


def test_generates_result_explanation_through_graph(
) -> None:
    proposal = {
        "core_id": "eoq",
        "selection_status": "complete",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten berücksichtigen.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten berücksichtigen.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }
    products = [
        {
            "product_id": "product_1",
            "forecast": {
                "expected_value": 3500.0,
                "horizon_days": 365,
            },
            "facts": {
                "ordering_cost": 120.0,
                "holding_cost": 12.0,
            },
        }
    ]
    specification = build_specification_from_proposal(
        proposal,
        products,
        {},
    )
    optimization_result = {
        "status": "optimal",
        "solution_method": "analytical",
        "objective_value": 3174.9,
        "decisions": [
            {
                "quantity_id": "order_quantity",
                "product_id": "product_1",
                "value": 264.58,
            }
        ],
        "metrics": [],
        "technical_representation": None,
        "errors": [],
    }

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "explanation": (
                    "Empfohlen wird eine Bestellmenge "
                    "von 264,58 Einheiten."
                )
            }
        )
    )
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "control": {
                "event": (
                    EVENT_MODULAR_RESULT_EXPLANATION_REQUESTED
                ),
                "workflow_status": (
                    WORKFLOW_OPTIMIZATION_COMPLETED
                ),
            },
            "modular_model": {
                "core_id": "eoq",
                "products": products,
                "global_facts": {},
                "proposal": proposal,
                "specification": specification,
                "optimization_result": (
                    optimization_result
                ),
            },
        }
    )

    assert get_workflow_status(result) == (
        WORKFLOW_OPTIMIZATION_COMPLETED
    )
    assert "event" not in result["control"]
    assert result["modular_model"][
        "result_explanation"
    ].startswith("Empfohlen wird")
    assert result["modular_model"][
        "result_explanation_source"
    ] == "llm"
    assert llm.invoke.call_count == 1
