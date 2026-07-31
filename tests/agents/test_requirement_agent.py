import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agentic_optimization.agents.clarification import (
    clarification_node,
)
from agentic_optimization.agents.requirement_agent import (
    create_requirement_agent,
)
from agentic_optimization.agents.validation_agent import (
    validate_general_requirements,
)
from agentic_optimization.state import (
    WORKFLOW_GENERAL_INPUTS_VALID,
    WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    get_workflow_status,
)


def create_mock_llm(
    response: dict,
) -> Mock:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(response),
    )
    return llm


def create_llm_response(
    **overrides: object,
) -> dict:
    response = {
        "decision": None,
        "objective": None,
        "demand_type": None,
        "product_scope": None,
        "planning_horizon": None,
        "constraints": [],
        "unsupported_requirements": [],
    }
    response.update(overrides)
    return response


def create_complete_form_data() -> dict:
    return {
        "decision": "order_quantity",
        "objective": "maximize_profit",
        "demand_type": "uncertain",
        "product_scope": "single_product",
        "planning_horizon": "single_period",
    }


def test_uses_canonical_form_values() -> None:
    llm = create_mock_llm(
        create_llm_response()
    )
    agent = create_requirement_agent(llm)

    result = agent(
        {
            "form_data": create_complete_form_data(),
            "raw_input": "",
        }
    )

    requirements = result["requirements"]

    assert requirements["decision"] == "order_quantity"
    assert requirements["objective"] == "maximize_profit"
    assert requirements["demand_type"] == "uncertain"
    assert requirements["product_scope"] == "single_product"
    assert (
        requirements["planning_horizon"]
        == "single_period"
    )
    assert requirements["constraints"] == []
    assert result["missing_fields"] == []
    assert result["requirement_conflicts"] == []


def test_llm_fills_missing_form_values() -> None:
    llm = create_mock_llm(
        create_llm_response(
            product_scope="single_product",
            planning_horizon="single_period",
        )
    )
    agent = create_requirement_agent(llm)

    form_data = create_complete_form_data()
    form_data["product_scope"] = None
    form_data["planning_horizon"] = None

    result = agent(
        {
            "form_data": form_data,
            "raw_input": (
                "Es geht um ein Produkt und eine einmalige "
                "Bestellentscheidung."
            ),
        }
    )

    assert (
        result["requirements"]["product_scope"]
        == "single_product"
    )
    assert (
        result["requirements"]["planning_horizon"]
        == "single_period"
    )
    assert result["missing_fields"] == []
    assert result["requirement_conflicts"] == []


def test_detects_conflict_between_form_and_raw_input() -> None:
    llm = create_mock_llm(
        create_llm_response(
            product_scope="multiple_products",
            planning_horizon="single_period",
        )
    )
    agent = create_requirement_agent(llm)

    result = agent(
        {
            "form_data": create_complete_form_data(),
            "raw_input": (
                "Es sollen mehrere Produkte geplant werden."
            ),
        }
    )

    assert (
        result["requirements"]["product_scope"]
        == "single_product"
    )

    assert result["requirement_conflicts"] == [
        {
            "field": "product_scope",
            "form_value": "single_product",
            "raw_input_value": "multiple_products",
            "message": (
                "Die Angaben zu 'product_scope' widersprechen "
                "sich zwischen Formular und Freitext."
            ),
        }
    ]


def test_conflict_requires_clarification() -> None:
    llm = create_mock_llm(
        create_llm_response(
            product_scope="multiple_products",
            planning_horizon="single_period",
        )
    )
    agent = create_requirement_agent(llm)

    extracted_state = agent(
        {
            "form_data": create_complete_form_data(),
            "raw_input": (
                "Es sollen mehrere Produkte geplant werden."
            ),
        }
    )

    validated_state = validate_general_requirements(
        extracted_state
    )

    assert (
        get_workflow_status(validated_state)
        == WORKFLOW_NEEDS_GENERAL_CLARIFICATION
    )

    clarification_state = clarification_node(
        validated_state
    )

    assert (
        "widersprechen"
        in clarification_state["clarification_question"]
    )


def test_extracts_constraints_from_raw_input() -> None:
    llm = create_mock_llm(
        create_llm_response(
            constraints=[
                "Maximal 500 Einheiten Lagerkapazität",
                "Budget maximal 10000 Euro",
            ],
        )
    )
    agent = create_requirement_agent(llm)

    result = agent(
        {
            "form_data": create_complete_form_data(),
            "raw_input": (
                "Das Lager fasst maximal 500 Einheiten "
                "und das Budget beträgt 10000 Euro."
            ),
        }
    )

    assert result["requirements"]["constraints"] == [
        "Maximal 500 Einheiten Lagerkapazität",
        "Budget maximal 10000 Euro",
    ]


def test_missing_product_scope_requires_clarification() -> None:
    state = {
        "requirements": {
            "decision": "order_quantity",
            "objective": "maximize_profit",
            "demand_type": "uncertain",
            "product_scope": None,
            "planning_horizon": "single_period",
            "constraints": [],
            "raw_description": "",
        },
        "missing_fields": ["product_scope"],
        "requirement_conflicts": [],
    }

    result = validate_general_requirements(state)

    assert (
        get_workflow_status(result)
        == WORKFLOW_NEEDS_GENERAL_CLARIFICATION
    )
    assert "product_scope" in result["missing_fields"]


def test_complete_requirements_are_valid() -> None:
    state = {
        "requirements": {
            "decision": "order_quantity",
            "objective": "maximize_profit",
            "demand_type": "uncertain",
            "product_scope": "single_product",
            "planning_horizon": "single_period",
            "constraints": [],
            "raw_description": "",
        },
        "missing_fields": [],
        "requirement_conflicts": [],
    }

    result = validate_general_requirements(state)

    assert (
        get_workflow_status(result)
        == WORKFLOW_GENERAL_INPUTS_VALID
    )
    assert result["general_validation_errors"] == []


def test_invalid_requirement_value_is_rejected() -> None:
    state = {
        "requirements": {
            "decision": "tour_planning",
            "objective": "maximize_profit",
            "demand_type": "uncertain",
            "product_scope": "single_product",
            "planning_horizon": "single_period",
            "constraints": [],
            "raw_description": "",
        },
        "missing_fields": [],
        "requirement_conflicts": [],
    }

    result = validate_general_requirements(state)

    assert (
        get_workflow_status(result)
        == WORKFLOW_NEEDS_GENERAL_CLARIFICATION
    )
    assert result["general_validation_errors"] == [
        "Die Art der Entscheidung besitzt einen ungültigen Wert."
    ]


def test_safe_fallback_preserves_form_values() -> None:
    llm = Mock()
    llm.invoke.side_effect = RuntimeError(
        "LLM temporarily unavailable"
    )

    agent = create_requirement_agent(llm)

    result = agent(
        {
            "form_data": create_complete_form_data(),
            "raw_input": "Zusätzliche Beschreibung",
        }
    )

    requirements = result["requirements"]

    assert requirements["decision"] == "order_quantity"
    assert requirements["objective"] == "maximize_profit"
    assert requirements["demand_type"] == "uncertain"
    assert requirements["product_scope"] == "single_product"
    assert (
        requirements["planning_horizon"]
        == "single_period"
    )
    assert requirements["constraints"] == []


def test_separates_problem_structure_from_constraints() -> None:
    llm = create_mock_llm(
        create_llm_response(
            decision="order_quantity",
            demand_type="uncertain",
            product_scope="single_product",
            planning_horizon="single_period",
            constraints=[
                "Mindestbestellmenge von 30 Stück",
                "Maximales Budget von 5000 Euro",
            ],
        )
    )

    agent = create_requirement_agent(llm)

    result = agent(
        {
            "form_data": create_complete_form_data(),
            "raw_input": (
                "Wir verkaufen ein saisonales Produkt mit "
                "unsicherer Nachfrage. Dabei haben wir eine "
                "Mindestbestellmenge von 30 Stück. Außerdem "
                "haben wir ein maximales Budget von 5000 Euro."
            ),
        }
    )

    requirements = result["requirements"]

    assert requirements["demand_type"] == "uncertain"

    assert requirements["constraints"] == [
        "Mindestbestellmenge von 30 Stück",
        "Maximales Budget von 5000 Euro",
    ]

    assert "saisonales Produkt" not in requirements["constraints"]
    assert "unsichere Nachfrage" not in requirements["constraints"]


def test_preserves_unsupported_requirement_from_llm() -> None:
    llm = create_mock_llm(
        create_llm_response(
            demand_type="uncertain",
            unsupported_requirements=[
                {
                    "field": "decision",
                    "user_value": "Bestellzeitpunkt optimieren",
                    "message": (
                        "Die Optimierung des Bestellzeitpunkts "
                        "wird derzeit nicht unterstützt."
                    ),
                }
            ],
        )
    )
    agent = create_requirement_agent(llm)

    result = agent(
        {
            "form_data": create_complete_form_data(),
            "raw_input": (
                "Wir verkaufen ein Produkt mit unsicherer "
                "Nachfrage und möchten den Bestellzeitpunkt "
                "optimieren."
            ),
        }
    )

    assert result["unsupported_requirements"] == [
        {
            "field": "decision",
            "user_value": "Bestellzeitpunkt optimieren",
            "message": (
                "Die Optimierung des Bestellzeitpunkts "
                "wird derzeit nicht unterstützt."
            ),
        }
    ]


def test_unsupported_requirement_requires_clarification() -> None:
    llm = create_mock_llm(
        create_llm_response(
            demand_type="uncertain",
            unsupported_requirements=[
                {
                    "field": "decision",
                    "user_value": "Bestellzeitpunkt optimieren",
                    "message": (
                        "Die Optimierung des Bestellzeitpunkts "
                        "wird derzeit nicht unterstützt."
                    ),
                }
            ],
        )
    )
    agent = create_requirement_agent(llm)

    extracted_state = agent(
        {
            "form_data": create_complete_form_data(),
            "raw_input": (
                "Das Ziel ist, den Bestellzeitpunkt "
                "zu optimieren."
            ),
        }
    )

    validated_state = validate_general_requirements(
        extracted_state
    )

    assert (
        get_workflow_status(validated_state)
        == WORKFLOW_NEEDS_GENERAL_CLARIFICATION
    )

    clarification_state = clarification_node(
        validated_state
    )

    assert (
        "Bestellzeitpunkts"
        in clarification_state["clarification_question"]
    )


def test_new_analysis_clears_old_unsupported_requirements() -> None:
    llm = create_mock_llm(
        create_llm_response(
            demand_type="uncertain",
            unsupported_requirements=[],
        )
    )
    agent = create_requirement_agent(llm)

    result = agent(
        {
            "form_data": create_complete_form_data(),
            "raw_input": (
                "Wir verkaufen ein Produkt mit unsicherer "
                "Nachfrage."
            ),
            "unsupported_requirements": [
                {
                    "field": "decision",
                    "user_value": "Bestellzeitpunkt optimieren",
                    "message": (
                        "Die Optimierung des Bestellzeitpunkts "
                        "wird derzeit nicht unterstützt."
                    ),
                }
            ],
        }
    )

    assert result["unsupported_requirements"] == []