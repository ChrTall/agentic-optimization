import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agentic_optimization.agents.constraint_mapping_agent import (
    create_constraint_mapping_agent,
)
from agentic_optimization.state import (
    WORKFLOW_MODEL_INPUTS_INVALID,
    get_workflow_status,
)


def create_mock_llm(response_content: str) -> Mock:
    llm = Mock()

    llm.invoke.return_value = AIMessage(
        content=response_content
    )

    return llm


def create_base_state(
    constraints: list[str],
) -> dict:
    return {
        "selected_model_family": "newsvendor",
        "selected_model_variant": (
            "single_product_single_period"
        ),
        "requirements": {
            "constraints": constraints,
        },
    }


def test_skips_llm_when_no_constraints_exist() -> None:
    llm = create_mock_llm(
        response_content="This response must not be used."
    )

    agent = create_constraint_mapping_agent(llm)

    result = agent(
        create_base_state(constraints=[])
    )

    assert result["structured_constraints"] == []
    assert result["unsupported_constraints"] == []
    assert result["constraint_mapping_errors"] == []

    llm.invoke.assert_not_called()


def test_maps_budget_constraint() -> None:
    response = {
        "structured_constraints": [
            {
                "type": "budget_limit",
                "parameters": {
                    "budget": 1000,
                },
                "source_ids": [
                    "constraint_0",
                ],
            }
        ],
        "unsupported_constraints": [],
    }

    llm = create_mock_llm(
        response_content=json.dumps(response)
    )

    agent = create_constraint_mapping_agent(llm)

    result = agent(
        create_base_state(
            constraints=[
                "Das Budget beträgt maximal 1000 Euro."
            ]
        )
    )

    assert result["constraint_mapping_errors"] == []
    assert result["unsupported_constraints"] == []

    assert result["structured_constraints"] == [
        {
            "type": "budget_limit",
            "parameters": {
                "budget": 1000,
            },
            "source_text": (
                "Das Budget beträgt maximal 1000 Euro."
            ),
        }
    ]

    llm.invoke.assert_called_once()

    messages = llm.invoke.call_args.args[0]
    human_message = messages[1].content

    assert "budget_limit" in human_message
    assert "minimum_order_quantity" in human_message
    assert "maximum_order_quantity" in human_message
    assert "storage_capacity" in human_message


def test_maps_one_text_to_multiple_constraints() -> None:
    response = {
        "structured_constraints": [
            {
                "type": "minimum_order_quantity",
                "parameters": {
                    "quantity": 40,
                },
                "source_ids": [
                    "constraint_0",
                ],
            },
            {
                "type": "maximum_order_quantity",
                "parameters": {
                    "quantity": 100,
                },
                "source_ids": [
                    "constraint_0",
                ],
            },
        ],
        "unsupported_constraints": [],
    }

    llm = create_mock_llm(
        response_content=json.dumps(response)
    )

    agent = create_constraint_mapping_agent(llm)

    result = agent(
        create_base_state(
            constraints=[
                (
                    "Es müssen mindestens 40 und dürfen "
                    "höchstens 100 Einheiten bestellt werden."
                )
            ]
        )
    )

    assert result["constraint_mapping_errors"] == []

    assert [
        constraint["type"]
        for constraint in result["structured_constraints"]
    ] == [
        "minimum_order_quantity",
        "maximum_order_quantity",
    ]


def test_maps_missing_parameter_without_inventing_value() -> None:
    response = {
        "structured_constraints": [
            {
                "type": "budget_limit",
                "parameters": {},
                "source_ids": [
                    "constraint_0",
                ],
            }
        ],
        "unsupported_constraints": [],
    }

    llm = create_mock_llm(
        response_content=json.dumps(response)
    )

    agent = create_constraint_mapping_agent(llm)

    result = agent(
        create_base_state(
            constraints=[
                "Es gibt eine feste Budgetgrenze."
            ]
        )
    )

    assert result["constraint_mapping_errors"] == []

    assert result["structured_constraints"] == [
        {
            "type": "budget_limit",
            "parameters": {},
            "source_text": (
                "Es gibt eine feste Budgetgrenze."
            ),
        }
    ]


def test_preserves_unsupported_constraint_text() -> None:
    response = {
        "structured_constraints": [],
        "unsupported_constraints": [
            {
                "source_id": "constraint_0",
            }
        ],
    }

    llm = create_mock_llm(
        response_content=json.dumps(response)
    )

    agent = create_constraint_mapping_agent(llm)

    result = agent(
        create_base_state(
            constraints=[
                "Die Lieferung muss am Dienstag eintreffen."
            ]
        )
    )

    assert result["structured_constraints"] == []

    assert result["unsupported_constraints"] == [
        "Die Lieferung muss am Dienstag eintreffen."
    ]

    assert result["constraint_mapping_errors"] == []


def test_rejects_llm_invented_constraint_type() -> None:
    response = {
        "structured_constraints": [
            {
                "type": "delivery_deadline",
                "parameters": {},
                "source_ids": [
                    "constraint_0",
                ],
            }
        ],
        "unsupported_constraints": [],
    }

    llm = create_mock_llm(
        response_content=json.dumps(response)
    )

    agent = create_constraint_mapping_agent(llm)

    result = agent(
        create_base_state(
            constraints=[
                "Die Lieferung muss am Dienstag eintreffen."
            ]
        )
    )

    assert result["structured_constraints"] == []
    assert result["constraint_mapping_errors"]

    assert any(
        "nicht erlaubten Constraint-Typ" in error
        for error in result["constraint_mapping_errors"]
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_MODEL_INPUTS_INVALID
    )


def test_rejects_unknown_constraint_parameter() -> None:
    response = {
        "structured_constraints": [
            {
                "type": "budget_limit",
                "parameters": {
                    "budget": 1000,
                    "currency": "EUR",
                },
                "source_ids": [
                    "constraint_0",
                ],
            }
        ],
        "unsupported_constraints": [],
    }

    llm = create_mock_llm(
        response_content=json.dumps(response)
    )

    agent = create_constraint_mapping_agent(llm)

    result = agent(
        create_base_state(
            constraints=[
                "Das Budget beträgt 1000 Euro."
            ]
        )
    )

    assert result["structured_constraints"] == []
    assert result["constraint_mapping_errors"]

    assert any(
        "unbekannte Parameter" in error
        for error in result["constraint_mapping_errors"]
    )


def test_rejects_unprocessed_source_constraint() -> None:
    response = {
        "structured_constraints": [
            {
                "type": "budget_limit",
                "parameters": {
                    "budget": 1000,
                },
                "source_ids": [
                    "constraint_0",
                ],
            }
        ],
        "unsupported_constraints": [],
    }

    llm = create_mock_llm(
        response_content=json.dumps(response)
    )

    agent = create_constraint_mapping_agent(llm)

    result = agent(
        create_base_state(
            constraints=[
                "Das Budget beträgt maximal 1000 Euro.",
                "Die Lieferung muss am Dienstag eintreffen.",
            ]
        )
    )

    assert result["structured_constraints"] == []
    assert result["constraint_mapping_errors"]

    assert any(
        "nicht alle Nutzeranforderungen verarbeitet"
        in error
        for error in result["constraint_mapping_errors"]
    )

    assert any(
        "constraint_1" in error
        for error in result["constraint_mapping_errors"]
    )


def test_rejects_unknown_source_id() -> None:
    response = {
        "structured_constraints": [
            {
                "type": "budget_limit",
                "parameters": {
                    "budget": 1000,
                },
                "source_ids": [
                    "constraint_99",
                ],
            }
        ],
        "unsupported_constraints": [],
    }

    llm = create_mock_llm(
        response_content=json.dumps(response)
    )

    agent = create_constraint_mapping_agent(llm)

    result = agent(
        create_base_state(
            constraints=[
                "Das Budget beträgt maximal 1000 Euro."
            ]
        )
    )

    assert result["structured_constraints"] == []
    assert result["constraint_mapping_errors"]

    assert any(
        "unbekannte source_id" in error
        for error in result["constraint_mapping_errors"]
    )


def test_rejects_invalid_json_response() -> None:
    llm = create_mock_llm(
        response_content="Das ist kein gültiges JSON."
    )

    agent = create_constraint_mapping_agent(llm)

    result = agent(
        create_base_state(
            constraints=[
                "Das Budget beträgt maximal 1000 Euro."
            ]
        )
    )

    assert result["structured_constraints"] == []
    assert result["unsupported_constraints"] == []
    assert result["constraint_mapping_errors"]

    assert any(
        "Constraint-Mapping ist fehlgeschlagen"
        in error
        for error in result["constraint_mapping_errors"]
    )


def test_rejects_missing_model_selection() -> None:
    llm = create_mock_llm(
        response_content="{}"
    )

    agent = create_constraint_mapping_agent(llm)

    result = agent(
        {
            "requirements": {
                "constraints": [
                    "Das Budget beträgt 1000 Euro."
                ],
            },
        }
    )

    assert result["constraint_mapping_errors"]

    assert (
        get_workflow_status(result)
        == WORKFLOW_MODEL_INPUTS_INVALID
    )

    llm.invoke.assert_not_called()