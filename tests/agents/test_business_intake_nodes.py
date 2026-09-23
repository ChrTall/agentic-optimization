import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage
import pytest

from agentic_optimization.agents.business_intake_nodes import (
    create_business_intake_node,
    initialize_modular_core_node,
)
from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
    create_profile_field,
)
from agentic_optimization.state import (
    WORKFLOW_BUSINESS_CASE_UNSUPPORTED,
    WORKFLOW_BUSINESS_CORE_SELECTED,
    WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED,
    WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
    get_workflow_status,
)


def confirmed(value):
    return create_profile_field(
        status="confirmed",
        value=value,
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )


def test_node_selects_core_from_analyzed_description() -> None:
    raw_input = (
        "Wir möchten nur die Bestellmenge bestimmen. "
        "Das Produkt wird regelmäßig nachbestellt. "
        "Es geht um ein Produkt. "
        "Wir möchten die Gesamtkosten reduzieren."
    )

    response = {
        "inferences": [
            {
                "field_id": "recommendation",
                "value": "order_quantity",
                "excerpt": (
                    "nur die Bestellmenge bestimmen"
                ),
            },
            {
                "field_id": "procurement_pattern",
                "value": "recurring",
                "excerpt": (
                    "Das Produkt wird regelmäßig nachbestellt"
                ),
            },
            {
                "field_id": "product_scope",
                "value": "single_product",
                "excerpt": "Es geht um ein Produkt",
            },
            {
                "field_id": "primary_goal",
                "value": "minimize_cost",
                "excerpt": (
                    "Wir möchten die Gesamtkosten reduzieren"
                ),
            },
        ],
        "ambiguities": [],
    }

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(response)
    )

    node = create_business_intake_node(llm)

    state = {
        "raw_input": raw_input,
        "business_intake": {
            "profile": (
                create_empty_business_problem_profile()
            ),
            "asked_question_ids": [],
        },
    }

    result = node(state)
    intake_result = result["business_intake"]

    assert get_workflow_status(result) == (
        WORKFLOW_BUSINESS_CORE_SELECTED
    )
    assert intake_result["core_selection"]["status"] == (
        "selected"
    )
    assert intake_result["core_selection"][
        "selected_core_id"
    ] == "eoq"
    assert intake_result["pending_question_ids"] == []
    assert intake_result["analysis_error"] is None
    assert result["raw_input"] == raw_input
    assert llm.invoke.call_count == 1


def test_node_requests_missing_business_answers() -> None:
    llm = Mock()
    node = create_business_intake_node(llm)

    result = node(
        {
            "raw_input": "",
            "business_intake": {
                "profile": (
                    create_empty_business_problem_profile()
                ),
                "asked_question_ids": [],
            },
        }
    )

    intake_result = result["business_intake"]

    assert get_workflow_status(result) == (
        WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT
    )
    assert intake_result["pending_question_ids"] == [
        "recommendation",
        "procurement_pattern",
        "product_scope",
        "primary_goal",
    ]
    assert intake_result["core_selection"]["status"] == (
        "needs_clarification"
    )
    llm.invoke.assert_not_called()


def test_node_reports_unsupported_business_case() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity_and_reorder_point"
    )
    profile["procurement_pattern"] = confirmed(
        "recurring"
    )
    profile["inventory_review"] = confirmed(
        "periodic"
    )
    profile["product_scope"] = confirmed(
        "single_product"
    )
    profile["primary_goal"] = confirmed(
        "minimize_cost"
    )

    node = create_business_intake_node(Mock())

    result = node(
        {
            "raw_input": "",
            "business_intake": {
                "profile": profile,
                "asked_question_ids": [],
            },
        }
    )

    assert get_workflow_status(result) == (
        WORKFLOW_BUSINESS_CASE_UNSUPPORTED
    )
    assert result["business_intake"][
        "core_selection"
    ]["status"] == "unsupported"


def test_node_handles_invalid_agent_response() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content="keine gültige JSON-Antwort"
    )

    node = create_business_intake_node(llm)

    result = node(
        {
            "raw_input": "Eine unklare Beschreibung.",
            "business_intake": {
                "profile": (
                    create_empty_business_problem_profile()
                ),
                "asked_question_ids": [],
                "pending_question_ids": [
                    "recommendation",
                ],
            },
        }
    )

    intake_result = result["business_intake"]

    assert get_workflow_status(result) == (
        WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED
    )
    assert intake_result["analysis_error"] == (
        "Die Beschreibung konnte nicht zuverlässig "
        "ausgewertet werden. Bitte starten Sie die "
        "Analyse erneut."
    )
    assert intake_result["pending_question_ids"] == [
        "recommendation",
    ]
    assert "core_selection" not in intake_result


def test_node_applies_pending_business_answers() -> None:
    profile = create_empty_business_problem_profile()
    llm = Mock()
    node = create_business_intake_node(llm)

    pending_question_ids = [
        "recommendation",
        "procurement_pattern",
        "product_scope",
        "primary_goal",
    ]

    result = node(
        {
            "raw_input": "",
            "business_intake": {
                "profile": profile,
                "asked_question_ids": [],
                "pending_question_ids": (
                    pending_question_ids
                ),
                "pending_answers": [
                    {
                        "question_id": "recommendation",
                        "option_id": "order_quantity",
                        "source_id": "form_1",
                    },
                    {
                        "question_id": (
                            "procurement_pattern"
                        ),
                        "option_id": "recurring",
                        "source_id": "form_1",
                    },
                    {
                        "question_id": "product_scope",
                        "option_id": "single_product",
                        "source_id": "form_1",
                    },
                    {
                        "question_id": "primary_goal",
                        "option_id": "minimize_cost",
                        "source_id": "form_1",
                    },
                ],
            },
        }
    )

    intake_result = result["business_intake"]

    assert get_workflow_status(result) == (
        WORKFLOW_BUSINESS_CORE_SELECTED
    )
    assert intake_result["core_selection"][
        "selected_core_id"
    ] == "eoq"
    assert intake_result["pending_question_ids"] == []
    assert intake_result["pending_answers"] == []
    assert intake_result["asked_question_ids"] == (
        pending_question_ids
    )
    llm.invoke.assert_not_called()


def test_node_rejects_answer_for_stale_question() -> None:
    node = create_business_intake_node(Mock())

    with pytest.raises(
        ValueError,
        match="nicht mehr offen",
    ):
        node(
            {
                "raw_input": "",
                "business_intake": {
                    "profile": (
                        create_empty_business_problem_profile()
                    ),
                    "asked_question_ids": [],
                    "pending_question_ids": [
                        "recommendation",
                    ],
                    "pending_answers": [
                        {
                            "question_id": "primary_goal",
                            "option_id": "minimize_cost",
                            "source_id": "form_1",
                        }
                    ],
                },
            }
        )


def test_initializes_modular_core_from_selection() -> None:
    state = {
        "business_intake": {
            "profile": (
                create_empty_business_problem_profile()
            ),
            "core_selection": {
                "status": "selected",
                "selected_core_id": "eoq",
                "candidate_core_ids": ["eoq"],
                "reason": "Eindeutige Auswahl.",
            },
        },
        "modular_model": {
            "core_id": "single_period_newsvendor",
            "products": [
                {
                    "product_id": "product_1",
                    "forecast": {},
                    "facts": {},
                }
            ],
            "global_facts": {},
            "evidence": [],
            "proposal": {},
            "specification": {},
            "validation": {},
            "pending_input_submissions": [],
            "review_confirmed": True,
            "optimization_result": {
                "status": "optimal",
            },
            "preparation_error": "Alter Fehler",
        },
    }

    result = initialize_modular_core_node(state)
    modular_result = result["modular_model"]

    assert modular_result["core_id"] == "eoq"
    assert modular_result["products"] == (
        state["modular_model"]["products"]
    )
    assert modular_result["review_confirmed"] is False
    assert modular_result["optimization_result"] is None
    assert modular_result["preparation_error"] is None

    assert "proposal" not in modular_result
    assert "specification" not in modular_result
    assert "validation" not in modular_result
    assert modular_result[
        "pending_input_submissions"
    ] == []

    assert state["modular_model"]["core_id"] == (
        "single_period_newsvendor"
    )
    assert "proposal" in state["modular_model"]


def test_rejects_modular_initialization_before_selection() -> None:
    with pytest.raises(
        ValueError,
        match="noch kein Grundmodell",
    ):
        initialize_modular_core_node(
            {
                "business_intake": {
                    "profile": (
                        create_empty_business_problem_profile()
                    ),
                    "core_selection": {
                        "status": "needs_clarification",
                        "selected_core_id": None,
                        "candidate_core_ids": [
                            "eoq",
                            "single_period_newsvendor",
                        ],
                        "reason": "Weitere Angaben fehlen.",
                    },
                },
            }
        )


def test_rejects_modular_initialization_without_intake() -> None:
    with pytest.raises(
        ValueError,
        match="Geschäftseinstiegs fehlt",
    ):
        initialize_modular_core_node({})


def test_node_does_not_reanalyze_unchanged_description() -> None:
    raw_input = (
        "Das Produkt wird regelmäßig nachbestellt."
    )
    profile = create_empty_business_problem_profile()
    profile["procurement_pattern"] = create_profile_field(
        status="inferred",
        value="recurring",
        evidence=[
            {
                "source_id": "raw_input",
                "source_type": "free_text",
                "excerpt": raw_input,
            }
        ],
    )

    llm = Mock()
    node = create_business_intake_node(llm)

    result = node(
        {
            "raw_input": raw_input,
            "business_intake": {
                "profile": profile,
                "asked_question_ids": [],
                "pending_question_ids": [
                    "recommendation",
                    "product_scope",
                    "primary_goal",
                ],
                "pending_answers": [],
                "analyzed_raw_input": raw_input,
            },
        }
    )

    llm.invoke.assert_not_called()

    assert result["business_intake"][
        "profile"
    ]["procurement_pattern"]["value"] == "recurring"
    assert result["business_intake"][
        "analyzed_raw_input"
    ] == raw_input


def test_initializes_single_product_and_raw_evidence() -> None:
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed(
        "single_product"
    )

    result = initialize_modular_core_node(
        {
            "raw_input": (
                "Das Produkt wird regelmäßig nachbestellt."
            ),
            "business_intake": {
                "profile": profile,
                "core_selection": {
                    "status": "selected",
                    "selected_core_id": "eoq",
                    "candidate_core_ids": ["eoq"],
                    "reason": "Eindeutige Auswahl.",
                },
            },
        }
    )

    modular_state = result["modular_model"]

    assert modular_state["products"] == [
        {
            "product_id": "product_1",
            "forecast": {},
            "facts": {},
        }
    ]
    assert modular_state["global_facts"] == {}
    assert modular_state["evidence"] == [
        {
            "source_id": "raw_input",
            "content": (
                "Das Produkt wird regelmäßig nachbestellt."
            ),
        },
        {
            "source_id": "business_intake_form",
            "content": '{"product_scope": "single_product"}',
        },
    ]
    assert result["requirements"] == {
        "product_scope": "single_product",
        "constraints": [],
        "raw_description": (
            "Das Produkt wird regelmäßig nachbestellt."
        ),
    }


def test_initializes_from_form_without_text_or_stale_proposal() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed("order_quantity")
    profile["procurement_pattern"] = confirmed("recurring")
    profile["product_scope"] = confirmed("single_product")
    profile["primary_goal"] = confirmed("minimize_cost")

    result = initialize_modular_core_node(
        {
            "raw_input": "",
            "business_intake": {
                "profile": profile,
                "core_selection": {
                    "status": "selected",
                    "selected_core_id": "eoq",
                    "candidate_core_ids": ["eoq"],
                    "reason": "Eindeutige Auswahl.",
                },
            },
            "modular_model": {
                "evidence": [
                    {
                        "source_id": "raw_input",
                        "content": "Alter Beschreibungstext",
                    }
                ],
                "preliminary_proposal": {
                    "core_id": "single_period_newsvendor",
                },
            },
        }
    )

    modular_state = result["modular_model"]

    assert modular_state["core_id"] == "eoq"
    assert "preliminary_proposal" not in modular_state
    assert len(modular_state["evidence"]) == 1
    assert modular_state["evidence"][0]["source_id"] == (
        "business_intake_form"
    )
    assert json.loads(
        modular_state["evidence"][0]["content"]
    ) == {
        "recommendation": "order_quantity",
        "procurement_pattern": "recurring",
        "product_scope": "single_product",
        "primary_goal": "minimize_cost",
    }


def test_does_not_invent_multiple_products() -> None:
    profile = create_empty_business_problem_profile()
    profile["product_scope"] = confirmed(
        "multiple_products"
    )

    result = initialize_modular_core_node(
        {
            "raw_input": "",
            "business_intake": {
                "profile": profile,
                "core_selection": {
                    "status": "selected",
                    "selected_core_id": (
                        "single_period_newsvendor"
                    ),
                    "candidate_core_ids": [
                        "single_period_newsvendor",
                    ],
                    "reason": "Eindeutige Auswahl.",
                },
            },
        }
    )

    assert result["modular_model"]["products"] == []
