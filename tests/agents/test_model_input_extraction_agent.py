import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agentic_optimization.agents.model_input_extraction_agent import (
    create_model_input_extraction_agent,
)


def create_mock_llm(response_content: str) -> Mock:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=response_content
    )
    return llm


def create_q_r_state(
    raw_input: str = "Jährliche Nachfrage 7.300 Einheiten.",
) -> dict:
    return {
        "selected_model_family": "reorder_point",
        "selected_model_variant": "q_r_policy",
        "raw_input": raw_input,
        "model_specific_inputs": {},
    }


def test_extracts_all_q_r_fields_as_unconfirmed_candidates() -> None:
    response = {
        "model_inputs": {
            "annual_demand": 7300,
            "fixed_order_cost": 250,
            "holding_cost": 50,
            "penalty_cost": 10,
            "lead_time_demand_mean": 60,
            "lead_time_demand_std": 6.928,
        }
    }
    llm = create_mock_llm(json.dumps(response))
    agent = create_model_input_extraction_agent(llm)

    result = agent(
        create_q_r_state(
            "Jährliche Nachfrage 7.300, Bestellkosten 250, "
            "Lagerkosten 50, Fehlmengenkosten 10, "
            "Lieferzeitnachfrage 60 mit Standardabweichung 6,928."
        )
    )

    assert result["model_input_extraction_errors"] == []
    assert result["model_input_candidates"] == {
        "reorder_point": {
            "q_r_policy": {
                "annual_demand": 7300.0,
                "fixed_order_cost": 250.0,
                "holding_cost": 50.0,
                "penalty_cost": 10.0,
                "lead_time_demand_mean": 60.0,
                "lead_time_demand_std": 6.928,
            }
        }
    }
    assert result["model_specific_inputs"] == {}

    messages = llm.invoke.call_args.args[0]
    human_message = messages[1].content

    for field_name in response["model_inputs"]:
        assert field_name in human_message


def test_accepts_partial_extraction_and_omits_null_values() -> None:
    llm = create_mock_llm(
        json.dumps(
            {
                "model_inputs": {
                    "annual_demand": 7300,
                    "fixed_order_cost": None,
                }
            }
        )
    )
    agent = create_model_input_extraction_agent(llm)

    result = agent(create_q_r_state())

    assert result["model_input_extraction_errors"] == []
    assert result["model_input_candidates"] == {
        "reorder_point": {
            "q_r_policy": {
                "annual_demand": 7300.0,
            }
        }
    }


def test_extracts_newsvendor_number_lists() -> None:
    llm = create_mock_llm(
        json.dumps(
            {
                "model_inputs": {
                    "demand_scenarios": [5, 6, 7, 8],
                    "scenario_probabilities": [
                        0.2,
                        0.25,
                        0.3,
                        0.25,
                    ],
                }
            }
        )
    )
    agent = create_model_input_extraction_agent(llm)

    result = agent(
        {
            "selected_model_family": "newsvendor",
            "selected_model_variant": (
                "single_product_single_period"
            ),
            "raw_input": "Vier Nachfrageszenarien.",
        }
    )

    candidates = result["model_input_candidates"][
        "newsvendor"
    ]["single_product_single_period"]

    assert candidates["demand_scenarios"] == [
        5.0,
        6.0,
        7.0,
        8.0,
    ]
    assert candidates["scenario_probabilities"] == [
        0.2,
        0.25,
        0.3,
        0.25,
    ]


def test_accepts_json_response_wrapped_in_code_fence() -> None:
    code_fence = chr(96) * 3
    llm = create_mock_llm(
        (
            f"{code_fence}json\n"
            '{"model_inputs": {"annual_demand": 7300}}\n'
            f"{code_fence}"
        )
    )
    agent = create_model_input_extraction_agent(llm)

    result = agent(create_q_r_state())

    assert result["model_input_extraction_errors"] == []
    assert (
        result["model_input_candidates"]["reorder_point"]
        ["q_r_policy"]["annual_demand"]
        == 7300.0
    )


def test_rejects_unknown_model_field_without_blocking_form() -> None:
    llm = create_mock_llm(
        json.dumps(
            {
                "model_inputs": {
                    "annual_demand": 7300,
                    "budget": 1000,
                }
            }
        )
    )
    agent = create_model_input_extraction_agent(llm)

    result = agent(create_q_r_state())

    assert result["model_input_candidates"] == {
        "reorder_point": {
            "q_r_policy": {},
        }
    }
    assert len(result["model_input_extraction_errors"]) == 1
    assert "unbekannte Modellfelder" in (
        result["model_input_extraction_errors"][0]
    )


def test_rejects_invalid_field_type() -> None:
    llm = create_mock_llm(
        json.dumps(
            {
                "model_inputs": {
                    "annual_demand": "7300",
                }
            }
        )
    )
    agent = create_model_input_extraction_agent(llm)

    result = agent(create_q_r_state())

    assert result["model_input_candidates"][
        "reorder_point"
    ]["q_r_policy"] == {}
    assert len(result["model_input_extraction_errors"]) == 1


def test_skips_llm_when_description_is_empty() -> None:
    llm = create_mock_llm("This response must not be used.")
    agent = create_model_input_extraction_agent(llm)

    result = agent(create_q_r_state(raw_input=""))

    assert result["model_input_candidates"] == {
        "reorder_point": {
            "q_r_policy": {},
        }
    }
    assert result["model_input_extraction_errors"] == []
    llm.invoke.assert_not_called()
