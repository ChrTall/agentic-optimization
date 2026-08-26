import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agentic_optimization.agents.document_extraction_agent import (
    build_field_specs_from_variant,
    extract_document_parameters,
    format_parameter_display,
    run_decision_first_extraction,
)
from agentic_optimization.models.registry import MODEL_REGISTRY


def create_mock_llm(response: object) -> Mock:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(response),
    )
    return llm


# --- bestehende Tests (falls noch nicht vorhanden, ergänzen) ---

def test_extracts_parameters_from_document_text() -> None:
    llm = create_mock_llm([
        {
            "label": "Stückpreis",
            "value": "4.50",
            "unit": "€",
            "evidence": "...",
        }
    ])

    result = extract_document_parameters(
        llm, "Der Preis pro Einheit beträgt 4,50 €."
    )

    assert len(result) == 1
    assert result[0]["label"] == "Stückpreis"


def test_format_parameter_display_flags_low_confidence() -> None:
    parameter = {
        "field_name": "purchase_cost",
        "label": "Einkaufskosten",
        "value": "2.10",
        "unit": "€",
        "confidence": 0.4,
        "evidence": None,
    }

    display = format_parameter_display(parameter)

    assert "⚠️" in display


def test_format_parameter_display_no_warning_for_high_confidence() -> None:
    parameter = {
        "field_name": "selling_price",
        "label": "Verkaufspreis",
        "value": "4.50",
        "unit": "€",
        "confidence": 0.95,
        "evidence": None,
    }

    display = format_parameter_display(parameter)

    assert "⚠️" not in display


# --- neue Tests für Decision-first Extraction ---

def test_build_field_specs_includes_input_fields_and_constraints() -> None:
    variant = MODEL_REGISTRY["newsvendor"]["variants"][
        "single_product_single_period"
    ]

    specs = build_field_specs_from_variant(variant)
    names = [spec["name"] for spec in specs]

    assert "selling_price" in names
    assert any("." in name for name in names)  # Constraint-Parameter


def test_decision_first_extraction_finds_matching_model() -> None:
    requirement_response = {
        "decision": "order_quantity",
        "objective": "maximize_profit",
        "demand_type": "uncertain",
        "product_scope": "single_product",
        "planning_horizon": "single_period",
        "constraints": [],
        "unsupported_requirements": [],
    }
    targeted_response = [
        {
            "field_name": "selling_price",
            "label": "Verkaufspreis pro Einheit",
            "value": "4.50",
            "unit": "€",
            "confidence": 0.9,
            "evidence": "Der Preis beträgt 4,50 €.",
        }
    ]

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(requirement_response)),
        AIMessage(content=json.dumps(targeted_response)),
    ]

    result = run_decision_first_extraction(
        llm, "Der Preis pro Einheit beträgt 4,50 €."
    )

    assert result["decision"] == "order_quantity"
    assert result["matched_variant_name"] == (
        "Single Product, Single Period"
    )
    assert len(result["parameters"]) == 1
    assert llm.invoke.call_count == 2


def test_decision_first_extraction_falls_back_without_match() -> None:
    requirement_response = {
        "decision": None,
        "objective": None,
        "demand_type": None,
        "product_scope": None,
        "planning_horizon": None,
        "constraints": [],
        "unsupported_requirements": [],
    }
    generic_response = [
        {
            "label": "Irgendein Wert",
            "value": "1",
            "unit": None,
            "evidence": None,
        }
    ]

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps(requirement_response)),
        AIMessage(content=json.dumps(generic_response)),
    ]

    result = run_decision_first_extraction(llm, "Unklarer Text.")

    assert result["matched_variant_name"] is None
    assert len(result["parameters"]) == 1