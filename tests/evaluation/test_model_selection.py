from agentic_optimization.evaluation.model_selection import (
    classify_outcome,
    evaluate,
    format_report,
    load_cases,
    run_case_once,
)
from agentic_optimization.models.core_registry import CORE_REGISTRY

_NON_CORE_OUTCOMES = {"needs_clarification", "unsupported"}


def test_case_file_is_valid() -> None:
    cases = load_cases()

    assert len(cases) >= 20
    for case in cases:
        assert case["description"].strip()
        for accepted in case["accepted"]:
            assert accepted in CORE_REGISTRY or accepted in _NON_CORE_OUTCOMES

    covered = {
        accepted for case in cases for accepted in case["accepted"]
    }
    assert set(CORE_REGISTRY) <= covered


def test_classifies_selected_core() -> None:
    state = {
        "control": {"workflow_status": "business_core_selected"},
        "business_intake": {
            "core_selection": {"status": "selected", "selected_core_id": "eoq"}
        },
    }

    assert classify_outcome(state) == "eoq"


def test_classifies_other_outcomes() -> None:
    def state(status: str) -> dict:
        return {"control": {"workflow_status": status}}

    assert classify_outcome(state("needs_business_profile_input")) == (
        "needs_clarification"
    )
    assert classify_outcome(state("business_case_unsupported")) == "unsupported"
    assert classify_outcome(state("business_profile_analysis_failed")) == (
        "analysis_failed"
    )


def test_run_case_once_reports_errors() -> None:
    class BrokenWorkflow:
        def invoke(self, state):
            raise RuntimeError("API down")

    assert run_case_once(BrokenWorkflow(), "Text") == "error:RuntimeError"


def test_evaluate_and_report() -> None:
    cases = [
        {"id": "a", "description": "A", "accepted": ["eoq"]},
        {"id": "b", "description": "B", "accepted": ["unsupported"]},
    ]
    answers = iter(["eoq", "eoq", "unsupported", "needs_clarification"])

    results = evaluate(cases, lambda _: next(answers), repeats=2)
    report = format_report(results)

    assert [r["hits"] for r in results] == [2, 1]
    assert "Trefferquote: 3/4 = 75%" in report
    assert "Stabil (jeder Durchlauf gleich): 1/2 Fälle" in report
