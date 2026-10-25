import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentic_optimization.agents.chat_intake_validation import validate_chat_intake
from agentic_optimization.ui.business_problem_form import BUSINESS_QUESTIONS
from agentic_optimization.models.problem_profile import create_empty_business_problem_profile, create_profile_field


def validate(message, candidates, findings, profile=None, current="primary_goal"):
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({"findings": findings}))
    return validate_chat_intake(llm, message=message,
        questions=list(BUSINESS_QUESTIONS.values()),
        proposed_answers=candidates, profile=profile or create_empty_business_problem_profile(),
        current_question_id=current)


def goal_profile():
    profile = create_empty_business_problem_profile()
    profile["primary_goal"] = create_profile_field(status="confirmed", value="maximize_profit",
        evidence=[{"source_id": "form", "source_type": "user_selection"}])
    return profile


@pytest.mark.parametrize("finding", [
    {"question_id": "procurement_pattern", "option_id": "one_time", "excerpt": "jedes Jahr", "verdict": "unclear", "intent": "answer"},
    {"question_id": "procurement_pattern", "option_id": "one_time", "excerpt": "einmal im Voraus", "verdict": "explicit", "intent": "answer"},
])
def test_proposal_without_semantic_or_literal_evidence_is_not_accepted(finding):
    assert validate("Wir verkaufen jedes Jahr Ware.",
                    [{"question_id": "procurement_pattern", "option_id": "one_time"}], [finding]) == []


def test_known_goal_is_not_silently_overwritten_even_if_reviewer_calls_it_correction():
    answer = {"question_id": "primary_goal", "option_id": "minimize_cost"}
    findings = [{**answer, "excerpt": "Kosten minimieren", "verdict": "explicit", "intent": "correction"}]
    result = validate("Kosten minimieren", [answer], findings, goal_profile())
    assert result[0]["source_id"] == "chat_assistant_conflict"


def test_both_checks_must_agree_on_an_explicit_correction():
    answer = {"question_id": "primary_goal", "option_id": "minimize_cost", "change_intent": "correction"}
    findings = [{**answer, "excerpt": "Statt Gewinn bitte Kosten minimieren", "verdict": "explicit", "intent": "correction"}]
    result = validate("Statt Gewinn bitte Kosten minimieren", [answer], findings, goal_profile())
    assert result[0]["source_id"] == "chat_assistant_correction"


def test_verification_can_detect_a_known_field_change_missing_from_proposals():
    finding = {"question_id": "primary_goal", "option_id": "minimize_cost", "excerpt": "Kosten minimieren",
               "verdict": "explicit", "intent": "conflict"}
    result = validate("Kosten minimieren", [], [finding], goal_profile())
    assert result[0]["source_id"] == "chat_assistant_conflict"


def test_contradictory_verification_results_cannot_select_the_first_answer():
    findings = [{"question_id": "primary_goal", "option_id": value, "excerpt": "Gewinn oder Kosten",
                 "verdict": "explicit", "intent": "answer"} for value in ["maximize_profit", "minimize_cost"]]
    assert validate("Gewinn oder Kosten", [{"question_id": "primary_goal", "option_id": "maximize_profit"}], findings) == []


def test_unclear_additional_goal_triggers_clarification_without_updating_it():
    answer = {"question_id": "primary_goal", "option_id": "minimize_cost"}
    findings = [{**answer, "excerpt": "Gewinn maximieren, aber auch Kosten beachten",
                 "verdict": "unclear", "intent": "conflict"}]
    result = validate("Gewinn maximieren, aber auch Kosten beachten", [answer], findings, goal_profile())
    assert result[0]["source_id"] == "chat_assistant_conflict"


@pytest.mark.parametrize("message", [
    "Wir verkaufen jedes Jahr zur Weihnachtszeit Weihnachtsbäume und müssen die Bestellmenge für dieses Jahr planen.",
    "Wir verkaufen im Sommer Eis und planen die Bestellmenge.",
    "Wir planen die jährliche Bestellung für unsere Veranstaltung.",
])
def test_initial_structural_guesses_are_blocked_even_if_both_llms_agree(message):
    answer = {"question_id": "procurement_pattern", "option_id": "one_time"}
    finding = {**answer, "excerpt": message, "verdict": "explicit", "intent": "answer"}
    assert validate(message, [answer], [finding], current="recommendation") == []


def test_focused_procurement_answer_can_be_saved_after_targeted_question():
    message = "Wir legen die gesamte Menge einmal im Voraus fest."
    answer = {"question_id": "procurement_pattern", "option_id": "one_time"}
    finding = {**answer, "excerpt": message, "verdict": "explicit", "intent": "answer"}
    result = validate(message, [answer], [finding], current="core_condition:single_decision_period")
    assert result[0]["source_id"] == "chat_assistant_verified"


@pytest.mark.parametrize("basis, accepted", [
    ("direct_statement", True), ("context_inference", False), ("unclear", False),
])
def test_unsolicited_explicit_ordering_rule_requires_direct_evidence(basis, accepted):
    message = "Sobald der Meldebestand erreicht wird, bestellen wir regelmäßig nach."
    answer = {"question_id": "procurement_pattern", "option_id": "recurring"}
    finding = {**answer, "excerpt": message, "verdict": "explicit",
               "intent": "answer", "evidence_basis": basis}
    result = validate(message, [answer], [finding], current="primary_goal")
    assert bool(result) is accepted


def test_omitted_direct_fact_is_checked_again_before_accepting():
    message = "Wir bestellen regelmäßig nach."
    finding = {"question_id": "procurement_pattern", "option_id": "recurring",
               "excerpt": message, "verdict": "explicit", "intent": "answer",
               "evidence_basis": "direct_statement"}
    result = validate(message, [], [finding])
    assert result[0]["option_id"] == "recurring"


def test_failed_completion_keeps_only_originally_verified_answers():
    message = "Wir möchten die Kosten minimieren."
    finding = {"question_id": "primary_goal", "option_id": "minimize_cost",
               "excerpt": message, "verdict": "explicit", "intent": "answer",
               "evidence_basis": "direct_statement"}
    llm = Mock()
    llm.invoke.side_effect = [AIMessage(content=json.dumps({"findings": [finding]})),
                              RuntimeError("unavailable")]
    assert validate_chat_intake(llm, message=message,
        questions=[BUSINESS_QUESTIONS["primary_goal"]], proposed_answers=[],
        profile=create_empty_business_problem_profile(), current_question_id=None) == []
    assert llm.invoke.call_count == 2


@pytest.mark.parametrize("message", [
    "Ein Laden prüft ein T-Shirt-Modell laufend und möchte Bestell-, Lager- und Fehlmengenkosten minimieren. Beim Meldebestand wird eine feste Menge bestellt.",
    "Für einen einzelnen Ersatzteilartikel wollen wir die jährlichen Beschaffungs- und Lagerkosten senken. Unterschreitet der Bestand die Grenze, wird eine feste Losgröße bestellt.",
    "Wir überwachen genau eine Kaffeesorte kontinuierlich und möchten die gesamten Kosten reduzieren. Bei Erreichen der Grenze bestellen wir die festgelegte Menge nach.",
])
def test_described_policy_does_not_authorize_optimization_but_explicit_facts_are_kept(message):
    answers = [{"question_id": qid, "option_id": oid} for qid, oid in [
        ("primary_goal", "minimize_cost"), ("product_scope", "single_product"),
        ("recommendation", "both")]]
    findings = [{**answer, "excerpt": message, "verdict": "explicit", "intent": "answer",
                 "evidence_basis": "direct_statement"} for answer in answers]
    # Use the registered product-scope ID rather than an invented option.
    scope = BUSINESS_QUESTIONS["product_scope"]["options"][0]["option_id"]
    answers[1]["option_id"] = findings[1]["option_id"] = scope
    result = validate(message, answers, findings)
    assert {item["question_id"] for item in result} == {"primary_goal", "product_scope"}


@pytest.mark.parametrize("option, message", [
    ("order_quantity", "Bitte bestimmen Sie die optimale Bestellmenge."),
    ("reorder_point", "Bitte berechnen Sie den passenden Meldebestand."),
    ("both", "Bitte optimieren Sie Bestellmenge und Meldebestand."),
])
def test_explicit_calculation_request_can_be_accepted(option, message):
    answer = {"question_id": "recommendation", "option_id": option}
    finding = {**answer, "excerpt": message, "verdict": "explicit", "intent": "answer",
               "evidence_basis": "direct_request"}
    assert validate(message, [answer], [finding])[0]["option_id"] == option


@pytest.mark.parametrize("message, bad_quote, exact_quote", [
    ("Wir möchten unsere jährlichen Bestell-, Lagerhaltungs- und Fehlmengenkosten minimieren.",
     "jährliche Bestell-, Lagerhaltungs- und Fehlmengenkosten minimieren",
     "jährlichen Bestell-, Lagerhaltungs- und Fehlmengenkosten minimieren"),
    ("Wir wollen die gesamten Beschaffungskosten senken.",
     "gesamte Beschaffungskosten senken", "gesamten Beschaffungskosten senken"),
])
def test_grammatically_changed_quote_is_repaired_and_validated(message, bad_quote, exact_quote):
    answer = {"question_id": "primary_goal", "option_id": "minimize_cost"}
    finding = {**answer, "excerpt": bad_quote, "verdict": "explicit", "intent": "answer",
               "evidence_basis": "direct_statement"}
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"findings": [finding]})),
        AIMessage(content=json.dumps({"excerpts": [{**answer, "excerpt": exact_quote}]})),
    ]
    result = validate_chat_intake(llm, message=message,
        questions=[BUSINESS_QUESTIONS["primary_goal"]], proposed_answers=[answer],
        profile={}, current_question_id=None)
    assert result == [{**answer, "source_id": "chat_assistant_verified", "excerpt": exact_quote}]
    assert llm.invoke.call_count == 2


@pytest.mark.parametrize("repair", [
    {"question_id": "primary_goal", "option_id": "minimize_cost", "excerpt": "Kosten senken"},
    {"question_id": "primary_goal", "option_id": "maximize_profit", "excerpt": "Kosten minimieren"},
])
def test_quote_repair_cannot_invent_quotes_or_change_answers(repair):
    answer = {"question_id": "primary_goal", "option_id": "minimize_cost"}
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"findings": [{**answer, "excerpt": "falsches Zitat",
            "verdict": "explicit", "intent": "answer", "evidence_basis": "direct_statement"}]})),
        AIMessage(content=json.dumps({"excerpts": [repair]})),
    ]
    assert validate_chat_intake(llm, message="Kosten minimieren",
        questions=[BUSINESS_QUESTIONS["primary_goal"]], proposed_answers=[answer],
        profile={}, current_question_id=None) == []
    assert llm.invoke.call_count == 2


def test_incomplete_reviewer_response_rechecks_only_omitted_proposals():
    message = "Für einen Artikel prüfen wir laufend den Bestand und wollen die Kosten minimieren."
    answers = [{"question_id": qid, "option_id": oid} for qid, oid in [
        ("primary_goal", "minimize_cost"), ("product_scope", "single_product"),
        ("inventory_review", "continuous")]]
    def finding(answer):
        return {**answer, "excerpt": message, "verdict": "explicit", "intent": "answer",
                "evidence_basis": "direct_statement"}
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"findings": [finding(
            {"question_id": "recommendation", "option_id": "both"})]})),
        AIMessage(content=json.dumps({"findings": [finding(a) for a in answers]})),
    ]
    result = validate_chat_intake(llm, message=message,
        questions=list(BUSINESS_QUESTIONS.values()), proposed_answers=answers,
        profile={}, current_question_id=None)
    assert {a["question_id"] for a in result} == {a["question_id"] for a in answers}
    assert llm.invoke.call_count == 2
    retry = json.loads(llm.invoke.call_args.args[0][1].content)
    assert {q["question_id"] for q in retry["questions"]} == set(BUSINESS_QUESTIONS) - {"recommendation"}


def test_unclear_review_is_not_retried_as_if_it_were_omitted():
    answer = {"question_id": "primary_goal", "option_id": "minimize_cost"}
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({"findings": [{**answer,
        "excerpt": "Kosten beachten", "verdict": "unclear", "intent": "answer",
        "evidence_basis": "unclear"}]}))
    assert validate_chat_intake(llm, message="Kosten beachten",
        questions=[BUSINESS_QUESTIONS["primary_goal"]], proposed_answers=[answer],
        profile={}, current_question_id=None) == []
    assert llm.invoke.call_count == 1


def test_repeated_review_omission_stops_after_one_retry():
    llm = Mock()
    llm.invoke.return_value = AIMessage(content='{"findings": []}')
    assert validate_chat_intake(llm, message="Wir möchten die Kosten minimieren.",
        questions=list(BUSINESS_QUESTIONS.values()),
        proposed_answers=[{"question_id": "primary_goal", "option_id": "minimize_cost"}],
        profile={}, current_question_id=None) == []
    assert llm.invoke.call_count == 2


def test_complete_not_mentioned_assessments_do_not_create_user_answers_or_retry():
    questions = list(BUSINESS_QUESTIONS.values())
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({"findings": [
        {"question_id": q["question_id"], "option_id": None, "verdict": "not_mentioned",
         "excerpt": "", "intent": "answer", "evidence_basis": "unclear"}
        for q in questions]}))
    assert validate_chat_intake(llm, message="Hallo", questions=questions,
        proposed_answers=[], profile={}, current_question_id=None) == []
    assert llm.invoke.call_count == 1


def test_characteristic_missing_from_both_initial_outputs_is_still_checked():
    message = "Wir betrachten genau einen Artikel."
    finding = {"question_id": "product_scope", "option_id": "single_product",
               "excerpt": message, "intent": "answer", "verdict": "explicit",
               "evidence_basis": "direct_statement"}
    llm = Mock()
    llm.invoke.side_effect = [AIMessage(content='{"findings": []}'),
        AIMessage(content=json.dumps({"findings": [finding]})),
        AIMessage(content=json.dumps({"findings": [finding]}))]
    result = validate_chat_intake(llm, message=message,
        questions=[BUSINESS_QUESTIONS["product_scope"]], proposed_answers=[],
        profile={}, current_question_id=None)
    assert result[0]["option_id"] == "single_product"
    assert llm.invoke.call_count == 3


def test_conflicting_mentioned_and_not_mentioned_assessments_do_not_confirm():
    message = "Kosten minimieren"
    answer = {"question_id": "primary_goal", "option_id": "minimize_cost"}
    findings = [{**answer, "excerpt": message, "verdict": "explicit", "intent": "answer",
                 "evidence_basis": "direct_statement"},
                {"question_id": "primary_goal", "option_id": None, "excerpt": "",
                 "verdict": "not_mentioned", "intent": "answer", "evidence_basis": "unclear"}]
    assert validate(message, [answer], findings) == []


def test_plausible_inference_is_available_only_as_confirmable_assumption():
    from agentic_optimization.ui.chat_proposals import is_direct_answer, proposal_message
    message = "Nachbestellungen sind ausgeschlossen; wir müssen vorher planen."
    answer = {"question_id": "procurement_pattern", "option_id": "one_time"}
    finding = {**answer, "excerpt": message, "verdict": "explicit", "intent": "answer",
               "evidence_basis": "context_inference"}
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({"findings": [finding]}))
    questions = [BUSINESS_QUESTIONS["procurement_pattern"]]
    kwargs = dict(message=message, questions=questions, proposed_answers=[],
                  profile={}, current_question_id=None)
    assert validate_chat_intake(llm, **kwargs) == []
    proposal = validate_chat_intake(llm, **kwargs, allow_assumptions=True)
    assert proposal[0]["is_assumption"] is True
    assert "Annahmen" in proposal_message(proposal, questions)
    assert not is_direct_answer(proposal, "procurement_pattern", None)


def test_explicit_calculation_request_omitted_by_actor_is_proposed_without_inference_label():
    message = "Bitte berechnen Sie die optimale Bestellmenge."
    finding = {"question_id": "recommendation", "option_id": "order_quantity", "excerpt": message,
               "verdict": "explicit", "intent": "answer", "evidence_basis": "direct_request"}
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=json.dumps({"findings": [finding]}))
    result = validate_chat_intake(llm, message=message,
        questions=[BUSINESS_QUESTIONS["recommendation"]], proposed_answers=[], profile={},
        current_question_id=None, allow_assumptions=True)
    assert result[0]["option_id"] == "order_quantity"
    assert not result[0].get("is_assumption")
    assert llm.invoke.call_count == 1
