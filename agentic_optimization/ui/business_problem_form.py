"""Fragenlogik für den modellneutralen Geschäftseinstieg."""

from typing import Literal, NotRequired, TypedDict

from agentic_optimization.models.problem_profile import (
    BusinessProblemProfile,
    BusinessProfileFieldId,
    ProfileField,
    create_profile_field,
)
from collections.abc import Collection
from copy import deepcopy


BusinessQuestionId = Literal[
    "recommendation",
    "procurement_pattern",
    "procurement_pattern_follow_up",
    "replenishment_during_period",
    "inventory_review",
    "product_scope",
    "primary_goal",
]

class BusinessQuestionOption(TypedDict):
    """Eine verständliche Antwortmöglichkeit."""

    option_id: str
    label: str
    help: NotRequired[str]
    profile_value: NotRequired[str]


class BusinessQuestion(TypedDict):
    """Eine mögliche Frage des adaptiven Einstiegs."""

    question_id: BusinessQuestionId
    target_field: BusinessProfileFieldId
    prompt: str
    options: list[BusinessQuestionOption]

UNKNOWN_OPTION: BusinessQuestionOption = {
    "option_id": "user_unknown",
    "label": "Ich bin mir nicht sicher",
}


BUSINESS_QUESTIONS: dict[
    BusinessQuestionId,
    BusinessQuestion,
] = {
    "recommendation": {
        "question_id": "recommendation",
        "target_field": "recommendation",
        "prompt": "Wobei sollen wir Sie unterstützen?",
        "options": [
            {
                "option_id": "order_quantity",
                "profile_value": "order_quantity",
                "label": "Bestellmenge bestimmen",
                "help": "Wie viele Einheiten sollen bestellt werden?",
            },
            {
                "option_id": "reorder_point",
                "profile_value": "reorder_point",
                "label": "Nachbestellzeitpunkt bestimmen",
                "help": "Wann soll eine neue Bestellung ausgelöst werden?",
            },
            {
                "option_id": "both",
                "profile_value": (
                    "order_quantity_and_reorder_point"
                ),
                "label": "Beides bestimmen",
            },
            UNKNOWN_OPTION,
        ],
    },
    "procurement_pattern": {
        "question_id": "procurement_pattern",
        "target_field": "procurement_pattern",
        "prompt": "Wie wird das Produkt normalerweise beschafft?",
        "options": [
            {
                "option_id": "one_time",
                "profile_value": "one_time",
                "label": "Einmal vor dem Verkaufszeitraum",
                "help": (
                    "Zum Beispiel für eine Saison, ein Ereignis "
                    "oder einen begrenzten Verkaufszeitraum."
                ),
            },
            {
                "option_id": "recurring",
                "profile_value": "recurring",
                "label": "Regelmäßig nachbestellt",
                "help": (
                    "Das Produkt wird dauerhaft benötigt oder verkauft."
                ),
            },
            UNKNOWN_OPTION,
        ],
    },
    "procurement_pattern_follow_up": {
        "question_id": "procurement_pattern_follow_up",
        "target_field": "procurement_pattern",
        "prompt": (
            "Wird derselbe Artikel nach der ersten Bestellung "
            "später erneut bestellt, wenn der Bestand "
            "verbraucht ist?"
        ),
        "options": [
            {
                "option_id": "recurring",
                "profile_value": "recurring",
                "label": "Ja, der Artikel wird erneut bestellt",
            },
            {
                "option_id": "one_time",
                "profile_value": "one_time",
                "label": (
                    "Nein, es bleibt bei dieser einen Bestellung"
                ),
            },
            UNKNOWN_OPTION,
        ],
    },
    "replenishment_during_period": {
        "question_id": "replenishment_during_period",
        "target_field": "replenishment_during_period",
        "prompt": (
            "Können Sie später noch Ware nachbestellen, wenn die "
            "Nachfrage höher ist als erwartet?"
        ),
        "options": [
            {
                "option_id": "not_possible",
                "profile_value": "not_possible",
                "label": "Nein, die Menge muss vorher feststehen",
            },
            {
                "option_id": "possible",
                "profile_value": "possible",
                "label": "Ja, Nachbestellungen sind möglich",
            },
            UNKNOWN_OPTION,
        ],
    },
    "inventory_review": {
        "question_id": "inventory_review",
        "target_field": "inventory_review",
        "prompt": "Wie wird der Lagerbestand überprüft?",
        "options": [
            {
                "option_id": "continuous",
                "profile_value": "continuous",
                "label": "Laufend",
                "help": "Der aktuelle Bestand ist jederzeit bekannt.",
            },
            {
                "option_id": "periodic",
                "profile_value": "periodic",
                "label": "In festen Zeitabständen",
                "help": (
                    "Zum Beispiel täglich, wöchentlich oder monatlich."
                ),
            },
            UNKNOWN_OPTION,
        ],
    },
    "product_scope": {
        "question_id": "product_scope",
        "target_field": "product_scope",
        "prompt": "Für welche Produkte benötigen Sie eine Empfehlung?",
        "options": [
            {
                "option_id": "single_product",
                "profile_value": "single_product",
                "label": "Für ein Produkt",
            },
            {
                "option_id": "multiple_products",
                "profile_value": "multiple_products",
                "label": "Für mehrere Produkte",
            },
            UNKNOWN_OPTION,
        ],
    },
    "primary_goal": {
        "question_id": "primary_goal",
        "target_field": "primary_goal",
        "prompt": "Was ist Ihnen hauptsächlich wichtig?",
        "options": [
            {
                "option_id": "minimize_cost",
                "profile_value": "minimize_cost",
                "label": "Gesamtkosten reduzieren",
            },
            {
                "option_id": "maximize_profit",
                "profile_value": "maximize_profit",
                "label": "Gewinn oder Deckungsbeitrag verbessern",
            },
            {
                "option_id": "avoid_stockouts",
                "profile_value": "avoid_stockouts",
                "label": "Fehlbestände möglichst vermeiden",
            },
            {
                "option_id": "reach_service_level",
                "profile_value": "reach_service_level",
                "label": "Eine gewünschte Lieferfähigkeit erreichen",
            },
            UNKNOWN_OPTION,
        ],
    },
}


def _needs_clarification(field: ProfileField[object]) -> bool:
    return field["status"] in {
        "missing",
        "user_unknown",
        "ambiguous",
        "conflict",
    }


def _known_value(field: ProfileField[object]) -> object:
    if field["status"] not in {"inferred", "confirmed"}:
        return None

    return field["value"]


def get_relevant_business_questions(
    profile: BusinessProblemProfile,
    asked_question_ids: Collection[str] = (),
) -> list[BusinessQuestion]:
    """Ermittelt offene und aktuell relevante Geschäftsfragen."""

    asked_question_id_set = set(asked_question_ids)
    question_ids: list[BusinessQuestionId] = []

    if _needs_clarification(profile["recommendation"]):
        question_ids.append("recommendation")

    if _needs_clarification(
        profile["procurement_pattern"]
    ):
        if (
            profile["procurement_pattern"]["status"]
            == "user_unknown"
            and "procurement_pattern"
            in asked_question_id_set
        ):
            if (
                "procurement_pattern_follow_up"
                not in asked_question_id_set
            ):
                question_ids.append(
                    "procurement_pattern_follow_up"
                )
        else:
            question_ids.append("procurement_pattern")

    procurement_pattern = _known_value(
        profile["procurement_pattern"]
    )
    recommendation = _known_value(profile["recommendation"])

    if (
        procurement_pattern == "one_time"
        and _needs_clarification(
            profile["replenishment_during_period"]
        )
    ):
        question_ids.append("replenishment_during_period")

    if (
        procurement_pattern == "recurring"
        and recommendation
        in {
            "reorder_point",
            "order_quantity_and_reorder_point",
        }
        and _needs_clarification(profile["inventory_review"])
    ):
        question_ids.append("inventory_review")

    product_ids = _known_value(profile["product_ids"])

    if (
        _needs_clarification(profile["product_scope"])
        and not product_ids
    ):
        question_ids.append("product_scope")

    if _needs_clarification(profile["primary_goal"]):
        question_ids.append("primary_goal")

    return [
        BUSINESS_QUESTIONS[question_id]
        for question_id in question_ids
    ]


def apply_business_question_answer(
    profile: BusinessProblemProfile,
    *,
    question_id: str,
    option_id: str,
    source_id: str,
) -> BusinessProblemProfile:
    """Übernimmt eine strukturierte Antwort in eine Profilkopie."""

    question = BUSINESS_QUESTIONS.get(question_id)

    if question is None:
        raise ValueError(
            f"Die Geschäftsfrage '{question_id}' ist "
            "nicht registriert."
        )

    selected_option = next(
        (
            option
            for option in question["options"]
            if option["option_id"] == option_id
        ),
        None,
    )

    if selected_option is None:
        raise ValueError(
            f"Die Antwortoption '{option_id}' gehört "
            f"nicht zur Geschäftsfrage '{question_id}'."
        )

    if not source_id.strip():
        raise ValueError(
            "Für eine Geschäftsantwort wird eine "
            "gültige Quellen-ID benötigt."
        )

    evidence = [
        {
            "source_id": source_id,
            "source_type": "user_selection",
            "reference": question_id,
        }
    ]

    if option_id == "user_unknown":
        updated_field = create_profile_field(
            status="user_unknown",
            evidence=evidence,
        )
    else:
        profile_value = selected_option.get(
            "profile_value"
        )

        if profile_value is None:
            raise ValueError(
                f"Die Antwortoption '{option_id}' besitzt "
                "keinen fachlichen Profilwert."
            )

        updated_field = create_profile_field(
            status="confirmed",
            value=profile_value,
            evidence=evidence,
        )

    updated_profile = deepcopy(profile)
    target_field = question["target_field"]
    updated_profile[target_field] = updated_field

    return updated_profile