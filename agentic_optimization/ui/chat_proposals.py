"""Vorschläge für Geschäftsangaben vor ihrer ausdrücklichen Übernahme."""

from agentic_optimization.ui.business_problem_form import BUSINESS_QUESTIONS

PENDING_KEY = "chat_pending_proposal"
EDIT_KEY = "chat_edit_proposal"

SENTENCES = {
    ("primary_goal", "minimize_cost"): "Sie möchten die Gesamtkosten reduzieren.",
    ("primary_goal", "maximize_profit"): "Sie möchten den Gewinn oder Deckungsbeitrag verbessern.",
    ("primary_goal", "avoid_stockouts"): "Sie möchten Fehlbestände möglichst vermeiden.",
    ("primary_goal", "reach_service_level"): "Sie möchten eine gewünschte Lieferfähigkeit erreichen.",
    ("recommendation", "order_quantity"): "Sie möchten die Bestellmenge bestimmen lassen.",
    ("recommendation", "reorder_point"): "Sie möchten den Meldebestand bestimmen lassen.",
    ("recommendation", "both"): "Sie möchten Bestellmenge und Meldebestand bestimmen lassen.",
    ("procurement_pattern", "recurring"): "Der Artikel wird regelmäßig nachbestellt.",
    ("procurement_pattern", "one_time"): "Die Bestellung erfolgt einmal vor dem Verkaufszeitraum.",
    ("replenishment_during_period", "possible"): "Während des Verkaufszeitraums sind Nachbestellungen möglich.",
    ("replenishment_during_period", "not_possible"): "Während des Verkaufszeitraums sind Nachbestellungen ausgeschlossen.",
    ("inventory_review", "continuous"): "Der Lagerbestand wird laufend überprüft.",
    ("inventory_review", "periodic"): "Der Lagerbestand wird in festen Zeitabständen überprüft.",
    ("product_scope", "single_product"): "Es wird ein Produkt betrachtet.",
    ("product_scope", "multiple_products"): "Es werden mehrere Produkte betrachtet.",
}


def proposal_sentence(answer, questions):
    pair = answer["question_id"], answer["option_id"]
    if pair in SENTENCES:
        return SENTENCES[pair]
    question = next((q for q in questions if q["question_id"] == pair[0]), None)
    if question:
        option = next((o for o in question["options"] if o["option_id"] == pair[1]), None)
        if option:
            return f"{question['prompt']} — {option['label']}"
    return "Diese Angabe soll noch geklärt werden."


def proposal_message(answers, questions):
    if not answers:
        return ("Ich habe Ihre Ergänzung als Beschreibung vorgemerkt. Soll ich sie übernehmen "
                "und die noch offenen Angaben mit Ihnen klären?")
    explicit = [a for a in answers if not a.get("is_assumption") and a["source_id"] != "chat_assistant_assumption"]
    assumed = [a for a in answers if a.get("is_assumption") or a["source_id"] == "chat_assistant_assumption"]
    parts = []
    if explicit:
        parts.append("Ich habe Ihre Angaben so verstanden:\n\n"
                     + "\n".join(f"- {proposal_sentence(a, questions)}" for a in explicit))
    if assumed:
        parts.append("Daraus würde ich außerdem folgende Annahmen ableiten – bitte prüfen Sie diese:\n\n"
                     + "\n".join(f"- {proposal_sentence(a, questions)}" for a in assumed))
    return "\n\n".join(parts) + "\n\nSoll ich diese Angaben übernehmen? Sie können sie ändern oder direkt weitere Angaben ergänzen."


def is_direct_answer(answers, current_question_id, pending):
    return (not pending and len(answers) == 1
            and answers[0]["question_id"] == current_question_id
            and answers[0]["source_id"] == "chat_assistant_verified")


def match_current_option(message, question):
    """Eine unveränderte Auswahlbezeichnung ist eine ausdrückliche Antwort."""
    if not question:
        return None

    def normalize(value):
        return " ".join(value.casefold().split()).rstrip(".!?")

    matches = [option for option in question["options"]
               if normalize(option["label"]) == normalize(message)]
    if len(matches) != 1:
        return None
    return {"question_id": question["question_id"], "option_id": matches[0]["option_id"],
            "source_id": "chat_assistant_verified", "excerpt": message}
