"""Wiederaufnahme der Modellvorbereitung und sichere Fehlerbeschreibungen."""

from copy import deepcopy
from uuid import uuid4
import logging

from agentic_optimization.llm.errors import classify_llm_error, is_llm_service_error


STAGE_LABELS = {
    "business_analysis": "Beschreibung auswerten",
    "product_schema": "Produkteingabe vorbereiten",
    "facts": "Produktdaten und Angaben erfassen",
    "selection": "Modellbausteine auswählen",
    "integrity": "Bausteinauswahl prüfen",
    "coverage": "Anforderungen abgleichen",
    "final_integrity": "Ergänzte Bausteinauswahl prüfen",
    "specification": "Modellentwurf zusammenstellen",
    "validation": "Modellentwurf prüfen",
    "preparation": "Berechnung vorbereiten",
}


class ModelPreparationFailure(ValueError):
    def __init__(self, cause, stage, checkpoint):
        super().__init__(str(cause))
        self.cause = cause
        self.stage = stage
        self.checkpoint = deepcopy(checkpoint)


def preparation_failure_details(error, stage="preparation"):
    if isinstance(error, ModelPreparationFailure):
        stage, error = error.stage, error.cause
    kind = classify_llm_error(error) if is_llm_service_error(error) else (
        "invalid_response" if isinstance(error, (ValueError, TypeError)) else "unknown")
    messages = {
        "timeout": "Der KI-Dienst hat nicht rechtzeitig geantwortet.",
        "connection": "Die Verbindung zum KI-Dienst konnte nicht hergestellt werden.",
        "rate_limit": "Der KI-Dienst nimmt derzeit zu viele Anfragen entgegen. Bitte warten Sie kurz vor einem erneuten Versuch.",
        "authentication": "Der Zugriff auf den KI-Dienst wurde abgelehnt. Die Zugangskonfiguration muss geprüft werden.",
        "not_found": "Das konfigurierte KI-Modell ist beim Anbieter nicht verfügbar.",
        "invalid_response": "Die automatische Antwort konnte nicht zuverlässig verarbeitet werden.",
        "other": "Beim Aufruf des KI-Dienstes ist ein technischer Fehler aufgetreten.",
        "unknown": "Die Verarbeitung wurde durch einen technischen Fehler unterbrochen. Die genaue Ursache ist noch unklar.",
    }
    # Keine rohen Ausnahmetexte: Diese können Anbieterdaten, Eingaben oder
    # Zugangsdaten enthalten. Die Fehlerkennung genügt zur Zuordnung im Log.
    return {"reference": uuid4().hex[:12], "stage": stage,
            "stage_label": STAGE_LABELS.get(stage, STAGE_LABELS["preparation"]),
            "kind": kind, "message": messages[kind]}


def failed_preparation_state(state, error, stage="preparation"):
    from agentic_optimization.state import (
        WORKFLOW_MODEL_PREPARATION_FAILED, invalidate_optimization_state, with_control,
    )
    details = preparation_failure_details(error, stage)
    result = invalidate_optimization_state(state)
    model = {**result.get("modular_model", {}),
             "preparation_error": details["message"], "preparation_error_details": details}
    if isinstance(error, ModelPreparationFailure):
        model["preparation_checkpoint"] = error.checkpoint
    for key in ("proposal", "specification", "validation"):
        model.pop(key, None)
    result["modular_model"] = model
    logging.getLogger(__name__).error("Preparation failed [%s]: stage=%s kind=%s",
                                    details["reference"], details["stage"], details["kind"])
    return with_control(result, event=None, workflow_status=WORKFLOW_MODEL_PREPARATION_FAILED)
