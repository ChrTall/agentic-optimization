"""Erkennung und Beschreibung von Fehlern der LLM-Anbieter.

Die SDKs von OpenAI, Anthropic und Groq werfen eigene Ausnahmen
(Rate-Limit, ungültiger API-Key, Timeout, Verbindungsabbruch). Diese erben
nicht von ``ValueError`` und würden ohne gesonderte Behandlung als
Traceback in der Oberfläche landen.
"""

import importlib

_ERROR_KINDS = (
    ("RateLimitError", "rate_limit"),
    ("AuthenticationError", "authentication"),
    ("PermissionDeniedError", "authentication"),
    ("APITimeoutError", "timeout"),
    ("APIConnectionError", "connection"),
    ("NotFoundError", "not_found"),
)

_MESSAGES = {
    "rate_limit": (
        "Der Sprachmodell-Anbieter hat zu viele Anfragen gemeldet "
        "(Rate-Limit). Bitte warten Sie kurz und versuchen Sie es erneut."
    ),
    "authentication": (
        "Der Sprachmodell-Anbieter hat den API-Key abgelehnt. "
        "Bitte prüfen Sie LLM_API_KEY und LLM_PROVIDER in der .env-Datei."
    ),
    "timeout": (
        "Das Sprachmodell hat nicht rechtzeitig geantwortet. "
        "Bitte versuchen Sie es erneut."
    ),
    "connection": (
        "Das Sprachmodell ist gerade nicht erreichbar. "
        "Bitte prüfen Sie die Internetverbindung und versuchen Sie es erneut."
    ),
    "not_found": (
        "Das konfigurierte Sprachmodell wurde beim Anbieter nicht gefunden. "
        "Bitte prüfen Sie LLM_MODEL in der .env-Datei."
    ),
    "other": (
        "Beim Aufruf des Sprachmodells ist ein Fehler aufgetreten. "
        "Bitte versuchen Sie es erneut."
    ),
}


def _load_provider_errors() -> tuple[type[BaseException], ...]:
    """Sammelt die Basisklassen der API-Fehler aller installierten SDKs."""

    errors: list[type[BaseException]] = []

    for module_name in ("openai", "anthropic", "groq"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        error_class = getattr(module, "APIError", None)

        if isinstance(error_class, type) and issubclass(
            error_class, BaseException
        ):
            errors.append(error_class)

    return tuple(errors)


LLM_SERVICE_ERRORS: tuple[type[BaseException], ...] = _load_provider_errors()


def is_llm_service_error(error: BaseException) -> bool:
    """Prüft, ob ein Fehler vom LLM-Anbieter stammt."""

    return bool(LLM_SERVICE_ERRORS) and isinstance(error, LLM_SERVICE_ERRORS)


def classify_llm_error(error: BaseException) -> str:
    """Ordnet einen Anbieterfehler einer groben Fehlerart zu."""

    class_names = {cls.__name__ for cls in type(error).__mro__}

    for class_name, kind in _ERROR_KINDS:
        if class_name in class_names:
            return kind

    return "other"


def describe_llm_error(error: BaseException) -> str:
    """Liefert eine verständliche deutsche Fehlermeldung."""

    return _MESSAGES[classify_llm_error(error)]
