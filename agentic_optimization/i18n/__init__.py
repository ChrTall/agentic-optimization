"""Zweisprachigkeit (Deutsch/Englisch).

Deutsche Texte bleiben im Code stehen und dienen als Schlüssel. ``t()``
liefert in der gewählten Sprache den passenden Text; die englischen
Übersetzungen liegen in ``agentic_optimization/i18n/en.py``.

Die Sprache wird pro Streamlit-Durchlauf über ``set_language`` gesetzt und
in einer ContextVar gehalten. LangGraph kopiert den Kontext in seine
Ausführung, sodass auch Workflowknoten die gewählte Sprache sehen.
"""

from contextvars import ContextVar
from typing import Any, Literal

Language = Literal["de", "en"]
SUPPORTED_LANGUAGES: tuple[Language, ...] = ("de", "en")
DEFAULT_LANGUAGE: Language = "de"

_LANGUAGE: ContextVar[Language] = ContextVar(
    "agentic_optimization_language",
    default=DEFAULT_LANGUAGE,
)


def set_language(language: str | None) -> None:
    """Setzt die Sprache für den aktuellen Durchlauf."""

    _LANGUAGE.set(language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE)


def get_language() -> Language:
    """Gibt die aktuell gewählte Sprache zurück."""

    return _LANGUAGE.get()


def _translations() -> dict[str, str]:
    from agentic_optimization.i18n.en import EN

    return EN


def t(text: Any, /, **params: Any) -> Any:
    """Übersetzt einen deutschen Text in die aktuelle Sprache.

    ``params`` füllt Platzhalter wie ``{count}`` aus. Fehlt eine
    Übersetzung, bleibt der deutsche Text erhalten. Andere Werte als Text
    (z. B. ``None``) werden unverändert zurückgegeben.
    """

    if not isinstance(text, str):
        return text

    template = text

    if get_language() == "en":
        template = _translations().get(text, text)

    if params:
        return template.format(**params)

    return template


def language_instruction() -> str:
    """Zusatz für Systemprompts, damit Modelltexte in der gewählten Sprache
    erscheinen. Auf Deutsch leer, weil die Prompts bereits deutsch sind."""

    if get_language() == "en":
        return (
            "\n\nWICHTIG – Sprache: Der Nutzer arbeitet auf Englisch. "
            "Formuliere alle Texte, die der Nutzer liest (z. B. reply, "
            "reason, description, explanation, Rückfragen), auf Englisch. "
            "IDs, Feldnamen und JSON-Schlüssel bleiben unverändert."
        )

    return ""
