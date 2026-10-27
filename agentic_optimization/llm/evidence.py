"""Tolerante Prüfung von Textbelegen aus Modellantworten.

Sprachmodelle zitieren Belege oft fast, aber nicht exakt wörtlich: andere
Anführungszeichen, ein anderer Bindestrich, abweichende Groß-/Kleinschreibung,
zusätzliche Leerzeichen oder ein Punkt am Ende. Eine strikte Prüfung
(``excerpt in text``) verwirft solche Belege, und die App fragt dann nach
etwas, das der Nutzer längst geschrieben hat.

``find_verbatim_excerpt`` sucht den Beleg tolerant und liefert den
**wörtlichen** Ausschnitt aus dem Originaltext zurück. Nachgelagerte Prüfungen
auf exakte Belege bleiben dadurch gültig.
"""

import re

_QUOTE_CHARS = "\"'„“”‚‘’«»"
_DASH_CHARS = "-‐‑‒–—"
_TRIM_CHARS = " \t\r\n.,;:!?" + _QUOTE_CHARS


def _token_pattern(token: str) -> str:
    parts: list[str] = []

    for char in token:
        if char in _QUOTE_CHARS:
            parts.append(f"[{re.escape(_QUOTE_CHARS)}]")
        elif char in _DASH_CHARS:
            parts.append(f"[{re.escape(_DASH_CHARS)}]")
        else:
            parts.append(re.escape(char))

    return "".join(parts)


def find_verbatim_excerpt(
    excerpt: object,
    text: str,
) -> str | None:
    """Findet ``excerpt`` tolerant in ``text``.

    Returns:
        Den passenden Ausschnitt wörtlich aus ``text`` oder ``None``.
    """

    if not isinstance(excerpt, str) or not isinstance(text, str):
        return None

    if excerpt and excerpt in text:
        return excerpt

    tokens = excerpt.strip(_TRIM_CHARS).split()

    if not tokens:
        return None

    pattern = r"\s+".join(_token_pattern(token) for token in tokens)
    match = re.search(pattern, text, flags=re.IGNORECASE)

    return match.group(0) if match else None
