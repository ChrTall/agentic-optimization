"""Feste Regeln, damit Chat und Formular sauber zusammenpassen.

Sprachmodelle ordnen Werte gelegentlich dem falschen Feld zu (z. B. einen
genannten Verkaufspreis als Einkaufskosten) oder übersehen eine zusätzliche
Vorgabe wie „Höchstbestellmenge von 250“. Die Regeln hier sind bewusst
einfach und vorsichtig:

- ``value_conflicts_with_text`` erkennt, wenn ein Wert im Text nur zusammen
  mit dem Begriff einer **anderen** Größe vorkommt. Solche Zuordnungen
  werden verworfen.
- ``detect_component_mentions`` findet ausdrücklich genannte zusätzliche
  Vorgaben samt Zahlenwert.
"""

import re
from typing import Any, TypedDict

# Begriffe, an denen sich eine Größe im Text eindeutig erkennen lässt
# (deutsch und englisch, kleingeschrieben, als reguläre Ausdrücke).
FIELD_TERMS: dict[str, tuple[str, ...]] = {
    "selling_price": (
        r"\bverkaufs?preis", r"\bverkaufen?\b", r"\bverkauft\b",
        r"\bvk[- ]?preis", r"\bselling price", r"\bsale price",
        r"\bsell(?:s|ing)? (?:for|at)\b", r"\bretail price",
    ),
    "purchase_cost": (
        r"\beinkauf", r"\bbeschaffungs(?:kosten|preis)", r"\bstückkosten",
        r"\bkaufen\b", r"\bbezahlen\b", r"\bzahlen wir\b",
        r"\bpurchas", r"\bbuy(?:ing)?\b", r"\bunit cost", r"\bwe pay\b",
    ),
    "salvage_value": (
        r"\brestwert", r"\bunverkauft", r"\bübrig", r"\brücknahme",
        r"\bzurück(?:geben|nehmen)", r"\bsalvage", r"\bleftover",
        r"\bunsold", r"\bresidual",
    ),
    "disposal_cost": (r"\bentsorg", r"\bdispos"),
    "shortage_cost": (
        r"\bfehlmenge", r"\bfehlbestand", r"\bentgangen", r"\bshortage",
        r"\bstock-?outs?\b", r"\blost sales?\b",
    ),
    "ordering_cost": (
        r"\bbestellkosten", r"\b(?:je|pro|jede|einer?) bestellung",
        r"\bbestellfix", r"\bordering cost", r"\bper order\b",
        r"\border cost", r"\beach order\b",
    ),
    "holding_cost": (
        r"\blager(?:haltungs)?kosten", r"\blagerung", r"\blager\b",
        r"\bholding", r"\bstorage cost",
    ),
    "lead_time_days": (
        r"\blieferzeit", r"\bvorlaufzeit", r"\bwiederbeschaffungszeit",
        r"\blead ?time",
    ),
    "budget": (r"\bbudget",),
    "service_level_percent": (
        r"\bservicegrad", r"\bservice ?level", r"\blieferbereitschaft",
        r"\blieferfähigkeit",
    ),
}

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_SEGMENT_SPLIT = re.compile(
    r"(?<!\d)[.,](?!\d)|[;!?\n]|\.(?=\s)|\b(?:und|aber|sowie|and|but)\b",
    re.IGNORECASE,
)


def split_segments(text: str) -> list[str]:
    """Teilt Text in kurze Satzteile (Dezimalzahlen bleiben ganz)."""

    return [
        segment.strip()
        for segment in _SEGMENT_SPLIT.split(text or "")
        if segment and segment.strip()
    ]


def _number_candidates(raw: str) -> set[float]:
    candidates = set()

    for variant in (
        raw.replace(",", "."),
        raw.replace(".", "").replace(",", "."),
        raw.replace(",", ""),
    ):
        try:
            candidates.add(float(variant))
        except ValueError:
            continue

    return candidates


def numbers_in(text: str) -> list[float]:
    """Alle Zahlen im Text (deutsche und englische Schreibweise)."""

    values: list[float] = []

    for match in _NUMBER.finditer(text or ""):
        values.extend(_number_candidates(match.group(0)))

    return values


def _mentions(text: str, field_name: str) -> bool:
    lowered = text.lower()
    return any(
        re.search(pattern, lowered)
        for pattern in FIELD_TERMS.get(field_name, ())
    )


def _contains_value(text: str, value: float) -> bool:
    return any(abs(number - value) < 1e-9 for number in numbers_in(text))


def value_conflicts_with_text(
    field_name: str,
    value: Any,
    text: str,
) -> bool:
    """True, wenn ``value`` im Text nur bei einer anderen Größe steht.

    Beispiel: „Es liegt ein Verkaufspreis von 25 Euro vor.“ passt nicht zu
    ``purchase_cost`` = 25. Kommt der Wert gar nicht vor oder steht er
    neutral (ohne Begriff), gibt es keinen Konflikt.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False

    if field_name not in FIELD_TERMS:
        return False

    segments = [
        segment
        for segment in split_segments(text)
        if _contains_value(segment, float(value))
    ]

    if not segments:
        return False

    for segment in segments:
        if _mentions(segment, field_name):
            return False

        others = [
            other
            for other in FIELD_TERMS
            if other != field_name and _mentions(segment, other)
        ]

        if not others:
            return False

    return True


# ---------------------------------------------------------------------------
# Zusätzliche Vorgaben (Bausteine)
# ---------------------------------------------------------------------------

_MONEY = r"(?:€|\beuro\b|\beur\b|\bdollar|\$)"
_NOT_ORDER_LIMIT = re.compile(
    _MONEY + r"|budget|lager|platz|kapazität|capacity|storage|space|"
    r"servicegrad|service ?level|%|prozent|percent|meldebestand|reorder|"
    r"bestand|inventory|stock|lieferzeit|tage|days|wochen|weeks|"
    r"nachfrage|bedarf|demand|verkaufen|verkauft|sell",
    re.IGNORECASE,
)

# Reihenfolge = Priorität. Jeder Eintrag: (Bausteine in Wunschreihenfolge,
# Muster, Ausschlussmuster oder None).
_COMPONENT_RULES: tuple[tuple[tuple[str, ...], str, re.Pattern[str] | None], ...] = (
    (
        ("minimum_reorder_point",),
        r"mindestmeldebestand|meldebestand\w* (?:von )?(?:mindestens|minimal)"
        r"|minimum reorder point|reorder point of at least",
        None,
    ),
    (
        ("maximum_reorder_point",),
        r"höchstmeldebestand|meldebestand\w* (?:von )?(?:höchstens|maximal)"
        r"|maximum reorder point|reorder point of at most",
        None,
    ),
    (
        ("maximum_inventory_position", "storage_capacity"),
        r"maximale\w* bestandsposition|höchstbestand|maximale\w* bestand"
        r"|maximum inventory|max(?:imum)? stock",
        None,
    ),
    (
        ("storage_capacity", "maximum_inventory_position"),
        r"lagerkapazität|lagerplatz|lagerfläche|stellpl[aä]tze?|platz für"
        r"|storage capacity|warehouse capacity|space for",
        None,
    ),
    (
        ("budget_limit",),
        r"budget|(?:höchstens|maximal|nicht mehr als)\s*\d[\d.,]*\s*"
        + _MONEY + r"[^.]*ausgeben|spend (?:at most|no more than|up to)",
        None,
    ),
    (
        ("minimum_cycle_service_level",),
        r"servicegrad|service ?level|lieferbereitschaft|lieferfähigkeit"
        r"|wahrscheinlichkeit ohne fehlbestand|in \d+ ?% der fälle",
        None,
    ),
    (
        ("order_multiple",),
        r"verpackungseinheit|vielfache|gebinde|packungsgröße|packungseinheit"
        r"|(?:kartons?|paletten?|kisten?|packs?) (?:à|a|zu|mit|von|of)\s*\d"
        r"|\d+er[- ]?(?:pack|packung|gebinde|karton)|multiples? of"
        r"|pack size|case size|in packs of",
        None,
    ),
    (
        ("maximum_order_quantity",),
        r"höchstbestellmenge|maximale\w* bestellmenge|bestellobergrenze"
        r"|obergrenze|maximum order quantity|max(?:imum)?\.? order"
        r"|order (?:at most|no more than|up to)",
        None,
    ),
    (
        ("maximum_order_quantity",),
        r"(?:höchstens|maximal|max\.|nicht mehr als|bis zu|at most"
        r"|no more than|up to)\s*\d",
        _NOT_ORDER_LIMIT,
    ),
    (
        ("minimum_order_quantity",),
        r"mindestbestellmenge|mindestabnahme|mindestmenge"
        r"|minimale\w* bestellmenge|minimum order|\bmoq\b"
        r"|order at least",
        None,
    ),
    (
        ("minimum_order_quantity",),
        r"(?:mindestens|minimal|nicht weniger als|at least|no less than)\s*\d",
        _NOT_ORDER_LIMIT,
    ),
)


class ComponentMention(TypedDict):
    """Eine im Text genannte zusätzliche Vorgabe."""

    component_ids: tuple[str, ...]
    value: float | None
    segment: str


def _value_near(segment: str, position: int) -> float | None:
    after = _NUMBER.search(segment, position)
    match = after or next(iter(reversed(list(_NUMBER.finditer(segment)))), None)

    if match is None:
        return None

    raw = match.group(0)
    # „1.000“ als Tausender lesen, „12,5“ als Dezimalzahl.
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):
        return float(raw.replace(".", ""))

    return float(raw.replace(",", "."))


def detect_component_mentions(text: str) -> list[ComponentMention]:
    """Findet ausdrücklich genannte zusätzliche Vorgaben im Text.

    Pro Satzteil zählt höchstens eine Vorgabe (die erste passende Regel).
    """

    mentions: list[ComponentMention] = []

    for segment in split_segments(text):
        lowered = segment.lower()

        for component_ids, pattern, exclude in _COMPONENT_RULES:
            match = re.search(pattern, lowered)

            if match is None:
                continue

            if exclude is not None and exclude.search(lowered):
                continue

            mentions.append({
                "component_ids": component_ids,
                "value": _value_near(lowered, match.start()),
                "segment": segment,
            })
            break

    return mentions
