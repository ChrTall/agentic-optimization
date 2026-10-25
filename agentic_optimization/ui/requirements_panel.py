"""Übersicht der erkannten Anforderungen und Annahmen.

Der Chat erkennt aus den Aussagen des Nutzers Anforderungen (z. B. das
Optimierungsziel) und Modellannahmen (z. B. „keine Mengenrabatte“). Diese
werden hier als Vorschläge angezeigt und können bestätigt, geändert oder
verworfen werden. Bestätigte Angaben bleiben sichtbar und änderbar.
"""

from typing import Any, Literal, TypedDict

import streamlit as st

from agentic_optimization.models.core_applicability import CONDITION_LABELS
from agentic_optimization.ui.business_problem_form import BUSINESS_QUESTIONS
from agentic_optimization.i18n import t

REQUIREMENT_FIELDS = (
    ("recommendation", "Gesuchte Empfehlung"),
    ("procurement_pattern", "Beschaffung"),
    ("replenishment_during_period", "Nachbestellung im Zeitraum"),
    ("inventory_review", "Bestandsüberprüfung"),
    ("product_scope", "Produktumfang"),
    ("primary_goal", "Ziel"),
)

ASSUMPTION_OPTIONS = (
    {"option_id": "supported", "label": "Trifft zu"},
    {"option_id": "contradicted", "label": "Trifft nicht zu"},
)


class RequirementItem(TypedDict):
    """Ein Eintrag der Übersicht."""

    target_id: str
    kind: Literal["requirement", "assumption"]
    title: str
    value: str | None
    value_label: str
    status: Literal["proposed", "confirmed"]
    quote: str | None
    options: list[dict[str, str]]


class RequirementChange(TypedDict):
    """Eine Korrektur, die an den Geschäftseinstieg geht."""

    target_id: str
    action: Literal["confirm", "set", "discard"]
    value: str | None


def _quote(evidence: list[dict[str, Any]]) -> str | None:
    for item in evidence or []:
        excerpt = item.get("excerpt")
        if isinstance(excerpt, str) and excerpt.strip():
            return excerpt.strip()
    return None


def _is_user_confirmed(field: dict[str, Any]) -> bool:
    return field.get("status") == "confirmed" or any(
        item.get("source_type") == "user_selection"
        for item in field.get("evidence") or []
    )


def build_requirement_items(
    intake_state: dict[str, Any] | None,
) -> list[RequirementItem]:
    """Erkannte und bestätigte Angaben als Liste für die Übersicht."""

    intake_state = intake_state or {}
    profile = intake_state.get("profile") or {}
    items: list[RequirementItem] = []

    for field_id, german_title in REQUIREMENT_FIELDS:
        title = t(german_title)
        field = profile.get(field_id) or {}

        if field.get("status") not in {"inferred", "confirmed"}:
            continue

        question = BUSINESS_QUESTIONS.get(field_id)
        question_options = [
            option
            for option in (question or {}).get("options", [])
            if option["option_id"] != "user_unknown"
        ]
        options = [
            {"option_id": option["option_id"], "label": t(option["label"])}
            for option in question_options
        ]
        # Antwort-IDs und gespeicherte Profilwerte können sich
        # unterscheiden (z. B. "both" -> "order_quantity_and_reorder_point").
        matching = next(
            (
                option
                for option in question_options
                if option.get("profile_value", option["option_id"])
                == field.get("value")
            ),
            None,
        )
        value = matching["option_id"] if matching else None
        value_label = (
            t(matching["label"]) if matching else str(field.get("value"))
        )
        items.append({
            "target_id": f"profile:{field_id}",
            "kind": "requirement",
            "title": title,
            "value": value,
            "value_label": value_label,
            "status": "confirmed" if _is_user_confirmed(field) else "proposed",
            "quote": _quote(field.get("evidence") or []),
            "options": options,
        })

    user_answers = intake_state.get("applicability_user_answers") or {}
    text_findings = intake_state.get("applicability_text_findings") or {}
    evidence_by_condition = {
        item.get("condition_id"): item.get("excerpt")
        for item in intake_state.get("applicability_text_evidence") or []
        if isinstance(item, dict)
    }
    condition_ids = [
        condition_id
        for condition_id in CONDITION_LABELS
        if condition_id in user_answers or condition_id in text_findings
    ]

    for condition_id in condition_ids:
        confirmed = condition_id in user_answers
        value = (
            user_answers[condition_id]
            if confirmed
            else text_findings[condition_id]
        )

        if value not in {"supported", "contradicted"}:
            continue

        items.append({
            "target_id": f"condition:{condition_id}",
            "kind": "assumption",
            "title": t(CONDITION_LABELS[condition_id]),
            "value": value,
            "value_label": (
                t("Trifft zu") if value == "supported" else t("Trifft nicht zu")
            ),
            "status": "confirmed" if confirmed else "proposed",
            "quote": evidence_by_condition.get(condition_id),
            "options": [
                {**option, "label": t(option["label"])}
                for option in ASSUMPTION_OPTIONS
            ],
        })

    return items


def _render_item(item: RequirementItem, *, key_prefix: str) -> RequirementChange | None:
    badge = (
        t("🟦 Annahme") if item["kind"] == "assumption" else t("🟩 Anforderung")
    )
    state = t("bestätigt") if item["status"] == "confirmed" else t("Vorschlag")

    if item["kind"] == "assumption":
        st.markdown(f"{badge} · _{state}_  \n{item['title']}")
    else:
        st.markdown(f"{badge} · _{state}_  \n**{item['title']}:** {item['value_label']}")

    if item["quote"]:
        st.caption(t("Aus Ihren Angaben: „{quote}“", quote=item['quote']))

    option_ids = [option["option_id"] for option in item["options"]]
    labels = {option["option_id"]: option["label"] for option in item["options"]}
    widget_key = f"{key_prefix}_{item['target_id']}_{item['value']}"
    columns = st.columns([3, 1, 1])

    with columns[0]:
        selected = st.selectbox(
            t("Ändern"),
            options=option_ids,
            index=option_ids.index(item["value"]) if item["value"] in option_ids else None,
            format_func=lambda option_id: labels[option_id],
            key=f"{widget_key}_select",
            label_visibility="collapsed",
        )

    if selected is not None and selected != item["value"]:
        return {"target_id": item["target_id"], "action": "set", "value": selected}

    with columns[1]:
        if item["status"] == "proposed" and st.button(
            "✓",
            key=f"{widget_key}_confirm",
            help=t("Bestätigen"),
            width="stretch",
        ):
            return {"target_id": item["target_id"], "action": "confirm", "value": None}

    with columns[2]:
        if st.button(
            "✕",
            key=f"{widget_key}_discard",
            help=t("Verwerfen"),
            width="stretch",
        ):
            return {"target_id": item["target_id"], "action": "discard", "value": None}

    return None


def render_requirements_panel(
    intake_state: dict[str, Any] | None,
    *,
    title: str,
    expanded: bool,
    note: str | None = None,
    key_prefix: str = "requirements",
) -> RequirementChange | None:
    """Zeigt die Übersicht als aufklappbaren Bereich.

    Liefert eine Korrektur, sobald der Nutzer etwas bestätigt, ändert oder
    verwirft.
    """

    items = build_requirement_items(intake_state)

    if not items:
        return None

    proposed = sum(item["status"] == "proposed" for item in items)
    suffix = t(" – {proposed} zu prüfen", proposed=proposed) if proposed else ""
    change: RequirementChange | None = None

    with st.expander(f"{title} ({len(items)}){suffix}", expanded=expanded):
        if note:
            st.caption(note)

        for kind, heading in (
            ("requirement", t("Anforderungen")),
            ("assumption", t("Annahmen für die Berechnung")),
        ):
            group = [item for item in items if item["kind"] == kind]

            if not group:
                continue

            st.markdown(f"##### {heading}")

            for item in group:
                result = _render_item(item, key_prefix=key_prefix)
                if result is not None and change is None:
                    change = result

    return change
