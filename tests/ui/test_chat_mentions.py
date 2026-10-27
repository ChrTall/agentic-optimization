"""Feste Regeln für die Zuordnung zwischen Chat und Formular."""

import pytest

from agentic_optimization.ui.chat_mentions import (
    detect_component_mentions,
    value_conflicts_with_text,
)


@pytest.mark.parametrize(
    ("text", "component_id", "value"),
    [
        ("Es gibt eine Höchstbestellmenge von 250 Weihnachtsbäumen.",
         "maximum_order_quantity", 250),
        ("Wir dürfen höchstens 500 Stück bestellen.",
         "maximum_order_quantity", 500),
        ("We can order at most 250 trees.", "maximum_order_quantity", 250),
        ("Mindestbestellmenge sind 40 Stück", "minimum_order_quantity", 40),
        ("Unser Budget liegt bei 5.000 €.", "budget_limit", 5000),
        ("Bäume gibt es nur in 10er-Gebinden.", "order_multiple", 10),
        ("Wir haben Platz für 400 Bäume", "storage_capacity", 400),
        ("Servicegrad 95 %", "minimum_cycle_service_level", 95),
    ],
)
def test_detects_component_with_value(text, component_id, value) -> None:
    mentions = detect_component_mentions(text)

    assert [m["component_ids"][0] for m in mentions] == [component_id]
    assert mentions[0]["value"] == value


@pytest.mark.parametrize(
    "text",
    [
        "Wir verkaufen höchstens 300 Bäume.",
        "Es liegt ein Verkaufspreis von 25 Euro vor.",
        "Die Lieferzeit beträgt höchstens 5 Tage.",
    ],
)
def test_ignores_texts_without_component(text) -> None:
    assert detect_component_mentions(text) == []


def test_selling_price_is_not_purchase_cost() -> None:
    text = "Es liegt ein Verkaufspreis von 25 Euro vor."

    assert value_conflicts_with_text("purchase_cost", 25, text)
    assert not value_conflicts_with_text("selling_price", 25, text)


def test_several_values_in_one_message() -> None:
    text = "3500 Stück im Jahr, 120 € je Bestellung, 12 € Lager."

    assert not value_conflicts_with_text("ordering_cost", 120, text)
    assert not value_conflicts_with_text("holding_cost", 12, text)
    assert value_conflicts_with_text("holding_cost", 120, text)
    assert value_conflicts_with_text("ordering_cost", 12, text)


def test_neutral_or_missing_values_are_not_conflicts() -> None:
    assert not value_conflicts_with_text("purchase_cost", 25, "25 Euro")
    assert not value_conflicts_with_text("purchase_cost", 9, "Verkaufspreis 25 €")
    assert not value_conflicts_with_text(
        "purchase_cost", 10, "Wir kaufen für 10 € ein und verkaufen für 25 €"
    )


def test_money_value_does_not_fit_a_quantity_field() -> None:
    assert value_conflicts_with_text(
        "quantity", 3, "Es liegt ein Restwert von 3 Euro vor.",
        unit="Einheiten", component_id="maximum_order_quantity",
    )
    assert not value_conflicts_with_text(
        "quantity", 270, "Die Höchstbestellmenge beträgt 270 Weihnachtsbäume.",
        unit="Einheiten", component_id="maximum_order_quantity",
    )
    assert value_conflicts_with_text(
        "quantity", 250, "Die Höchstbestellmenge liegt bei 250 Stück.",
        unit="Einheiten", component_id="minimum_order_quantity",
    )


def test_quote_must_contain_the_number() -> None:
    from agentic_optimization.ui.chat_mentions import value_supported_by_quote

    assert not value_supported_by_quote(365, "jedes Jahr")
    assert not value_supported_by_quote(1, "Weihnachtsbäume")
    assert value_supported_by_quote(14, "2 Wochen Lieferzeit")
    assert value_supported_by_quote(250, "Höchstbestellmenge von 250")
