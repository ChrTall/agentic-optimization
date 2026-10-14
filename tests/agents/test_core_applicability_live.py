"""Optionale Live-Prüfung der fachlichen LLM-Auswertung.

Nur mit RUN_LIVE_CORE_APPLICABILITY=1 ausführen. Die Tests rufen den
konfigurierten LLM-Provider auf und können API-Kosten verursachen.
"""

import os

import pytest

from agentic_optimization.agents.core_applicability_agent import (
    create_core_applicability_agent,
)
from agentic_optimization.configuration.llm_config import (
    load_llm_configuration,
)
from agentic_optimization.llm.factory import create_chat_model


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_CORE_APPLICABILITY") != "1",
    reason="Live-LLM-Test nur nach expliziter Aktivierung",
)


@pytest.fixture(scope="module")
def agent():
    config = load_llm_configuration()
    return create_core_applicability_agent(create_chat_model(config))


@pytest.mark.parametrize(
    ("core_id", "description", "expected"),
    [
        (
            "eoq",
            "Der Jahresbedarf beträgt 3.500 Einheiten. Wir bestellen "
            "regelmäßig nach.",
            {"recurring_orders": "supported"},
        ),
        (
            "eoq",
            "Wir führen einen Artikel dauerhaft und bestellen regelmäßig "
            "nach. Der Jahresbedarf beträgt 3.500 Einheiten. Pro Bestellung "
            "entstehen Kosten von 120 Euro, die jährlichen Lagerhaltungskosten "
            "betragen 12 Euro je Einheit. Die Bestell- und Lagerhaltungskosten "
            "sollen minimiert werden. Bitte bestimme die Bestellmenge.",
            {"recurring_orders": "supported"},
        ),
        (
            "eoq",
            "Wir bestellen regelmäßig nach. Unser Lieferant gewährt "
            "Mengenrabatte abhängig von der Bestellmenge.",
            {"no_quantity_discounts": "contradicted"},
        ),
        (
            "single_period_newsvendor",
            "Für einen 30-tägigen Pop-up-Shop bestellen wir einmal "
            "vor Beginn. Während der 30 Tage ist keine Nachbestellung "
            "möglich.",
            {
                "single_decision_period": "supported",
                "no_replenishment_within_period": "supported",
            },
        ),
        (
            "continuous_review_q_r",
            "Wir prüfen den Lagerbestand nur jeden Freitag; "
            "die Lieferzeit schwankt von Woche zu Woche.",
            {
                "continuous_inventory_review": "contradicted",
                "constant_lead_time": "contradicted",
            },
        ),
    ],
)
def test_live_model_assumptions(agent, core_id, description, expected):
    result = agent(
        core_id=core_id,
        evidence=[{"source_id": "raw_input", "content": description}],
    )

    assert result["findings"] == expected
