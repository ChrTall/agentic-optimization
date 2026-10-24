import pytest

from agentic_optimization.models.fact_extraction import merge_extracted_facts
from agentic_optimization.models.product_names import product_names


def test_names_fall_back_and_disambiguate_without_changing_ids():
    products = [{"product_id": "product_1"}, {"product_id": "produkt_2"},
                {"product_id": "SKU-42", "display_name": "  Weihnachtsbäume  "}]
    assert product_names(products) == {
        "product_1": "Produkt 1", "produkt_2": "Produkt 2", "SKU-42": "Weihnachtsbäume"}
    assert products[0]["product_id"] == "product_1"
    products[0]["display_name"] = "Weihnachtsbäume"
    labels = product_names(products)
    assert labels["product_1"] != labels["SKU-42"]


@pytest.mark.parametrize("name,quote,source,accepted", [
    ("Weihnachtsbäume", "Wir verkaufen Weihnachtsbäume.", "Wir verkaufen Weihnachtsbäume.", True),
    ("Palmen", "Wir verkaufen Weihnachtsbäume.", "Wir verkaufen Weihnachtsbäume.", False),
    ("Weihnachtsbäume", "Wir verkaufen Weihnachtsbäume.", "Wir bestellen einen Artikel.", False),
    ("Weihnachtsbäume", "", "Wir verkaufen Weihnachtsbäume.", False),
    ({"name": "Weihnachtsbäume"}, "Wir verkaufen Weihnachtsbäume.", "Wir verkaufen Weihnachtsbäume.", False),
])
def test_extracted_name_requires_a_literal_source(name, quote, source, accepted):
    original = [{"product_id": "product_1", "forecast": {}, "facts": {}}]
    products, _ = merge_extracted_facts(
        core_id="single_period_newsvendor", products=original, global_facts={},
        evidence_texts=[source], extraction={"products": [{
            "product_id": "product_1", "display_name": name, "display_name_evidence": quote,
        }]})
    assert (products[0].get("display_name") == "Weihnachtsbäume") == accepted
    assert products[0]["product_id"] == "product_1"
    assert "display_name" not in original[0]


def test_manual_name_is_never_overwritten_by_extraction():
    products, _ = merge_extracted_facts(
        core_id="eoq", products=[{"product_id": "product_1", "display_name": "Nordmanntannen"}],
        global_facts={}, evidence_texts=["Wir verkaufen Weihnachtsbäume."],
        extraction={"products": [{"product_id": "product_1", "display_name": "Weihnachtsbäume",
                                   "display_name_evidence": "Wir verkaufen Weihnachtsbäume."}]})
    assert products[0]["display_name"] == "Nordmanntannen"


def test_ambiguous_multi_product_names_are_not_assigned():
    source = "product_1 und product_2 sind Weihnachtsbäume und Ständer."
    products, _ = merge_extracted_facts(
        core_id="single_period_newsvendor", products=[{"product_id": "product_1"}, {"product_id": "product_2"}],
        global_facts={}, evidence_texts=[source], extraction={"products": [{
            "product_id": "product_1", "display_name": "Weihnachtsbäume", "display_name_evidence": source,
        }]})
    assert all("display_name" not in product for product in products)


@pytest.mark.parametrize("candidates,old_name,accepted", [
    ([{"product_id": "product_1", "status": "identified", "display_name": "Weihnachtsbäume",
       "display_name_evidence": "Wir verkaufen Weihnachtsbäume."}], None, "Weihnachtsbäume"),
    ([{"product_id": "product_1", "status": "unclear"}], "Weihnachtsbäume", None),
    ([{"product_id": "product_1", "status": "not_mentioned"}], None, None),
    ([{"product_id": "product_1", "status": "identified", "display_name": "Palmen",
       "display_name_evidence": "Wir verkaufen Palmen."}], None, None),
    ([{"product_id": "product_1", "status": "identified", "display_name": "Palmen",
       "display_name_evidence": "Wir verkaufen Weihnachtsbäume."}], "Weihnachtsbäume", None),
    ([{"product_id": "product_1", "status": "identified"},
      {"product_id": "product_1", "status": "identified"}], "Weihnachtsbäume", None),
    ([{"product_id": "unknown", "status": "identified", "display_name": "Weihnachtsbäume",
       "display_name_evidence": "Wir verkaufen Weihnachtsbäume."}], None, None),
])
def test_separate_name_results_reject_uncertain_conflicting_and_unproven_candidates(candidates, old_name, accepted):
    extracted = {"product_id": "product_1", "forecast": {"expected_value": 100}}
    if old_name:
        extracted.update(display_name=old_name, display_name_evidence="Wir verkaufen Weihnachtsbäume.")
    products, _ = merge_extracted_facts(
        core_id="single_period_newsvendor", products=[{"product_id": "product_1"}],
        global_facts={}, evidence_texts=["Wir verkaufen Weihnachtsbäume."],
        extraction={"products": [extracted], "product_names": candidates})
    assert products[0].get("display_name") == accepted
    assert products[0]["forecast"]["expected_value"] == 100
