from agentic_optimization.llm.evidence import find_verbatim_excerpt

TEXT = "Ein Baum kostet 10 €. Während der Saison können wir nicht nachbestellen."


def test_exact_match_is_returned_unchanged() -> None:
    assert find_verbatim_excerpt("kostet 10 €", TEXT) == "kostet 10 €"


def test_tolerates_case_whitespace_and_trailing_punctuation() -> None:
    assert find_verbatim_excerpt(
        "während der  saison können wir nicht nachbestellen.", TEXT
    ) == "Während der Saison können wir nicht nachbestellen"


def test_tolerates_quote_and_dash_variants() -> None:
    assert find_verbatim_excerpt(
        '"E-Bike"', "Wir verkaufen das „E–Bike“ dauerhaft."
    ) == "E–Bike"


def test_returns_none_for_invented_quote() -> None:
    assert find_verbatim_excerpt("kostet 12 €", TEXT) is None
    assert find_verbatim_excerpt(None, TEXT) is None
    assert find_verbatim_excerpt("   ", TEXT) is None
