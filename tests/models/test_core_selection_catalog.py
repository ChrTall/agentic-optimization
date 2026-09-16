from agentic_optimization.models.core_selection_catalog import (
    build_core_selection_catalog,
)
from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
    create_profile_field,
)


def confirmed(value):
    return create_profile_field(
        status="confirmed",
        value=value,
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )


def candidate_ids(profile) -> list[str]:
    return [
        candidate["core_id"]
        for candidate in build_core_selection_catalog(
            profile
        )
    ]


def test_empty_profile_keeps_all_registered_candidates() -> None:
    profile = create_empty_business_problem_profile()

    assert candidate_ids(profile) == [
        "single_period_newsvendor",
        "eoq",
        "continuous_review_q_r",
    ]


def test_one_time_non_replenishable_case_selects_newsvendor() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity"
    )
    profile["procurement_pattern"] = confirmed(
        "one_time"
    )
    profile["replenishment_during_period"] = confirmed(
        "not_possible"
    )

    assert candidate_ids(profile) == [
        "single_period_newsvendor"
    ]


def test_recurring_order_quantity_case_selects_eoq() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity"
    )
    profile["procurement_pattern"] = confirmed(
        "recurring"
    )

    assert candidate_ids(profile) == ["eoq"]


def test_continuous_reorder_case_selects_q_r() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity_and_reorder_point"
    )
    profile["procurement_pattern"] = confirmed(
        "recurring"
    )
    profile["inventory_review"] = confirmed(
        "continuous"
    )

    assert candidate_ids(profile) == [
        "continuous_review_q_r"
    ]


def test_periodic_review_has_no_supported_candidate() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity_and_reorder_point"
    )
    profile["procurement_pattern"] = confirmed(
        "recurring"
    )
    profile["inventory_review"] = confirmed(
        "periodic"
    )

    assert candidate_ids(profile) == []


def test_conflict_does_not_prematurely_remove_candidates() -> None:
    profile = create_empty_business_problem_profile()
    profile["procurement_pattern"] = (
        create_profile_field(
            status="conflict",
            evidence=[
                {
                    "source_id": "form_1",
                    "source_type": "user_selection",
                },
                {
                    "source_id": "raw_input",
                    "source_type": "free_text",
                },
            ],
            candidate_values=[
                "one_time",
                "recurring",
            ],
        )
    )

    assert candidate_ids(profile) == [
        "single_period_newsvendor",
        "eoq",
        "continuous_review_q_r",
    ]


def test_reports_unresolved_candidate_requirements() -> None:
    profile = create_empty_business_problem_profile()
    profile["recommendation"] = confirmed(
        "order_quantity"
    )
    profile["procurement_pattern"] = confirmed(
        "one_time"
    )

    catalog = build_core_selection_catalog(profile)

    assert len(catalog) == 1
    assert catalog[0]["core_id"] == (
        "single_period_newsvendor"
    )
    assert catalog[0]["unresolved_profile_fields"] == [
        "replenishment_during_period"
    ]
