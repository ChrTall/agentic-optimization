import pytest

from agentic_optimization.models.problem_profile import (
    create_empty_business_problem_profile,
    create_profile_field,
    merge_profile_fields,
)


def test_creates_empty_business_problem_profile() -> None:
    profile = create_empty_business_problem_profile()

    assert profile["forecasts"] == {}

    structural_fields = {
        field_name: field
        for field_name, field in profile.items()
        if field_name != "forecasts"
    }

    assert structural_fields
    assert all(
        field["value"] is None
        and field["status"] == "missing"
        and field["evidence"] == []
        for field in structural_fields.values()
    )


def test_empty_profile_fields_do_not_share_evidence() -> None:
    profile = create_empty_business_problem_profile()

    profile["recommendation"]["evidence"].append(
        {
            "source_id": "form_1",
            "source_type": "user_selection",
        }
    )

    assert profile["procurement_pattern"]["evidence"] == []


def test_creates_inferred_field_with_evidence() -> None:
    field = create_profile_field(
        status="inferred",
        value="recurring",
        evidence=[
            {
                "source_id": "raw_input",
                "source_type": "free_text",
                "excerpt": "Wir bestellen regelmäßig nach.",
            }
        ],
    )

    assert field == {
        "value": "recurring",
        "status": "inferred",
        "evidence": [
            {
                "source_id": "raw_input",
                "source_type": "free_text",
                "excerpt": "Wir bestellen regelmäßig nach.",
            }
        ],
    }


def test_represents_user_unknown_without_domain_value() -> None:
    field = create_profile_field(
        status="user_unknown",
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )

    assert field["value"] is None
    assert field["status"] == "user_unknown"


def test_represents_conflict_with_candidates() -> None:
    field = create_profile_field(
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
        candidate_values=["one_time", "recurring"],
    )

    assert field["value"] is None
    assert field["candidate_values"] == [
        "one_time",
        "recurring",
    ]


@pytest.mark.parametrize(
    ("status", "value", "evidence", "candidate_values"),
    [
        ("missing", "recurring", [], None),
        ("user_unknown", "recurring", [], None),
        ("inferred", None, [], None),
        ("confirmed", None, [], None),
        ("user_unknown", None, [], None),
        (
            "ambiguous",
            None,
            [
                {
                    "source_id": "raw_input",
                    "source_type": "free_text",
                }
            ],
            ["one_time"],
        ),
    ],
)
def test_rejects_inconsistent_profile_field(
    status,
    value,
    evidence,
    candidate_values,
) -> None:
    with pytest.raises(ValueError):
        create_profile_field(
            status=status,
            value=value,
            evidence=evidence,
            candidate_values=candidate_values,
        )


def test_inferred_value_resolves_user_unknown() -> None:
    current_field = create_profile_field(
        status="user_unknown",
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )
    inferred_field = create_profile_field(
        status="inferred",
        value="recurring",
        evidence=[
            {
                "source_id": "raw_input",
                "source_type": "free_text",
            }
        ],
    )

    merged_field = merge_profile_fields(
        current_field,
        inferred_field,
    )

    assert merged_field["value"] == "recurring"
    assert merged_field["status"] == "inferred"
    assert {
        evidence["source_id"]
        for evidence in merged_field["evidence"]
    } == {
        "form_1",
        "raw_input",
    }


def test_confirmation_strengthens_equal_inferred_value() -> None:
    inferred_field = create_profile_field(
        status="inferred",
        value="recurring",
        evidence=[
            {
                "source_id": "raw_input",
                "source_type": "free_text",
            }
        ],
    )
    confirmed_field = create_profile_field(
        status="confirmed",
        value="recurring",
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )

    merged_field = merge_profile_fields(
        inferred_field,
        confirmed_field,
    )

    assert merged_field["value"] == "recurring"
    assert merged_field["status"] == "confirmed"
    assert len(merged_field["evidence"]) == 2


def test_different_values_create_conflict() -> None:
    selected_field = create_profile_field(
        status="confirmed",
        value="one_time",
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )
    inferred_field = create_profile_field(
        status="inferred",
        value="recurring",
        evidence=[
            {
                "source_id": "raw_input",
                "source_type": "free_text",
            }
        ],
    )

    merged_field = merge_profile_fields(
        selected_field,
        inferred_field,
    )

    assert merged_field["value"] is None
    assert merged_field["status"] == "conflict"
    assert merged_field["candidate_values"] == [
        "one_time",
        "recurring",
    ]


def test_merging_profile_fields_does_not_mutate_inputs() -> None:
    current_field = create_profile_field(
        status="inferred",
        value="recurring",
        evidence=[
            {
                "source_id": "raw_input",
                "source_type": "free_text",
            }
        ],
    )
    incoming_field = create_profile_field(
        status="confirmed",
        value="recurring",
        evidence=[
            {
                "source_id": "form_1",
                "source_type": "user_selection",
            }
        ],
    )

    merge_profile_fields(current_field, incoming_field)

    assert current_field["status"] == "inferred"
    assert len(current_field["evidence"]) == 1
    assert incoming_field["status"] == "confirmed"
    assert len(incoming_field["evidence"]) == 1
