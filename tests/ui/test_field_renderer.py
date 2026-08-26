import pytest

from agentic_optimization.ui.field_renderer import (
    parse_input_value,
    render_input_field,
)


def create_number_field() -> dict:
    return {
        "name": "selling_price",
        "label": "Verkaufspreis",
        "type": "number",
        "required": True,
        "help": None,
        "min_value": 0.0,
        "step": 0.5,
        "unit": "€",
    }


def create_number_list_field() -> dict:
    return {
        "name": "probabilities",
        "label": "Wahrscheinlichkeiten",
        "type": "number_list",
        "required": True,
        "help": None,
        "min_value": 0.0,
        "max_value": 1.0,
    }


def create_integer_field() -> dict:
    return {
        "name": "reorder_point",
        "label": "Meldebestand",
        "type": "integer",
        "required": True,
        "help": None,
        "min_value": 0,
        "step": 1,
        "unit": "Einheiten",
    }


def test_parses_number() -> None:
    result = parse_input_value(
        create_number_field(),
        12.5,
    )

    assert result == 12.5


def test_returns_none_for_empty_number() -> None:
    result = parse_input_value(
        create_number_field(),
        None,
    )

    assert result is None


def test_parses_integer_units() -> None:
    result = parse_input_value(
        create_integer_field(),
        42.0,
    )

    assert result == 42
    assert isinstance(result, int)


def test_rejects_fractional_integer_units() -> None:
    with pytest.raises(
        ValueError,
        match="ganzen Einheiten",
    ):
        parse_input_value(
            create_integer_field(),
            42.5,
        )


def test_rejects_number_below_minimum() -> None:
    with pytest.raises(
        ValueError,
        match="kleiner als",
    ):
        parse_input_value(
            create_number_field(),
            -1,
        )


def test_parses_comma_separated_number_list() -> None:
    result = parse_input_value(
        create_number_list_field(),
        "0.2, 0.5, 0.3",
    )

    assert result == [0.2, 0.5, 0.3]


def test_parses_existing_number_list() -> None:
    result = parse_input_value(
        create_number_list_field(),
        [0.2, 0.5, 0.3],
    )

    assert result == [0.2, 0.5, 0.3]


def test_returns_empty_list_for_empty_number_list() -> None:
    result = parse_input_value(
        create_number_list_field(),
        "",
    )

    assert result == []


def test_rejects_non_numeric_list_item() -> None:
    with pytest.raises(
        ValueError,
        match="ausschließlich Zahlen",
    ):
        parse_input_value(
            create_number_list_field(),
            "0.2, falsch, 0.8",
        )


def test_rejects_list_item_above_maximum() -> None:
    with pytest.raises(
        ValueError,
        match="größer als",
    ):
        parse_input_value(
            create_number_list_field(),
            "0.2, 1.2",
        )


def test_rejects_unsupported_field_type() -> None:
    field = {
        "name": "unknown",
        "label": "Unbekannt",
        "type": "date",
        "required": True,
        "help": None,
    }

    with pytest.raises(
        ValueError,
        match="wird noch nicht unterstützt",
    ):
        parse_input_value(
            field,
            "2026-01-01",
        )


def test_rejects_invalid_number_data_type() -> None:
    with pytest.raises(
        TypeError,
        match="muss eine Zahl sein",
    ):
        parse_input_value(
            create_number_field(),
            ["12.5"],
        )


def test_rejects_invalid_number_list_data_type() -> None:
    with pytest.raises(
        TypeError,
        match="kommagetrennte Zahlenliste",
    ):
        parse_input_value(
            create_number_list_field(),
            {
                "first": 0.2,
                "second": 0.8,
            },
        )


def test_rejects_invalid_item_type_in_number_list() -> None:
    with pytest.raises(
        TypeError,
        match="ungültigem Datentyp",
    ):
        parse_input_value(
            create_number_list_field(),
            [0.2, True, 0.8],
        )


def test_disables_number_input_and_preserves_initial_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_arguments: dict = {}
    fake_session_state: dict = {}

    def fake_number_input(
        **arguments,
    ) -> None:
        received_arguments.update(arguments)
        return fake_session_state.get(
            arguments["key"]
        )

    monkeypatch.setattr(
        (
            "agentic_optimization.ui."
            "field_renderer.st.number_input"
        ),
        fake_number_input,
    )

    monkeypatch.setattr(
        (
            "agentic_optimization.ui."
            "field_renderer.st.session_state"
        ),
        fake_session_state,
    )

    result = render_input_field(
        create_number_field(),
        7400.0,
        key_prefix="test_constraint",
        disabled=True,
    )

    widget_key = (
        "test_constraint_selling_price"
    )

    assert received_arguments["disabled"] is True
    assert received_arguments["value"] is None
    assert fake_session_state[widget_key] == 7400.0
    assert result == 7400.0


def test_does_not_restore_cleared_number_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_arguments: dict = {}

    widget_key = (
        "test_constraint_selling_price"
    )

    fake_session_state = {
        widget_key: None,
    }

    def fake_number_input(
        **arguments,
    ) -> None:
        received_arguments.update(arguments)
        return fake_session_state.get(
            arguments["key"]
        )

    monkeypatch.setattr(
        (
            "agentic_optimization.ui."
            "field_renderer.st.number_input"
        ),
        fake_number_input,
    )

    monkeypatch.setattr(
        (
            "agentic_optimization.ui."
            "field_renderer.st.session_state"
        ),
        fake_session_state,
    )

    result = render_input_field(
        create_number_field(),
        40.0,
        key_prefix="test_constraint",
    )

    assert received_arguments["value"] is None
    assert fake_session_state[widget_key] is None
    assert result is None
