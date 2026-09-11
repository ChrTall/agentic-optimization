from copy import deepcopy

from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
    OBJECTIVE_COMPONENTS,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
)
from agentic_optimization.models.registry_validator import (
    validate_registry_integrity,
)


def test_current_registries_are_valid() -> None:
    errors = validate_registry_integrity()

    assert errors == []


def test_rejects_unknown_compatible_core() -> None:
    objectives = deepcopy(OBJECTIVE_COMPONENTS)

    objectives["sales_revenue"][
        "compatible_core_ids"
    ] = ["unknown_core"]

    errors = validate_registry_integrity(
        objective_registry=objectives
    )

    assert any(
        "sales_revenue" in error
        and "unknown_core" in error
        for error in errors
    )


def test_rejects_missing_model_quantity() -> None:
    objectives = deepcopy(OBJECTIVE_COMPONENTS)

    objectives["sales_revenue"][
        "required_model_quantities"
    ] = ["unknown_quantity"]

    errors = validate_registry_integrity(
        objective_registry=objectives
    )

    assert any(
        "sales_revenue" in error
        and "unknown_quantity" in error
        for error in errors
    )


def test_rejects_mismatched_core_registry_key() -> None:
    cores = deepcopy(CORE_REGISTRY)

    registered_core = cores.pop(
        "single_period_newsvendor"
    )
    cores["wrong_registry_key"] = registered_core

    errors = validate_registry_integrity(
        registered_cores=cores
    )

    assert any(
        "wrong_registry_key" in error
        and "single_period_newsvendor" in error
        for error in errors
    )


def test_rejects_duplicate_compatible_core() -> None:
    objectives = deepcopy(OBJECTIVE_COMPONENTS)

    objectives["sales_revenue"][
        "compatible_core_ids"
    ] = [
        "single_period_newsvendor",
        "single_period_newsvendor",
    ]

    errors = validate_registry_integrity(
        objective_registry=objectives
    )

    assert any(
        "sales_revenue" in error
        and "mehrfach" in error
        for error in errors
    )


def test_rejects_duplicate_required_quantity() -> None:
    objectives = deepcopy(OBJECTIVE_COMPONENTS)

    objectives["sales_revenue"][
        "required_model_quantities"
    ] = [
        "sold_quantity",
        "sold_quantity",
    ]

    errors = validate_registry_integrity(
        objective_registry=objectives
    )

    assert any(
        "sales_revenue" in error
        and "sold_quantity" in error
        and "mehrfach" in error
        for error in errors
    )


def test_rejects_invalid_field_bounds() -> None:
    constraints = deepcopy(CONSTRAINT_COMPONENTS)

    parameter = constraints[
        "minimum_order_quantity"
    ]["parameters"][0]

    parameter["min_value"] = 10.0
    parameter["max_value"] = 5.0

    errors = validate_registry_integrity(
        constraint_registry=constraints
    )

    assert any(
        "quantity" in error
        and "min_value" in error
        and "max_value" in error
        for error in errors
    )


def test_rejects_invalid_minimum_product_count() -> None:
    cores = deepcopy(CORE_REGISTRY)

    cores["single_period_newsvendor"][
        "definition"
    ]["minimum_products"] = 0

    errors = validate_registry_integrity(
        registered_cores=cores
    )

    assert any(
        "single_period_newsvendor" in error
        and "mindestens ein Produkt" in error
        for error in errors
    )


def test_rejects_missing_semantic_validator() -> None:
    cores = deepcopy(CORE_REGISTRY)

    cores["single_period_newsvendor"][
        "semantic_validator"
    ] = None

    errors = validate_registry_integrity(
        registered_cores=cores
    )

    assert any(
        "single_period_newsvendor" in error
        and "fachlicher Validator" in error
        for error in errors
    )


def test_rejects_missing_executor() -> None:
    cores = deepcopy(CORE_REGISTRY)
    cores["single_period_newsvendor"]["executor"] = None

    errors = validate_registry_integrity(
        registered_cores=cores
    )

    assert any(
        "single_period_newsvendor" in error
        and "Executor" in error
        for error in errors
    )


def test_rejects_invalid_solution_method() -> None:
    cores = deepcopy(CORE_REGISTRY)
    cores["single_period_newsvendor"][
        "solution_method"
    ] = "unknown_method"

    errors = validate_registry_integrity(
        registered_cores=cores
    )

    assert any(
        "single_period_newsvendor" in error
        and "Lösungsmethode" in error
        for error in errors
    )


def test_rejects_missing_linear_component_builder() -> None:
    objectives = deepcopy(OBJECTIVE_COMPONENTS)
    objectives["sales_revenue"]["linear_builder"] = None

    errors = validate_registry_integrity(
        objective_registry=objectives
    )

    assert any(
        "sales_revenue" in error
        and "linearer Builder" in error
        for error in errors
    )


def test_analytical_core_does_not_require_builder() -> None:
    cores = deepcopy(CORE_REGISTRY)
    registered_core = cores["single_period_newsvendor"]

    registered_core["solution_method"] = "analytical"
    registered_core.pop("builder")

    errors = validate_registry_integrity(
        registered_cores=cores
    )

    assert errors == []