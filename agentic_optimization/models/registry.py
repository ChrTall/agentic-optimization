from agentic_optimization.models.newsvendor.single_product_single_period import (
    VARIANT_DEFINITION as SINGLE_PRODUCT_SINGLE_PERIOD,
)
from agentic_optimization.models.types import (
    ConstraintDefinition,
    ModelFamilyDefinition,
    ModelVariantDefinition,
)

MODEL_REGISTRY: dict[str, ModelFamilyDefinition] = {
    "newsvendor": {
        "family_id": "newsvendor",
        "family_name": "Newsvendor",
        "variants": {
            SINGLE_PRODUCT_SINGLE_PERIOD["variant_id"]:
                SINGLE_PRODUCT_SINGLE_PERIOD,
        },
    },
}


def get_model_family_definition(
    family_id: str | None,
) -> ModelFamilyDefinition | None:
    """Gibt die Definition einer Modellfamilie zurück."""

    if not family_id:
        return None

    return MODEL_REGISTRY.get(family_id)


def get_model_variant_definition(
    family_id: str | None,
    variant_id: str | None,
) -> ModelVariantDefinition | None:
    """Gibt die Definition einer konkreten Modellvariante zurück."""

    if not family_id or not variant_id:
        return None

    family_definition = get_model_family_definition(family_id)

    if family_definition is None:
        return None

    return family_definition["variants"].get(variant_id)


def get_constraint_definition(
    family_id: str | None,
    variant_id: str | None,
    constraint_type: str | None,
) -> ConstraintDefinition | None:
    """Gibt einen unterstützten Constraint einer Modellvariante zurück."""

    if not constraint_type:
        return None

    variant_definition = get_model_variant_definition(
        family_id=family_id,
        variant_id=variant_id,
    )

    if variant_definition is None:
        return None

    return variant_definition["supported_constraints"].get(
        constraint_type
    )