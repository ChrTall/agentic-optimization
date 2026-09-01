"""
Zentrale Registry für verfügbare Modellfamilien und Modellvarianten.

Variantenspezifische Metadaten und Funktionen werden in den jeweiligen
Moduldateien definiert und hier lediglich registriert.
"""

from typing import Any

from agentic_optimization.models.eoq.basic_eoq import (
    VARIANT_DEFINITION as BASIC_EOQ,
)
from agentic_optimization.models.newsvendor.single_product_single_period import (
    VARIANT_DEFINITION as SINGLE_PRODUCT_SINGLE_PERIOD,
)
from agentic_optimization.models.reorder_point.q_r_policy import (
    VARIANT_DEFINITION as Q_R_POLICY,
)
from agentic_optimization.models.types import (
    ConstraintDefinition,
    ModelFamilyDefinition,
    ModelVariantDefinition,
)

MODEL_REGISTRY: dict[str, ModelFamilyDefinition] = {
    "eoq": {
        "family_id": "eoq",
        "family_name": "EOQ",
        "variants": {
            BASIC_EOQ["variant_id"]: BASIC_EOQ,
        },
    },
    "newsvendor": {
        "family_id": "newsvendor",
        "family_name": "Newsvendor",
        "variants": {
            SINGLE_PRODUCT_SINGLE_PERIOD["variant_id"]:
                SINGLE_PRODUCT_SINGLE_PERIOD,
        },
    },
    "reorder_point": {
        "family_id": "reorder_point",
        "family_name": "Reorder Point Policies",
        "variants": {
            Q_R_POLICY["variant_id"]: Q_R_POLICY,
        },
    },
}

def find_matching_model_variant(
    requirements: dict[str, Any],
) -> tuple[
    ModelFamilyDefinition,
    ModelVariantDefinition,
] | None:
    """
    Sucht die Modellvariante, deren selection_criteria zu den Requirements passen.

    Gibt None zurück, wenn keine Variante passt, und löst bei mehreren
    Treffern einen Fehler aus.
    """

    matches: list[
        tuple[
            ModelFamilyDefinition,
            ModelVariantDefinition,
        ]
    ] = []

    for family_definition in MODEL_REGISTRY.values():
        for variant_definition in family_definition[
            "variants"
        ].values():
            criteria = variant_definition[
                "selection_criteria"
            ]

            matches_variant = all(
                requirements.get(field_name)
                == expected_value
                for field_name, expected_value
                in criteria.items()
            )

            if matches_variant:
                matches.append(
                    (
                        family_definition,
                        variant_definition,
                    )
                )

    if len(matches) > 1:
        matching_variants = ", ".join(
            (
                f"{family['family_id']}/"
                f"{variant['variant_id']}"
            )
            for family, variant in matches
        )

        raise ValueError(
            "Die Modell-Registry enthält mehrere Varianten "
            "mit identischen Auswahlkriterien: "
            f"{matching_variants}."
        )

    if not matches:
        return None

    return matches[0]


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
