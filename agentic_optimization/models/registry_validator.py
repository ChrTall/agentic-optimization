"""
Integritätsprüfung der modularen Optimierungs-Registries.
"""

from collections import Counter

from agentic_optimization.models.component_registry import (
    CONSTRAINT_COMPONENTS,
    OBJECTIVE_COMPONENTS,
    ConstraintComponentDefinition,
    ObjectiveComponentDefinition,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
    RegisteredModelCore,
)
from agentic_optimization.models.types import (
    ConstraintParameterDefinition,
    ModelInputField,
)


FieldDefinition = (
    ModelInputField
    | ConstraintParameterDefinition
)

SUPPORTED_FIELD_TYPES = {
    "number",
    "integer",
    "forecast_quantiles",
}


def _find_duplicates(values: list[str]) -> set[str]:
    """Gibt mehrfach vorkommende Werte zurück."""

    counts = Counter(values)

    return {
        value
        for value, count in counts.items()
        if count > 1
    }


def _validate_field_definitions(
    owner: str,
    fields: list[FieldDefinition],
) -> list[str]:
    """Prüft Namen, Typen und Wertebereiche von Felddefinitionen."""

    errors: list[str] = []

    field_names = [
        field["name"]
        for field in fields
    ]

    for field_name in sorted(
        _find_duplicates(field_names)
    ):
        errors.append(
            f"Das Feld '{field_name}' ist in "
            f"{owner} mehrfach definiert."
        )

    for field in fields:
        field_name = field["name"]
        field_type = field["type"]

        if field_type not in SUPPORTED_FIELD_TYPES:
            errors.append(
                f"Das Feld '{field_name}' in {owner} "
                f"verwendet den unbekannten Feldtyp "
                f"'{field_type}'."
            )

        minimum = field.get("min_value")
        maximum = field.get("max_value")

        for bound_name, bound_value in (
            ("min_value", minimum),
            ("max_value", maximum),
        ):
            if (
                bound_value is not None
                and (
                    not isinstance(
                        bound_value,
                        (int, float),
                    )
                    or isinstance(bound_value, bool)
                )
            ):
                errors.append(
                    f"Das Feld '{field_name}' in {owner} "
                    f"besitzt einen ungültigen Wert für "
                    f"'{bound_name}'."
                )

        if (
            isinstance(minimum, (int, float))
            and not isinstance(minimum, bool)
            and isinstance(maximum, (int, float))
            and not isinstance(maximum, bool)
            and minimum > maximum
        ):
            errors.append(
                f"Beim Feld '{field_name}' in {owner} "
                "darf min_value nicht größer als "
                "max_value sein."
            )

    return errors


def validate_registry_integrity(
    registered_cores: (
        dict[str, RegisteredModelCore] | None
    ) = None,
    objective_registry: (
        dict[str, ObjectiveComponentDefinition] | None
    ) = None,
    constraint_registry: (
        dict[str, ConstraintComponentDefinition] | None
    ) = None,
) -> list[str]:
    """Prüft IDs, Core-Verweise und benötigte Modellgrößen."""

    cores = (
        CORE_REGISTRY
        if registered_cores is None
        else registered_cores
    )
    objectives = (
        OBJECTIVE_COMPONENTS
        if objective_registry is None
        else objective_registry
    )
    constraints = (
        CONSTRAINT_COMPONENTS
        if constraint_registry is None
        else constraint_registry
    )

    errors: list[str] = []

    for registry_key, registered_core in cores.items():
        definition = registered_core["definition"]
        core_id = definition["core_id"]

        minimum_products = definition["minimum_products"]

        if (
            not isinstance(minimum_products, int)
            or isinstance(minimum_products, bool)
            or minimum_products < 1
        ):
            errors.append(
                f"Das Grundmodell '{core_id}' muss mindestens "
                "ein Produkt verlangen."
            )

        if registry_key != core_id:
            errors.append(
                f"Der Core-Registry-Schlüssel '{registry_key}' "
                f"stimmt nicht mit der Core-ID '{core_id}' überein."
            )

        required_objective_component_ids = definition[
            "required_objective_component_ids"
        ]

        for duplicate_component_id in sorted(
            _find_duplicates(
                required_objective_component_ids
            )
        ):
            errors.append(
                f"Das Grundmodell '{core_id}' verlangt den "
                "Zielfunktionsbaustein "
                f"'{duplicate_component_id}' mehrfach."
            )

        for required_component_id in (
            required_objective_component_ids
        ):
            objective_definition = objectives.get(
                required_component_id
            )

            if objective_definition is None:
                errors.append(
                    f"Das Grundmodell '{core_id}' verlangt "
                    "den nicht registrierten "
                    "Zielfunktionsbaustein "
                    f"'{required_component_id}'."
                )
            elif (
                core_id
                not in objective_definition[
                    "compatible_core_ids"
                ]
            ):
                errors.append(
                    f"Der vom Grundmodell '{core_id}' "
                    "verlangte Zielfunktionsbaustein "
                    f"'{required_component_id}' ist nicht "
                    "mit diesem Grundmodell kompatibel."
                )

        solution_method = registered_core.get(
            "solution_method"
        )

        if solution_method not in {
            "linear_programming",
            "analytical",
            "numerical",
        }:
            errors.append(
                f"Das Grundmodell '{core_id}' besitzt keine "
                "gültige Lösungsmethode."
            )

        core_builder = registered_core.get("builder")

        if (
            solution_method == "linear_programming"
            and not callable(core_builder)
        ):
            errors.append(
                f"Für das lineare Grundmodell '{core_id}' "
                "ist kein aufrufbarer Builder registriert."
            )
        elif (
            core_builder is not None
            and not callable(core_builder)
        ):
            errors.append(
                f"Der Builder des Grundmodells '{core_id}' "
                "ist nicht aufrufbar."
            )

        if not callable(
            registered_core.get("forecast_validator")
        ):
            errors.append(
                f"Für das Grundmodell '{core_id}' "
                "ist kein aufrufbarer Forecast-Validator "
                "registriert."
            )

        if not callable(
            registered_core.get("semantic_validator")
        ):
            errors.append(
                f"Für das Grundmodell '{core_id}' "
                "ist kein aufrufbarer fachlicher "
                "Validator registriert."
            )

        if not callable(
            registered_core.get("executor")
        ):
            errors.append(
                f"Für das Grundmodell '{core_id}' "
                "ist kein aufrufbarer Executor registriert."
            )

        for quantity_key, quantity_definition in (
            definition["provided_quantities"].items()
        ):
            if (
                quantity_key
                != quantity_definition["quantity_id"]
            ):
                errors.append(
                    f"Der Modellgrößen-Schlüssel "
                    f"'{quantity_key}' im Grundmodell "
                    f"'{core_id}' stimmt nicht mit der "
                    f"Quantity-ID "
                    f"'{quantity_definition['quantity_id']}' "
                    "überein."
                )

        errors.extend(
            _validate_field_definitions(
                f"den Forecast-Feldern des Grundmodells "
                f"'{core_id}'",
                definition["required_forecast_fields"],
            )
        )

    component_registries = (
        (
            "Zielfunktionsbaustein",
            objectives,
        ),
        (
            "Constraint-Baustein",
            constraints,
        ),
    )

    for component_kind, component_registry in (
        component_registries
    ):
        for registry_key, definition in (
            component_registry.items()
        ):
            component_id = definition["component_id"]

            if registry_key != component_id:
                errors.append(
                    f"Der Registry-Schlüssel "
                    f"'{registry_key}' stimmt nicht mit "
                    f"der Baustein-ID '{component_id}' überein."
                )

            compatible_core_ids = definition[
                "compatible_core_ids"
            ]

            linear_core_ids = [
                compatible_core_id
                for compatible_core_id in compatible_core_ids
                if (
                    compatible_core_id in cores
                    and cores[compatible_core_id].get(
                        "solution_method"
                    )
                    == "linear_programming"
                )
            ]

            linear_builder = definition.get(
                "linear_builder"
            )

            if (
                linear_core_ids
                and not callable(linear_builder)
            ):
                errors.append(
                    f"Für den {component_kind} "
                    f"'{component_id}' ist kein "
                    "aufrufbarer linearer Builder registriert."
                )
            elif (
                linear_builder is not None
                and not callable(linear_builder)
            ):
                errors.append(
                    f"Der lineare Builder des "
                    f"{component_kind}s '{component_id}' "
                    "ist nicht aufrufbar."
                )

            for compatible_core_id in sorted(
                _find_duplicates(compatible_core_ids)
            ):
                errors.append(
                    f"Der {component_kind} '{component_id}' "
                    f"nennt das Grundmodell "
                    f"'{compatible_core_id}' mehrfach."
                )

            required_quantities = definition[
                "required_model_quantities"
            ]

            for quantity_id in sorted(
                _find_duplicates(required_quantities)
            ):
                errors.append(
                    f"Der {component_kind} '{component_id}' "
                    f"fordert die Modellgröße "
                    f"'{quantity_id}' mehrfach an."
                )

            errors.extend(
                _validate_field_definitions(
                    f"den Produktfeldern des "
                    f"{component_kind}s '{component_id}'",
                    definition["required_product_facts"],
                )
            )

            errors.extend(
                _validate_field_definitions(
                    f"den globalen Feldern des "
                    f"{component_kind}s '{component_id}'",
                    definition["required_global_facts"],
                )
            )

            if component_kind == "Constraint-Baustein":
                errors.extend(
                    _validate_field_definitions(
                        f"den Parametern des Constraint-Bausteins "
                        f"'{component_id}'",
                        definition["parameters"],
                    )
                )

            if not compatible_core_ids:
                errors.append(
                    f"Der {component_kind} '{component_id}' "
                    "ist keinem Grundmodell zugeordnet."
                )

            for core_id in compatible_core_ids:
                registered_core = cores.get(core_id)

                if registered_core is None:
                    errors.append(
                        f"Der {component_kind} "
                        f"'{component_id}' verweist auf das "
                        f"unbekannte Grundmodell '{core_id}'."
                    )
                    continue

                provided_quantities = set(
                    registered_core["definition"][
                        "provided_quantities"
                    ]
                )

                missing_quantities = (
                    set(
                        definition[
                            "required_model_quantities"
                        ]
                    )
                    - provided_quantities
                )

                for quantity_id in sorted(
                    missing_quantities
                ):
                    errors.append(
                        f"Der {component_kind} "
                        f"'{component_id}' benötigt die vom "
                        f"Grundmodell '{core_id}' nicht "
                        f"bereitgestellte Modellgröße "
                        f"'{quantity_id}'."
                    )

    return errors