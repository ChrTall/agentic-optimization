"""
Prüft und übernimmt extrahierte Fakten in Produktdaten.
"""

from copy import deepcopy
from typing import Any
from agentic_optimization.models.product_names import clean_product_name

from agentic_optimization.models.component_catalog import (
    build_component_catalog,
)
from agentic_optimization.models.specification import (
    ProductSpecification,
)


def _has_value(value: Any) -> bool:
    return value is not None and value != "" and value != []


def _collect_allowed_fields(
    core_id: str,
) -> tuple[set[str], set[str], set[str]]:
    """Ermittelt erlaubte Forecast-, Produkt- und Globalfelder."""

    catalog = build_component_catalog(core_id)

    forecast_fields = {
        field["name"]
        for field in catalog["core"][
            "required_forecast_fields"
        ]
    }
    product_fields = {
        field["name"]
        for field in catalog["core"][
            "required_product_facts"
        ]
    }
    global_fields = {
        field["name"]
        for field in catalog["core"][
            "required_global_facts"
        ]
    }

    for component in catalog["objective_components"]:
        product_fields.update(
            field["name"]
            for field in component[
                "required_product_facts"
            ]
        )
        global_fields.update(
            field["name"]
            for field in component[
                "required_global_facts"
            ]
        )

    for constraint in catalog["constraint_components"]:
        product_fields.update(
            field["name"]
            for field in constraint[
                "required_product_facts"
            ]
        )
        global_fields.update(
            field["name"]
            for field in constraint[
                "required_global_facts"
            ]
        )

    return (
        forecast_fields,
        product_fields,
        global_fields,
    )


def merge_extracted_facts(
    *,
    core_id: str,
    products: list[ProductSpecification],
    global_facts: dict[str, Any],
    extraction: dict[str, Any],
    evidence_texts: list[str] | None = None,
) -> tuple[
    list[ProductSpecification],
    dict[str, Any],
]:
    """
    Prüft extrahierte Felder und ergänzt nur bisher fehlende Werte.

    Bereits vorhandene Werte besitzen Vorrang vor LLM-Ausgaben.
    """

    (
        allowed_forecast_fields,
        allowed_product_fields,
        allowed_global_fields,
    ) = _collect_allowed_fields(core_id)

    extracted_products = extraction.get(
        "products",
        [],
    )
    extracted_global_facts = extraction.get(
        "global_facts",
        {},
    )

    if not isinstance(extracted_products, list):
        raise TypeError(
            "'products' der Faktenextraktion muss "
            "eine Liste sein."
        )

    if not isinstance(extracted_global_facts, dict):
        raise TypeError(
            "'global_facts' der Faktenextraktion muss "
            "ein Objekt sein."
        )

    updated_products = deepcopy(products)
    updated_global_facts = deepcopy(global_facts)

    products_by_id = {
        product["product_id"]: product
        for product in updated_products
    }
    # Namensmetadaten sind unabhängig von Zahlen: Das LLM darf deshalb
    # products=[] und trotzdem belegte Produktnamen zurückgeben.
    name_candidates = extraction.get("product_names", [])
    if isinstance(name_candidates, list):
        candidate_counts = {}
        for candidate in name_candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("product_id"), str):
                key = candidate["product_id"]
                candidate_counts[key] = candidate_counts.get(key, 0) + 1
        extracted_products = deepcopy(extracted_products)
        for candidate in name_candidates:
            if not isinstance(candidate, dict):
                continue
            key = candidate.get("product_id")
            if not isinstance(key, str) or key not in products_by_id:
                continue
            if candidate_counts.get(key) != 1 or candidate.get("status") != "identified":
                for item in extracted_products:
                    if isinstance(item, dict) and item.get("product_id") == key:
                        item.pop("display_name", None)
                        item.pop("display_name_evidence", None)
                continue
            # Bei widersprüchlichen Angaben im selben LLM-Ergebnis keinen
            # der Namen nach willkürlicher Reihenfolge übernehmen.
            existing = [p for p in extracted_products if isinstance(p, dict) and p.get("product_id") == key]
            if existing:
                if existing[0].get("display_name") not in (None, candidate.get("display_name")):
                    existing[0].pop("display_name", None)
                    existing[0].pop("display_name_evidence", None)
                    continue
                existing[0].update({field: candidate.get(field)
                                    for field in ("display_name", "display_name_evidence")})
            else:
                extracted_products.append({"product_id": key,
                                           "display_name": candidate.get("display_name"),
                                           "display_name_evidence": candidate.get("display_name_evidence")})
    seen_product_ids: set[str] = set()

    for extracted_product in extracted_products:
        if not isinstance(extracted_product, dict):
            raise TypeError(
                "Jedes extrahierte Produkt muss ein "
                "Objekt sein."
            )

        product_id = extracted_product.get(
            "product_id"
        )

        if not isinstance(product_id, str):
            raise TypeError(
                "Jedes extrahierte Produkt benötigt "
                "eine product_id."
            )

        if product_id in seen_product_ids:
            raise ValueError(
                f"Das Produkt '{product_id}' wurde in der "
                "Faktenextraktion mehrfach angegeben."
            )

        seen_product_ids.add(product_id)

        product = products_by_id.get(product_id)

        if product is None:
            raise ValueError(
                f"Das extrahierte Produkt '{product_id}' "
                "ist nicht im Modellentwurf vorhanden."
            )

        name = clean_product_name(extracted_product.get("display_name"))
        quote = extracted_product.get("display_name_evidence")
        # Ein belegter Wortlaut ist erforderlich. Bei mehreren Produkten
        # muss dieselbe Textstelle zusätzlich die feste ID zuordnen.
        if (not product.get("display_name") and name and isinstance(quote, str)
                and quote.strip() and name.casefold() in quote.casefold()
                and any(quote.casefold() in text.casefold() for text in (evidence_texts or []))
                and (len(products) == 1 or (
                    product_id.casefold() in quote.casefold()
                    and not any(other.casefold() in quote.casefold()
                                for other in products_by_id if other != product_id)))):
            product["display_name"] = name

        forecast = extracted_product.get(
            "forecast",
            {},
        )
        facts = extracted_product.get(
            "facts",
            {},
        )

        if not isinstance(forecast, dict):
            raise TypeError(
                f"Der Forecast von '{product_id}' muss "
                "ein Objekt sein."
            )

        if not isinstance(facts, dict):
            raise TypeError(
                f"Die Fakten von '{product_id}' müssen "
                "ein Objekt sein."
            )

        unknown_forecast_fields = (
            set(forecast)
            - allowed_forecast_fields
        )
        unknown_product_fields = (
            set(facts)
            - allowed_product_fields
        )

        if unknown_forecast_fields:
            raise ValueError(
                "Nicht erlaubte Forecast-Felder: "
                + ", ".join(
                    sorted(unknown_forecast_fields)
                )
            )

        if unknown_product_fields:
            raise ValueError(
                "Nicht erlaubte Produktfelder: "
                + ", ".join(
                    sorted(unknown_product_fields)
                )
            )

        target_forecast = product.setdefault(
            "forecast",
            {},
        )
        target_facts = product.setdefault(
            "facts",
            {},
        )

        for field_name, value in forecast.items():
            if not _has_value(
                target_forecast.get(field_name)
            ):
                target_forecast[field_name] = value

        for field_name, value in facts.items():
            if not _has_value(
                target_facts.get(field_name)
            ):
                target_facts[field_name] = value

    unknown_global_fields = (
        set(extracted_global_facts)
        - allowed_global_fields
    )

    if unknown_global_fields:
        raise ValueError(
            "Nicht erlaubte globale Felder: "
            + ", ".join(
                sorted(unknown_global_fields)
            )
        )

    for field_name, value in extracted_global_facts.items():
        if not _has_value(
            updated_global_facts.get(field_name)
        ):
            updated_global_facts[field_name] = value

    return updated_products, updated_global_facts
