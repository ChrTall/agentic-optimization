"""Anzeigenamen sind Metadaten; alle Modellzuordnungen verwenden weiterhin IDs."""

import re
from collections import Counter


def clean_product_name(value):
    if not isinstance(value, str):
        return None
    name = " ".join(value.split())
    return name if name and len(name) <= 120 else None


def fallback_product_name(product_id):
    match = re.fullmatch(r"(?:product|produkt)[_ -]?(\d+)", product_id, re.IGNORECASE)
    return f"Produkt {int(match[1])}" if match else product_id


def product_names(products):
    names = {p["product_id"]: clean_product_name(p.get("display_name"))
             or fallback_product_name(p["product_id"]) for p in products}
    counts = Counter(name.casefold() for name in names.values())
    return {pid: name if counts[name.casefold()] == 1 else f"{name} (Produkt {index})"
            for index, (pid, name) in enumerate(names.items(), 1)}


def product_name(specification, product_id):
    return product_names(specification.get("products", [])).get(
        product_id, fallback_product_name(product_id)) if product_id else "Gesamtmodell"
