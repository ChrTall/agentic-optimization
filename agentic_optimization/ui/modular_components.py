"""Explizite Ergänzungen eines bestehenden Modellentwurfs in Schritt 3."""

from copy import deepcopy
from typing import TypedDict

from agentic_optimization.models.component_catalog import build_component_catalog
from agentic_optimization.models.core_registry import CORE_REGISTRY
from agentic_optimization.models.specification import ModelSpecification
from agentic_optimization.models.validator import validate_model_specification
from agentic_optimization.state import invalidate_optimization_state
from agentic_optimization.ui.modular_input_builder import build_modular_input_layout


class ComponentOption(TypedDict):
    option_id: str
    component_id: str
    kind: str
    product_id: str | None
    label: str
    help: str


def input_address(specification, item):
    """Stabile Feldadresse, auch wenn ein Constraint davor entfernt wird."""
    constraint = (specification["constraints"][item["constraint_index"]]
                  if item["source"] == "constraint_parameter" else {})
    return (item["source"], item.get("product_id") or constraint.get("product_id"),
            constraint.get("component_id"), item["field"]["name"])


def component_options(
    specification: ModelSpecification, *, include_active: bool = False,
) -> list[ComponentOption]:
    """Kompatible Zusatzbausteine, optional einschließlich aktiver Auswahl."""
    catalog = build_component_catalog(specification["core_id"])
    options = []
    for component in catalog["objective_components"]:
        if component["component_id"] in specification["objective"]["component_ids"] and (
            not include_active or component["component_id"] not in removable_objectives(specification)
        ):
            continue
        options.append({
            "option_id": f"objective:{component['component_id']}",
            "component_id": component["component_id"],
            "kind": "objective", "product_id": None,
            "label": component["name"],
            "help": " ".join(field.get("help", "") for field in
                             component["required_product_facts"]),
        })
    for component in catalog["constraint_components"]:
        targets = ([p["product_id"] for p in specification["products"]]
                   if component["scope"] == "per_product" else [None])
        for product_id in targets:
            if not include_active and any(c["component_id"] == component["component_id"]
                   and c.get("product_id") == product_id
                   for c in specification["constraints"]):
                continue
            options.append({
                "option_id": f"constraint:{component['component_id']}:{product_id}",
                "component_id": component["component_id"],
                "kind": "constraint", "product_id": product_id,
                "label": component["name"] + (
                    f" – {product_id}" if product_id is not None else ""),
                "help": " ".join(field.get("help", "")
                                 for field in component["parameters"]),
            })
    return options


def add_component(specification: ModelSpecification, option_id: str) -> ModelSpecification:
    option = next((o for o in component_options(specification)
                   if o["option_id"] == option_id), None)
    if option is None:
        raise ValueError("Dieser Baustein ist nicht verfügbar oder bereits aktiv.")
    result = deepcopy(specification)
    if option["kind"] == "objective":
        result["objective"]["component_ids"].append(option["component_id"])
    else:
        constraint = {"component_id": option["component_id"], "parameters": {}}
        if option["product_id"] is not None:
            constraint["product_id"] = option["product_id"]
        result["constraints"].append(constraint)
    return result


def component_is_active(specification: ModelSpecification, option: ComponentOption) -> bool:
    if option["kind"] == "objective":
        return option["component_id"] in specification["objective"]["component_ids"]
    return any(c["component_id"] == option["component_id"]
               and c.get("product_id") == option["product_id"]
               for c in specification["constraints"])


def remove_component(specification: ModelSpecification, option: ComponentOption) -> ModelSpecification:
    """Entfernt eine Zusatzangabe; bereits erfasste Produktfakten bleiben erhalten."""
    if option not in component_options(specification, include_active=True):
        raise ValueError("Dieser Baustein kann nicht entfernt werden.")
    result = deepcopy(specification)
    if option["kind"] == "objective":
        result["objective"]["component_ids"].remove(option["component_id"])
    else:
        result["constraints"] = [c for c in result["constraints"]
                                 if not (c["component_id"] == option["component_id"]
                                         and c.get("product_id") == option["product_id"])]
    return result


def removable_objectives(specification: ModelSpecification) -> list[str]:
    core = CORE_REGISTRY[specification["core_id"]]["definition"]
    required = set(core["required_objective_component_ids"])
    for components in core.get("required_objective_components_by_goal", {}).values():
        required.update(components)
    return [c for c in specification["objective"]["component_ids"] if c not in required]


def editable_chat_inputs(specification: ModelSpecification) -> list[dict]:
    """Aktive Felder und eindeutig adressierte Felder möglicher Ergänzungen."""
    def flatten(spec):
        layout = build_modular_input_layout(spec, include_completed=True)
        inputs = [*[i for g in layout["product_groups"]
                  for i in [*g["forecast_inputs"], *g["fact_inputs"]]],
                *layout["global_inputs"],
                *[i for g in layout["constraint_groups"] for i in g["inputs"]]]
        return [{**item, "product_id": spec["constraints"][item["constraint_index"]].get("product_id")}
                if item["source"] == "constraint_parameter" else item
                for item in inputs]

    def address(item):
        return (item["source"], item.get("product_id"),
                item.get("constraint_index"), item["field"]["name"])

    active = flatten(specification)
    result = [{"input": item, "option_id": None} for item in active]
    addresses = {address(item) for item in active}
    for option in component_options(specification):
        for item in flatten(add_component(specification, option["option_id"])):
            if address(item) not in addresses:
                result.append({"input": item, "option_id": option["option_id"]})
    return result


def update_model_draft(graph_state: dict, specification: ModelSpecification) -> dict:
    """Synchronisiert Entwurf und Auswahl, ohne Schritt 3 zu verlassen."""
    result = invalidate_optimization_state(graph_state)
    model = result["modular_model"]
    proposal = deepcopy(model["proposal"])
    evidence = {"reason": "Vom Nutzer in Schritt 3 ausgewählt.",
                "sources": ["form_3"]}
    old_objectives = {c["component_id"]: c for c in proposal["objective_components"]}
    proposal["objective_components"] = [
        old_objectives.get(cid, {"component_id": cid, **evidence})
        for cid in specification["objective"]["component_ids"]]
    old_constraints = {(c["component_id"], c.get("product_id")): c
                       for c in proposal["constraints"]}
    proposal["constraints"] = [
        {**old_constraints.get((c["component_id"], c.get("product_id")), evidence),
         **deepcopy(c)} for c in specification["constraints"]]
    model.update(specification=specification, proposal=proposal,
                 products=deepcopy(specification["products"]),
                 validation=validate_model_specification(specification),
                 pending_input_submissions=[], form_3_review_confirmed=False)
    return result
