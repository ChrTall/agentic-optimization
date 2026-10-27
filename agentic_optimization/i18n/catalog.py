"""Sammelt alle übersetzbaren deutschen Texte.

Wird vom Test genutzt, der prüft, dass jeder Text eine englische
Übersetzung mit denselben Platzhaltern hat.
"""

import ast
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


def _static_keys() -> set[str]:
    """Erste Argumente aller Aufrufe ``t("…")`` im Code."""

    keys: set[str] = set()
    files = [PROJECT_ROOT / "app.py", *PACKAGE_ROOT.rglob("*.py")]

    for path in files:
        if "i18n" in path.parts:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "t"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)

    return keys


_DISPLAY_KEYS = {
    "label", "help", "unit", "prompt", "description",
    "decision_description",
}


def _collect_from(value: Any, keys: set[str], *, parent_key: str | None = None) -> None:
    if isinstance(value, dict):
        has_identity = any(
            key in value for key in ("component_id", "quantity_id", "core_id")
        )
        for key, item in value.items():
            if key in _DISPLAY_KEYS and isinstance(item, str):
                keys.add(item)
            elif key == "name" and has_identity and isinstance(item, str):
                keys.add(item)
            elif key == "assumptions" and isinstance(item, list):
                keys.update(entry for entry in item if isinstance(entry, str))
            else:
                _collect_from(item, keys, parent_key=key)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_from(item, keys, parent_key=parent_key)


def _constant_keys() -> set[str]:
    """Feste Texte aus Registries und Konstanten, die bei der Anzeige
    übersetzt werden."""

    from agentic_optimization.agents import business_intake_nodes
    from agentic_optimization.llm import errors
    from agentic_optimization.models.component_registry import (
        CONSTRAINT_COMPONENTS,
        OBJECTIVE_COMPONENTS,
    )
    from agentic_optimization.models.core_applicability import CONDITION_LABELS
    from agentic_optimization.models.core_registry import CORE_REGISTRY
    from agentic_optimization.ui import chat_assistant_builder as chat
    from agentic_optimization.ui import chat_assistant_panel as panel
    from agentic_optimization.ui import modular_result_builder as results
    from agentic_optimization.ui import requirements_panel
    from agentic_optimization.ui.business_problem_form import (
        APPLICABILITY_QUESTION_PROMPTS,
        BUSINESS_QUESTIONS,
    )

    keys: set[str] = set()

    for registered in CORE_REGISTRY.values():
        _collect_from(registered["definition"], keys)
        keys.add(registered["definition"]["name"])

    _collect_from(list(OBJECTIVE_COMPONENTS.values()), keys)
    _collect_from(list(CONSTRAINT_COMPONENTS.values()), keys)
    _collect_from(list(BUSINESS_QUESTIONS.values()), keys)

    keys.update(CONDITION_LABELS.values())
    keys.update(APPLICABILITY_QUESTION_PROMPTS.values())
    keys.update(results.METRIC_LABELS.values())
    keys.update(results.STATUS_LABELS.values())
    keys.update(results.SOLUTION_METHOD_LABELS.values())

    for labels in results.OBJECTIVE_LABELS_BY_CORE.values():
        keys.update(labels.values())

    keys.update({
        chat.GREETING,
        chat.SUPPORTED_CASES_TEXT,
        chat.UNSUPPORTED_MESSAGE,
        chat.MISSING_DECISIVE_INFO_MESSAGE,
    })
    keys.update(label for _, label in chat.STEP_PROGRESS.values())
    keys.update(title for _, title in requirements_panel.REQUIREMENT_FIELDS)
    keys.update(option["label"] for option in requirements_panel.ASSUMPTION_OPTIONS)
    keys.update(panel.CHAT_WIDTHS)
    keys.update(errors._MESSAGES.values())
    keys.add(business_intake_nodes.DEFAULT_SINGLE_PRODUCT_ID)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_app_labels", PROJECT_ROOT / "app.py"
    )
    app_tree = ast.parse((PROJECT_ROOT / "app.py").read_text(encoding="utf-8"))

    for node in app_tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "STEP_LABELS"
                for target in node.targets
            )
        ):
            keys.update(ast.literal_eval(node.value))

    del spec
    return keys


def collect_translation_keys() -> set[str]:
    """Alle deutschen Texte, für die eine Übersetzung existieren muss."""

    return {
        key
        for key in _static_keys() | _constant_keys()
        if key.strip()
    }
