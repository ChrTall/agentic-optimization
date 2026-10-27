"""Auswertung der Modellauswahl gegen ein festes Testset.

Jeder Fall wird mehrfach durch den echten Geschäftseinstieg geschickt
(gleicher Graph wie in der App, ohne Rückfragen zu beantworten). Gezählt
wird, wie oft das Ergebnis in der Liste der akzeptierten Ergebnisse liegt.

Aufruf (verursacht API-Kosten):

    uv run python -m agentic_optimization.evaluation.model_selection
    uv run python -m agentic_optimization.evaluation.model_selection --repeats 5
    uv run python -m agentic_optimization.evaluation.model_selection --case qr_tshirts

Mögliche Ergebnisse je Durchlauf:
    <core_id>            Grundmodell gewählt (z. B. "eoq")
    needs_clarification  Die App würde Rückfragen stellen
    unsupported          Kein unterstütztes Grundmodell
    analysis_failed      Die Analyse ist fehlgeschlagen
    error:<Typ>          Unerwarteter Fehler (z. B. API-Fehler)
"""

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

DEFAULT_CASES_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "model_selection_cases.json"
)


class EvaluationCase(TypedDict):
    id: str
    description: str
    accepted: list[str]


class CaseResult(TypedDict):
    id: str
    accepted: list[str]
    outcomes: list[str]
    hits: int
    runs: int


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[EvaluationCase]:
    """Lädt und prüft die Testfälle."""

    cases = json.loads(path.read_text(encoding="utf-8"))
    seen_ids: set[str] = set()

    for case in cases:
        if set(case) != {"id", "description", "accepted"}:
            raise ValueError(f"Ungültiger Testfall: {case!r}")
        if case["id"] in seen_ids:
            raise ValueError(f"Doppelte Fall-ID: {case['id']}")
        if not case["accepted"]:
            raise ValueError(f"Fall {case['id']} ohne akzeptierte Ergebnisse")
        seen_ids.add(case["id"])

    return cases


def classify_outcome(state: dict[str, Any]) -> str:
    """Übersetzt den Graph-Zustand nach dem Einstieg in ein Ergebnis."""

    from agentic_optimization.state import (
        WORKFLOW_BUSINESS_CASE_UNSUPPORTED,
        WORKFLOW_BUSINESS_CORE_SELECTED,
        WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED,
        WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT,
        get_workflow_status,
    )

    status = get_workflow_status(state)
    intake = state.get("business_intake", {})
    selection = intake.get("core_selection") or {}

    if status == WORKFLOW_BUSINESS_CORE_SELECTED:
        core_id = selection.get("selected_core_id") or state.get(
            "modular_model", {}
        ).get("core_id")
        return core_id or "unknown_core"

    if status == WORKFLOW_NEEDS_BUSINESS_PROFILE_INPUT:
        return "needs_clarification"

    if status == WORKFLOW_BUSINESS_CASE_UNSUPPORTED:
        return "unsupported"

    if status == WORKFLOW_BUSINESS_PROFILE_ANALYSIS_FAILED:
        return "analysis_failed"

    return f"status:{status}"


def run_case_once(
    workflow: Any,
    description: str,
) -> str:
    """Schickt eine Beschreibung einmal durch den Geschäftseinstieg."""

    from agentic_optimization.agents.business_intake import (
        create_initial_business_intake_state,
    )
    from agentic_optimization.state import EVENT_BUSINESS_INTAKE_REQUESTED

    try:
        state = workflow.invoke(
            {
                "control": {"event": EVENT_BUSINESS_INTAKE_REQUESTED},
                "raw_input": description,
                "business_intake": {
                    **create_initial_business_intake_state(),
                    "pending_answers": [],
                },
            }
        )
    except Exception as error:  # noqa: BLE001 - Auswertung soll weiterlaufen
        return f"error:{type(error).__name__}"

    return classify_outcome(state)


def evaluate(
    cases: list[EvaluationCase],
    run_once: Callable[[str], str],
    *,
    repeats: int,
    on_progress: Callable[[str, int, str], None] | None = None,
) -> list[CaseResult]:
    """Wertet alle Fälle mit der übergebenen Ausführungsfunktion aus."""

    results: list[CaseResult] = []

    for case in cases:
        outcomes: list[str] = []

        for run in range(repeats):
            outcome = run_once(case["description"])
            outcomes.append(outcome)

            if on_progress is not None:
                on_progress(case["id"], run + 1, outcome)

        results.append(
            {
                "id": case["id"],
                "accepted": case["accepted"],
                "outcomes": outcomes,
                "hits": sum(o in case["accepted"] for o in outcomes),
                "runs": repeats,
            }
        )

    return results


def format_report(results: list[CaseResult]) -> str:
    """Erstellt eine lesbare Übersicht."""

    lines = [
        f"{'Fall':<34} {'Treffer':>8}  Ergebnisse",
        "-" * 90,
    ]
    total_hits = 0
    total_runs = 0
    stable_cases = 0

    for result in results:
        total_hits += result["hits"]
        total_runs += result["runs"]
        stable_cases += len(set(result["outcomes"])) == 1
        counts = ", ".join(
            f"{outcome} ×{count}"
            for outcome, count in Counter(result["outcomes"]).most_common()
        )
        marker = "" if result["hits"] == result["runs"] else "  ←"
        lines.append(
            f"{result['id']:<34} {result['hits']:>3}/{result['runs']:<4}  "
            f"{counts}{marker}"
        )

    accuracy = total_hits / total_runs if total_runs else 0.0
    lines += [
        "-" * 90,
        f"Trefferquote: {total_hits}/{total_runs} = {accuracy:.0%}",
        f"Stabil (jeder Durchlauf gleich): {stable_cases}/{len(results)} Fälle",
    ]

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--cases-file", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    from agentic_optimization.configuration.llm_config import (
        load_llm_configuration,
    )
    from agentic_optimization.graph import build_graph
    from agentic_optimization.llm.factory import create_chat_model

    cases = load_cases(args.cases_file)

    if args.case:
        cases = [case for case in cases if case["id"] in set(args.case)]

    workflow = build_graph(create_chat_model(load_llm_configuration()))

    results = evaluate(
        cases,
        lambda description: run_case_once(workflow, description),
        repeats=args.repeats,
        on_progress=lambda case_id, run, outcome: print(
            f"  {case_id} #{run}: {outcome}", flush=True
        ),
    )

    print()
    print(format_report(results))

    if args.output is not None:
        args.output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nDetails gespeichert in {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
