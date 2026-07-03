def render_model_markdown(model_config: dict, requirements: dict) -> str:
    """
    Erstellt eine Markdown-Darstellung des ausgewählten Optimierungsmodells.
    """

    model_type = model_config.get("model_type")

    if model_type == "newsvendor":
        return _render_newsvendor_markdown(model_config, requirements)

    if model_type == "linear_programming":
        return _render_linear_programming_markdown(model_config, requirements)

    if model_type == "routing":
        return _render_routing_markdown(model_config, requirements)

    return _render_generic_markdown(model_config, requirements)


def _value_or_dash(value) -> str:
    if value is None or value == "":
        return "–"
    return str(value)


def _render_newsvendor_markdown(model_config: dict, requirements: dict) -> str:
    decision = _value_or_dash(requirements.get("decision"))
    objective = _value_or_dash(requirements.get("objective"))
    demand_type = _value_or_dash(requirements.get("demand_type"))
    constraints = _value_or_dash(requirements.get("constraints"))

    return f"""
### {model_config.get("model_name", "Newsvendor-Modell")}

{model_config.get("explanation", "")}

#### Erkannte Problemstruktur

| Element | Erkannter Wert |
|---|---|
| Entscheidung | {decision} |
| Ziel | {objective} |
| Nachfrage | {demand_type} |
| Nebenbedingungen | {constraints} |

#### Entscheidungsvariable

| Symbol | Bedeutung |
|---|---|
| `q` | Bestellmenge |

#### Parameter

| Symbol | Bedeutung |
|---|---|
| `D` | unsichere Nachfrage |
| `p` | Verkaufspreis pro Einheit |
| `c` | Einkaufskosten pro Einheit |
| `s` | Restwert pro Einheit |

#### Abgeleitete Kennzahlen

| Kennzahl | Formel | Bedeutung |
|---|---|---|
| Underage Cost `Cu` | `p − c` | Entgangener Gewinn pro zu wenig bestellter Einheit |
| Overage Cost `Co` | `c − s` | Verlust pro übrig gebliebener Einheit |
| Critical Ratio `CR` | `Cu / (Cu + Co)` | Schwellenwert für die optimale Bestellmenge |

#### Zielfunktion

```text
maximize  E[ p · min(q, D) + s · max(q − D, 0) − c · q ]
```

#### Nebenbedingungen

```text
q ≥ 0
```

#### Interpretation

Das Modell sucht die Bestellmenge `q`, bei der der erwartete Gewinn maximal wird.
Dabei wird berücksichtigt, dass die tatsächliche Nachfrage `D` unsicher ist.
"""


def _render_linear_programming_markdown(model_config: dict, requirements: dict) -> str:
    decision = _value_or_dash(requirements.get("decision"))
    objective = _value_or_dash(requirements.get("objective"))
    constraints = _value_or_dash(requirements.get("constraints"))

    return f"""
### {model_config.get("model_name", "Lineares Optimierungsmodell")}

{model_config.get("explanation", "")}

#### Erkannte Problemstruktur

| Element | Erkannter Wert |
|---|---|
| Entscheidung | {decision} |
| Ziel | {objective} |
| Nebenbedingungen | {constraints} |

#### Entscheidungsvariable

| Symbol | Bedeutung |
|---|---|
| `x` | {decision} |

#### Mathematische Struktur

```text
maximize  cᵀ · x

subject to
    A · x ≤ b
    x ≥ 0
```

#### Interpretation

Das Modell optimiert eine lineare Zielfunktion unter linearen Nebenbedingungen.
"""


def _render_routing_markdown(model_config: dict, requirements: dict) -> str:
    decision = _value_or_dash(requirements.get("decision"))
    objective = _value_or_dash(requirements.get("objective"))
    constraints = _value_or_dash(requirements.get("constraints"))

    return f"""
### {model_config.get("model_name", "Routing- oder Transportmodell")}

{model_config.get("explanation", "")}

#### Erkannte Problemstruktur

| Element | Erkannter Wert |
|---|---|
| Entscheidung | {decision} |
| Ziel | {objective} |
| Nebenbedingungen | {constraints} |

#### Mathematische Struktur

```text
minimize  total_distance_or_cost

subject to
    routes satisfy all required deliveries
    capacity and route constraints are respected
```

#### Interpretation

Das Modell beschreibt ein Routing- oder Transportproblem. Für eine konkrete Lösung
werden später Standorte, Distanzen, Kapazitäten und Lieferrestriktionen benötigt.
"""


def _render_generic_markdown(model_config: dict, requirements: dict) -> str:
    decision = _value_or_dash(requirements.get("decision"))
    objective = _value_or_dash(requirements.get("objective"))
    constraints = _value_or_dash(requirements.get("constraints"))

    optimization_sense = "maximize" if "gewinn" in objective.lower() else "minimize"

    return f"""
### {model_config.get("model_name", "Allgemeines Optimierungsmodell")}

{model_config.get("explanation", "")}

#### Erkannte Problemstruktur

| Element | Erkannter Wert |
|---|---|
| Entscheidung | {decision} |
| Ziel | {objective} |
| Nebenbedingungen | {constraints} |

#### Mathematische Struktur

```text
{optimization_sense}  f(x)

subject to
    x ≥ 0
    {constraints}
```

#### Interpretation

Das Problem wurde als allgemeines Optimierungsproblem erkannt. Für ein konkreteres
Modell müssen Entscheidungsvariablen, Parameter und Nebenbedingungen später genauer strukturiert werden.
"""