def render_model_markdown(
    model_config: dict,
    requirements: dict,
) -> str:
    """
    Erstellt eine Markdown-Darstellung der ausgewählten Modellvariante.
    """

    family_id = model_config.get("family_id")
    variant_id = model_config.get("variant_id")

    if (
        family_id == "newsvendor"
        and variant_id == "single_product_single_period"
    ):
        return _render_newsvendor_single_product_single_period_markdown(
            model_config,
            requirements,
        )

    return _render_generic_markdown(
        model_config,
        requirements,
    )


def _value_or_dash(value) -> str:
    if value is None or value == "":
        return "–"

    if isinstance(value, list):
        if not value:
            return "–"

        return ", ".join(
            _format_value(item)
            for item in value
        )

    return _format_value(value)


def _format_value(value) -> str:
    if value is None or value == "":
        return "–"

    if isinstance(value, (int, float)):
        return _format_number(value)

    return str(value)


def _format_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return f"{value:g}"


def _render_newsvendor_single_product_single_period_markdown(
    model_config: dict,
    requirements: dict,
) -> str:
    decision = _value_or_dash(
        requirements.get("decision")
    )
    objective = _value_or_dash(
        requirements.get("objective")
    )
    demand_type = _value_or_dash(
        requirements.get("demand_type")
    )
    original_constraints = _value_or_dash(
        requirements.get("constraints")
    )

    model_parameters = requirements.get(
        "model_parameters",
        {},
    )

    newsvendor_params = (
        model_parameters
        .get("newsvendor", {})
        .get("single_product_single_period", {})
    )

    demand_scenarios = _value_or_dash(
        newsvendor_params.get("demand_scenarios")
    )
    scenario_probabilities = _value_or_dash(
        newsvendor_params.get("scenario_probabilities")
    )
    selling_price = _value_or_dash(
        newsvendor_params.get("selling_price")
    )
    purchase_cost = _value_or_dash(
        newsvendor_params.get("purchase_cost")
    )
    salvage_value = _value_or_dash(
        newsvendor_params.get("salvage_value")
    )

    family_name = model_config.get(
        "family_name",
        "Newsvendor",
    )
    variant_name = model_config.get(
        "variant_name",
        "Single Product, Single Period",
    )

    explanation = model_config.get(
        "explanation",
        "",
    )

    return f"""
### {family_name}

**Variante:** {variant_name}

{explanation}

#### Erkannte Problemstruktur

| Element | Erkannter Wert |
|---|---|
| Entscheidung | {decision} |
| Ziel | {objective} |
| Nachfrage | {demand_type} |
| Ursprünglich genannte Nebenbedingungen | {original_constraints} |

#### Entscheidungsvariable

| Symbol | Bedeutung |
|---|---|
| `q` | Bestellmenge für das einzelne Produkt |

#### Parameter

| Symbol | Bedeutung | Erfasster Wert |
|---|---|---|
| `D` | Unsichere Nachfrage | {demand_scenarios} |
| `P(D)` | Szenario-Wahrscheinlichkeiten | {scenario_probabilities} |
| `p` | Verkaufspreis pro Einheit | {selling_price} |
| `c` | Einkaufskosten pro Einheit | {purchase_cost} |
| `s` | Restwert pro unverkaufter Einheit | {salvage_value} |

#### Szenarioabhängige Hilfsvariablen

| Symbol | Bedeutung |
|---|---|
| `sold_i` | Verkaufte Menge im Szenario `i` |
| `leftover_i` | Restbestand im Szenario `i` |

#### Zielfunktion

```text
maximize

sum_i probability_i * (
    selling_price * sold_i
    + salvage_value * leftover_i
)
- purchase_cost * order_quantity
```

#### Feste Modellbedingungen

```text
order_quantity >= 0
sold_i <= demand_i
sold_i + leftover_i == order_quantity
sold_i >= 0
leftover_i >= 0
```

#### Interpretation

Das Modell bestimmt die optimale Bestellmenge `q` für ein einzelnes Produkt
und eine einzelne Periode unter unsicherer Nachfrage.

Für jedes Nachfrageszenario wird berechnet, wie viele Einheiten verkauft
werden und wie viele Einheiten als Restbestand übrig bleiben. Die
Zielfunktion maximiert den erwarteten Gewinn über alle Szenarien.
"""

def _render_generic_markdown(
    model_config: dict,
    requirements: dict,
) -> str:
    decision = _value_or_dash(
        requirements.get("decision")
    )
    objective = _value_or_dash(
        requirements.get("objective")
    )
    constraints = _value_or_dash(
        requirements.get("constraints")
    )

    family_name = model_config.get(
        "family_name",
        "Keine unterstützte Modellfamilie",
    )
    variant_name = model_config.get("variant_name")
    explanation = model_config.get(
        "explanation",
        "",
    )

    title = family_name

    if variant_name:
        title = f"{family_name} – {variant_name}"

    return f"""
### {title}

{explanation}

#### Erkannte Problemstruktur

| Element | Erkannter Wert |
|---|---|
| Entscheidung | {decision} |
| Ziel | {objective} |
| Nebenbedingungen | {constraints} |

#### Hinweis

Für diese Problemstruktur ist derzeit keine vollständig unterstützte
Modellvariante registriert.
"""