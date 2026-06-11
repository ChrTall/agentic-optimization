from agentic_optimization.state import GraphState


def code_agent(state: GraphState) -> GraphState:
    """
    Erstellt auf Basis der Modellkonfiguration einen ersten PuLP-Code-Skeleton.

    Diese Version generiert noch keinen vollständig individuellen Code,
    sondern zeigt die Struktur, die später durch konkrete Parameter ergänzt wird.
    """

    model_config = state.get("model_config", {})
    model_type = model_config.get("model_type")

    if model_type == "newsvendor":
        pulp_code = _build_newsvendor_code()

    elif model_type == "linear_programming":
        pulp_code = _build_linear_programming_code()

    elif model_type == "routing":
        pulp_code = _build_routing_code()

    else:
        pulp_code = _build_generic_linear_programming_code()

    state["pulp_code"] = pulp_code

    return state


def _build_newsvendor_code() -> str:
    return '''import pulp

# Newsvendor-Modell
# Ziel: Optimale Bestellmenge bei unsicherer Nachfrage bestimmen

# Beispielparameter
selling_price = 10
purchase_cost = 5
salvage_value = 2
demand_scenarios = [80, 100, 120, 140]
scenario_probabilities = [0.25, 0.25, 0.25, 0.25]

# Modell erstellen
model = pulp.LpProblem("Newsvendor_Model", pulp.LpMaximize)

# Entscheidungsvariable
order_quantity = pulp.LpVariable("order_quantity", lowBound=0)

# Hilfsvariablen pro Szenario
sold = {
    i: pulp.LpVariable(f"sold_{i}", lowBound=0)
    for i in range(len(demand_scenarios))
}

leftover = {
    i: pulp.LpVariable(f"leftover_{i}", lowBound=0)
    for i in range(len(demand_scenarios))
}

# Nebenbedingungen
for i, demand in enumerate(demand_scenarios):
    model += sold[i] <= demand
    model += sold[i] <= order_quantity
    model += leftover[i] >= order_quantity - demand

# Zielfunktion: Erwarteter Gewinn
expected_profit = pulp.lpSum(
    scenario_probabilities[i] * (
        selling_price * sold[i]
        + salvage_value * leftover[i]
        - purchase_cost * order_quantity
    )
    for i in range(len(demand_scenarios))
)

model += expected_profit

# Optimierung lösen
model.solve()

print("Status:", pulp.LpStatus[model.status])
print("Optimale Bestellmenge:", order_quantity.value())
print("Erwarteter Gewinn:", pulp.value(model.objective))
'''


def _build_linear_programming_code() -> str:
    return '''import pulp

# Lineares Optimierungsmodell
# Beispiel: Produktionsmengen unter Kapazitätsrestriktionen optimieren

model = pulp.LpProblem("Linear_Optimization_Model", pulp.LpMaximize)

# Entscheidungsvariablen
x1 = pulp.LpVariable("product_1_quantity", lowBound=0)
x2 = pulp.LpVariable("product_2_quantity", lowBound=0)

# Zielfunktion
model += 30 * x1 + 50 * x2

# Nebenbedingungen
model += 2 * x1 + 4 * x2 <= 100
model += 3 * x1 + 2 * x2 <= 90

# Optimierung lösen
model.solve()

print("Status:", pulp.LpStatus[model.status])
print("Produkt 1:", x1.value())
print("Produkt 2:", x2.value())
print("Zielfunktionswert:", pulp.value(model.objective))
'''


def _build_routing_code() -> str:
    return '''# Routing-Modell Skeleton

# Für Routing-Probleme wird häufig OR-Tools statt PuLP verwendet.
# Dieser Skeleton dient zunächst nur als Platzhalter.

print("Routing-Modell erkannt.")
print("Hier könnte später ein OR-Tools-Modell erzeugt werden.")
'''


def _build_generic_linear_programming_code() -> str:
    return '''import pulp

# Allgemeines lineares Optimierungsmodell

model = pulp.LpProblem("Generic_Linear_Optimization_Model", pulp.LpMaximize)

# Entscheidungsvariable
x = pulp.LpVariable("x", lowBound=0)

# Zielfunktion
model += x

# Beispielhafte Nebenbedingung
model += x <= 100

# Optimierung lösen
model.solve()

print("Status:", pulp.LpStatus[model.status])
print("x:", x.value())
print("Zielfunktionswert:", pulp.value(model.objective))
'''