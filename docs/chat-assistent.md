# Chat-Assistent

Beim Start der App ist links in der Seitenleiste ein Chat-Assistent geöffnet. Er führt im Gespräch Schritt für Schritt
durch alle Angaben, die für die Berechnung gebraucht werden, und ist eine gleichwertige Alternative zu den Formularen –
beides lässt sich jederzeit mischen.

- **Interview:** Der Assistent fragt immer genau die nächste offene Angabe ab (Rückfragen, Produkte, Zahlenwerte).
  Mehrere Angaben auf einmal sind ebenfalls möglich.
- **Hilfe:** Bei Unklarheiten erklärt er Begriffe und hilft beim Schätzen; geschätzte Werte übernimmt er erst nach
  Bestätigung.
- **Überprüfung / Ergebnis:** Fragen werden beantwortet; die Berechnung startet nur nach ausdrücklicher Bestätigung.

Alle Werte aus dem Chat laufen durch dieselbe Validierung und dieselben Workflow-Events wie die Formulare.
Relevante Dateien:

- `agentic_optimization/agents/chat_assistant_agent.py` – LLM-Aufruf und Bereinigung der Antwort
- `agentic_optimization/ui/chat_assistant_builder.py` – Fragenreihenfolge, Kontext je Schritt, Zuordnung der
  Chat-Werte (UI-unabhängig)
- `agentic_optimization/ui/chat_assistant_panel.py` – Streamlit-Darstellung in der Seitenleiste