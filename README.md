# agentic-optimization

Repository for the agentic optimization project.

## 1. Introduction

This project explores how agentic AI can support users in formulating structured mathematical optimization models from
business requirements.

## 2. Developer Environment Setup

The project uses `uv` as the Python package and environment manager.

> [!NOTE]
> It is also possible to ignore `uv`, manually create a Python virtual environment, and install the dependencies listed
> in `pyproject.toml` with `pip`.
> However, using `uv` is recommended to ensure a consistent setup across the team.

### 2.1. Prerequisites

- Python 3.13 or newer
- `uv` package manager

If `uv` is not installed, follow the official installation instructions:

[Astral.sh uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

After installing `uv` for the first time, it may be necessary to reopen the terminal or IDE before the command is
available.

### 2.2. Workspace Setup with `uv`

After installing `uv`, the workspace setup consists of a single synchronization step.

Run the following command in the project root directory:

```bash
uv sync
```

This command

- creates or updates the project’s virtual environment, and
- installs all dependencies specified in `pyproject.toml`
- uses `uv.lock` to create a reproducible dependency setup.

The virtual environment location and activation are handled by `uv`.

> [!TIP]
> If problems arise, you can
> follow [Astral.sh UV Locking and Syncing](https://docs.astral.sh/uv/concepts/projects/sync/#creating-the-lockfile)
>
>As a quick local fix, deleting `uv.lock` and running `uv sync` again may help. This should only be done carefully,
> because it can change the resolved dependency versions.

Commands can be run inside the project environment with:

``` bash
uv run <command>
```

For example:

```bash
uv run python --version
```

Alternatively, the virtual environment can be activated manually.

Unix shell:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```ps1
.\.venv\Scripts\activate.ps1
```

### 2.3. Environment Variable Setup

The application requires access to a Large Language Model. The configuration can either be supplied through a local
`.env` file or by manually setting the required environment variables.

For local development, copy the example environment file:

```bash
cp example.env .env
```

On Windows PowerShell:

```ps1
Copy-Item example.env .env
```

Then edit the `.env` file and provide the required values:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.4-mini
LLM_API_KEY=
```

The following variables are required:

* `LLM_PROVIDER`: The LLM provider to use. Supported values are `openai`, `anthropic`, and `groq`.
* `LLM_MODEL`: The model name for the selected provider.
* `LLM_API_KEY`: The API key for the selected provider.

Optional variables:

* `LLM_TIMEOUT_SECONDS`: Request timeout in seconds (default: `60`).
* `LLM_MAX_RETRIES`: Automatic retries on rate limits and transient errors (default: `2`).
* `LLM_TEMPERATURE`: Sampling temperature, e.g. `0` for more reproducible model selection. Leave it unset for
  reasoning models that do not accept a temperature.

If the provider rejects a request (invalid key, rate limit, timeout), the app shows a message instead of a traceback.

Example configurations (small models such as `llama-3.1-8b-instant` are cheap but return invalid structured
answers more often; for demos a model like `gpt-5.4-mini` or `claude-haiku-4-5` is more reliable):

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.4-mini
LLM_API_KEY=your-openai-api-key
```

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
LLM_API_KEY=your-groq-api-key
```

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5
LLM_API_KEY=your-anthropic-api-key
```

> [!IMPORTANT]
> Never commit real API keys to the repository. The `.env` file should stay local and should be listed in `.gitignore`.
> Only `example.env` should be committed.

Alternatively, the variables can be set manually in the shell.

Unix shell:

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-5.4-mini
export LLM_API_KEY=your-api-key
```

Windows PowerShell:

```ps1
$env:LLM_PROVIDER = "openai"
$env:LLM_MODEL = "gpt-5.4-mini"
$env:LLM_API_KEY = "your-api-key"
```


## 3. Chat-Assistent

Beim Start der App ist links in der Seitenleiste ein Chat-Assistent geöffnet. Er führt im Gespräch Schritt für Schritt
durch alle Angaben, die für die Berechnung gebraucht werden, und ist eine gleichwertige Alternative zu den Formularen –
beides lässt sich jederzeit mischen.

- **Interview:** Der Assistent fragt immer genau die nächste offene Angabe ab (Rückfragen, Produkte, Zahlenwerte).
  Mehrere Angaben auf einmal sind ebenfalls möglich.
- **Hilfe:** Bei Unklarheiten erklärt er Begriffe und hilft beim Schätzen; geschätzte Werte übernimmt er erst nach
  Bestätigung.
- **Überprüfung / Ergebnis:** Fragen werden beantwortet; die Berechnung startet nur nach ausdrücklicher Bestätigung.

- **Gedächtnis:** Alles, was Sie im Chat schreiben, wird in der Sitzung gespeichert. Bevor der Assistent eine Frage
  stellt, prüft er Ihre bisherigen Aussagen und übernimmt bereits genannte Werte – mit wörtlichem Zitat als Beleg.
  Unter „🧠 Das weiß ich schon“ sehen Sie jederzeit, was bekannt ist.
- **Antwort-Buttons:** Rückfragen lassen sich per Klick beantworten.
- **Beschreibung nur im Chat:** Formular 1 hat kein Freitextfeld und keinen Dokumentenupload mehr. Was der Assistent
  erkennt, erscheint als **Erkannte Anforderungen** (aus dem Text, mit Zitat) und gesondert gekennzeichnete
  **Annahmen**. Jeder Eintrag kann bestätigt, geändert oder verworfen werden. In späteren Schritten bleibt die
  Übersicht als aufklappbarer Bereich **Bekannte Angaben** sichtbar und bearbeitbar.
- **Formular 3:** Alle Felder bleiben sichtbar und an ihrer Position, auch nachdem sie ausgefüllt sind. Der
  Produktname ist änderbar (Fallback „Produkt 1“). Zusätzliche kompatible Bausteine lassen sich im aufklappbaren
  Bereich **Zusätzliche Bausteine** an- und abwählen – oder im Chat nennen. Sind alle Pflichtangaben vorhanden, weist
  der Assistent auf diese Bausteine hin; weiter geht es mit **Weiter zur Überprüfung**.
- **Ergebnis:** Der optimierte Wert heißt so, wie er gemeint ist (z. B. „Erwarteter Gewinn“).
- **Chatbreite:** oben im Chat zwischen Schmal, Mittel, Breit und halber Fensterbreite umschaltbar.
- **Sprache:** oben im Chat zwischen **Deutsch** und **English** umschaltbar. Die gesamte Oberfläche (Formulare,
  Felder, Meldungen, Fehlertexte, Ergebnisse) wird übersetzt; neue Antworten des Assistenten kommen in der gewählten
  Sprache. Bereits geschriebene Chatnachrichten bleiben in ihrer ursprünglichen Sprache. Übersetzungen liegen in
  `agentic_optimization/i18n/en.py` (deutscher Text als Schlüssel, `t("…")` im Code); ein Test stellt sicher, dass
  jeder Text übersetzt ist.
- **Eine Sprechblase pro Runde:** Antwort, Hinweise und nächste Frage erscheinen zusammen in einer Nachricht;
  Antwortmöglichkeiten stehen nur noch als Buttons darunter.
- **Sichere Zuordnung:** Ein Wert wird nicht einem Feld zugeordnet, wenn der Satz ausdrücklich eine andere Größe nennt
  (z. B. „Verkaufspreis von 25 €“ landet nie bei den Einkaufskosten) – gilt für Chat und Erinnern
  (`agentic_optimization/ui/chat_mentions.py`).
- **Bausteine aus dem Gespräch:** Ausdrücklich genannte Vorgaben wie „Höchstbestellmenge von 250“, „Budget 5.000 €“,
  „10er-Gebinde“ oder „Servicegrad 95 %“ werden in Formular 3 automatisch angewählt und der Wert eingetragen – auch
  wenn sie schon in der Beschreibung standen. Im Formular abgewählte Bausteine werden nicht erneut hinzugefügt.
- **Werte per Chat ändern:** Nennt man im Chat einen neuen Wert für ein bereits ausgefülltes Feld (z. B. „Die
  Höchstbestellmenge beträgt 270“), wird er im Formular ersetzt.
- **Strenges Erinnern:** Aus früheren Aussagen wird ein Zahlenwert nur übernommen, wenn die Zahl im Zitat steht und
  Einheit und Begriff passen („Restwert von 3 Euro“ wird nie zur Höchstbestellmenge, „jedes Jahr“ nie zu 365 Tagen).
  Ein Umbenennen des Produkts löst kein erneutes Erinnern aus.
- **Rückfragen zuerst:** Eine Bestätigung in „Erkannte Anforderungen“ springt nicht an einer noch offenen Rückfrage
  vorbei zu Formular 3.
- **Geschwindigkeit:** Mit `LLM_REASONING_EFFORT=low` (oder `minimal`) in der `.env` antworten Reasoning-Modelle wie
  `gpt-5.4-mini` deutlich schneller. Frühere Aussagen werden pro offener Angabe nur einmal durchsucht, damit pro
  Chatnachricht nur ein Modellaufruf nötig ist.
- **Sofortige Übernahme:** Eingaben in den Formularen (Zahlen, Produktname, Antworten auf Rückfragen) werden ohne
  Bestätigungsknopf übernommen; der Assistent sieht sie sofort und stellt direkt die nächste Frage.
- **Ehrliche Grenzen:** Passt eine Situation zu keinem verfügbaren Rechenmodell, sagt der Assistent das klar, statt
  weiter nachzufragen. Bleibt eine Frage zweimal unbeantwortet, bietet er Auswege an.
- **Produktname:** Bei einem einzelnen Produkt wird dessen Name aus der Beschreibung übernommen (z. B. „Weihnachtsbaum“),
  sonst „Produkt 1“.

Alle Werte aus dem Chat laufen durch dieselbe Validierung und dieselben Workflow-Events wie die Formulare.
Relevante Dateien:

- `agentic_optimization/agents/chat_assistant_agent.py` – LLM-Aufruf und Bereinigung der Antwort
- `agentic_optimization/ui/chat_assistant_builder.py` – Fragenreihenfolge, Kontext je Schritt, Zuordnung der
  Chat-Werte (UI-unabhängig)
- `agentic_optimization/ui/chat_assistant_panel.py` – Streamlit-Darstellung in der Seitenleiste

## 4. Auswertung der Modellauswahl

`evals/model_selection_cases.json` enthält Problembeschreibungen mit den jeweils akzeptierten Ergebnissen
(Grundmodell, `needs_clarification` oder `unsupported`). Das Skript schickt jeden Fall mehrfach durch denselben
Geschäftseinstieg wie die App und gibt Trefferquote und Stabilität aus. **Es ruft das konfigurierte Sprachmodell auf
und verursacht API-Kosten.**

```bash
uv run python -m agentic_optimization.evaluation.model_selection              # alle Fälle, je 3 Durchläufe
uv run python -m agentic_optimization.evaluation.model_selection --repeats 5 --output report.json
uv run python -m agentic_optimization.evaluation.model_selection --case qr_tshirts
```

Neue Fälle einfach in der JSON-Datei ergänzen. Mit `LLM_TEMPERATURE=0` (sofern das Modell es unterstützt) werden
die Ergebnisse reproduzierbarer.
