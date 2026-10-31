# agentic-optimization <!-- omit from toc -->

Repository for the agentic optimization project.

- [1. Introduction](#1-introduction)
- [2. Application Configuration](#2-application-configuration)
- [3. Quickstart with Docker Compose](#3-quickstart-with-docker-compose)
  - [3.1. Prerequisites](#31-prerequisites)
  - [3.2. Start the Application](#32-start-the-application)
  - [3.3. View Logs](#33-view-logs)
  - [3.4. Stop the Application](#34-stop-the-application)
- [4. Developer Environment Setup](#4-developer-environment-setup)
  - [4.1. Prerequisites](#41-prerequisites)
  - [4.2. Install Dependencies](#42-install-dependencies)
  - [4.3. Run the Application](#43-run-the-application)
  - [4.4. Run Development Commands](#44-run-development-commands)

## 1. Introduction

This project explores how agentic AI can support users in formulating structured mathematical optimization models from business requirements and solving them.

You can run the application using a prebuilt container image or set up a local development environment. Both options use the same `.env` file for application settings and secrets.

## 2. Application Configuration

The application requires access to a Large Language Model. Before starting it, create a `.env` file from the provided [`example.env`](example.env) template.

Run the following command in the project root directory.

Linux or macOS:

```bash
cp example.env .env
```

Windows PowerShell:

```powershell
Copy-Item example.env .env
```

Open `.env` and set the values according to the instructions in that file. At minimum, configure:

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | LLM provider: `openai`, `anthropic`, or `groq`. |
| `LLM_MODEL` | Model name for the selected provider. |
| `LLM_API_KEY` | API key for the selected provider. |

The template also documents optional settings such as request timeouts, retries, and temperature.

Docker Compose loads this file through the service’s `env_file` configuration. During local development, the application reads the `.env` file from the project directory.

> [!IMPORTANT]
> Keep `.env` local. It contains secrets and must remain excluded through `.gitignore` and `.dockerignore`. Commit only the `example.env` template with placeholder values.

## 3. Quickstart with Docker Compose

Run the application using the prebuilt container image. No local Python installation or development tools are required.

### 3.1. Prerequisites

- Docker with Docker Compose installed.
- The [`docker-compose.yaml`](docker-compose.yaml) and a configured `.env` file
  as described in [Application Configuration](#2-application-configuration).

### 3.2. Start the Application

Run this command in the directory containing `docker-compose.yaml` and `.env`:

```bash
docker compose up -d
```

Compose downloads the configured image if it is not already available locally and starts the application in the background.

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3.3. View Logs

```bash
docker compose logs -f
```

Press `Ctrl+C` to stop following the logs. The application continues running.

### 3.4. Stop the Application

```bash
docker compose down
```

## 4. Developer Environment Setup

The project uses [`uv`](https://docs.astral.sh/uv/) to manage Python, virtual environments, and dependencies.

Project metadata, dependency requirements, build settings, and tool configuration are defined in [`pyproject.toml`](pyproject.toml). This follows Python packaging standards, including [PEP 621](https://peps.python.org/pep-0621/) for project metadata and dependencies and [PEP 518](https://peps.python.org/pep-0518/) for build-system requirements. The configuration is not specific to uv. Other compatible Python tools can also read it.

This project uses `pyproject.toml` and `uv.lock` rather than maintaining a separate `requirements.txt`:

| File | Purpose |
|---|---|
| [`pyproject.toml`](pyproject.toml) | Declares direct dependencies and their allowed versions, development dependency groups, project metadata, and build and tool settings. |
| [`uv.lock`](uv.lock) | Records the exact resolved versions of direct and transitive application dependencies for consistent installations. |
| [`.python-version`](.python-version) | Selects the Python version used for local development and container builds. |

Application settings and secrets are configured separately through `.env`, as described in [Application Configuration](#2-application-configuration).

### 4.1. Prerequisites

- A local checkout of this repository.
- The `uv` package manager.

Follow the official [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) if `uv` is not installed. You may need to reopen your terminal or IDE afterward.

Run the following commands from the project root directory. `uv` can download the required python version set in `.python-version`  automatically if necessary.

### 4.2. Install Dependencies

```bash
uv sync --locked
```

This command:

- Downloads the projects python version if necessary.
- Creates or updates the project’s `.venv` virtual environment.
- Installs the project and its dependencies, including development dependencies.
- Uses the committed `uv.lock` without modifying it.

If the lockfile is missing or inconsistent with `pyproject.toml`, the command fails. When intentionally changing dependencies, update and commit the lockfile alongside those changes.

For example, use the following commands to add an application dependency or a development dependency:

```bash
uv add <package>
uv add --dev <package>
```

These commands update `pyproject.toml`, `uv.lock`, and the local environment. Commit both configuration files so the team uses the same dependency resolution.

See the [uv dependency management documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/) and [locking and syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/) for details.

### 4.3. Run the Application

Create and configure `.env` as described in [Application Configuration](#2-application-configuration). If you already configured it for Docker Compose, reuse that file.

Start the application:

```bash
uv run --locked streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

Press `Ctrl+C` in the terminal to stop the application.

### 4.4. Run Development Commands

Use `uv run` to execute commands inside the project environment without activating it manually:

```bash
uv run --locked python --version
uv run --locked pytest
```

Alternatively, activate the virtual environment manually.

Bash or Zsh:

```bash
source .venv/bin/activate
```

Fish:

```fish
source .venv/bin/activate.fish
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```
