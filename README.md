# agentic-optimization

Repository for the agentic optimization project.

## 1. Introduction

This project explores how agentic AI can support users in formulating structured mathematical optimization models from business requirements.

## 2. Developer Environment Setup

The project uses `uv` as the Python package and environment manager.

> [!NOTE]
> It is also possible to ignore `uv`, manually create a Python virtual environment, and install the dependencies listed in `pyproject.toml` with `pip`.
> However, using `uv` is recommended to ensure a consistent setup across the team.

### 2.1. Prerequisites

- Python 3.13 or newer
- `uv` package manager

If `uv` is not installed, follow the official installation instructions:

[Astral.sh uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

After installing `uv` for the first time, it may be necessary to reopen the terminal or IDE before the command is available.

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
> If problems arise, you can follow [Astral.sh UV Locking and Syncing](https://docs.astral.sh/uv/concepts/projects/sync/#creating-the-lockfile)
> 
>As a quick local fix, deleting `uv.lock` and running `uv sync` again may help. This should only be done carefully, because it can change the resolved dependency versions.

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

The application requires access to a Large Language Model. The configuration can either be supplied through a local `.env` file or by manually setting the required environment variables.

For local development, copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```ps1
Copy-Item .env.example .env
```

Then edit the `.env` file and provide the required values:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=
```

The following variables are required:

* `LLM_PROVIDER`: The LLM provider to use. Supported values are `openai`, `anthropic`, and `groq`.
* `LLM_MODEL`: The model name for the selected provider.
* `LLM_API_KEY`: The API key for the selected provider.

Example configurations:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=your-openai-api-key
```

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
LLM_API_KEY=your-groq-api-key
```

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-haiku-latest
LLM_API_KEY=your-anthropic-api-key
```

> [!IMPORTANT]
> Never commit real API keys to the repository. The `.env` file should stay local and should be listed in `.gitignore`. Only `.env.example` should be committed.

Alternatively, the variables can be set manually in the shell.

Unix shell:

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4.1-mini
export LLM_API_KEY=your-api-key
```

Windows PowerShell:

```ps1
$env:LLM_PROVIDER = "openai"
$env:LLM_MODEL = "gpt-4.1-mini"
$env:LLM_API_KEY = "your-api-key"
```
