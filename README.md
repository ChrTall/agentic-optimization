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
> If problems arise you can follow [Astral.sh UV Locking and Syncing](https://docs.astral.sh/uv/concepts/projects/sync/#creating-the-lockfile)
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
