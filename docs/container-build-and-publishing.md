# Building and Publishing Container Images

This page explains how to build the application container image locally and how GitHub Actions publishes it to GitHub Container Registry (GHCR). For application configuration and the Docker Compose quick start, see the project's README.

## Relevant files

| File | Purpose |
| --- | --- |
| `Dockerfile` | Defines the build environment and final runtime image. |
| `.dockerignore` | Excludes local files from the build context. |
| `.python-version` | Selects the Python version used during the build. |
| `pyproject.toml` | Defines project metadata, dependencies, and the build backend. |
| `uv.lock` | Records resolved application dependency versions. |
| `.github/workflows/main.yml` | GitHub Action that builds and publishes images on release publication or manual execution (test). |
| `docker-compose.yaml` | Starts the application using a published image. |

## Build an image locally

Install Docker or Podman and check out the version of the project you want to build. Run the build from the repository root:

```bash
docker build -t localhost/agentic-optimization:dev .
```

With Podman:

```bash
podman build -t localhost/agentic-optimization:dev .
```

The final `.` selects the current directory as the build context. The builder reads the root `Dockerfile` and applies `.dockerignore` when preparing that context. Docker's build must support the `RUN --mount` syntax used by the Dockerfile; current Docker installations use BuildKit for this.

The build needs network access to retrieve base images, Python, and dependencies. No LLM API key is needed during the build. We keep sensitive information from the `.env` and the local `.venv` excluded through `.dockerignore`.

## Run the locally built image

Copy `example.env` to `.env` if you have not already done so, then configure it according to its comments:

```bash
cp example.env .env
```

On Windows PowerShell:

```powershell
Copy-Item example.env .env
```

Start the image with the configured environment:

```bash
docker run --rm -p 8501:8501 --env-file .env localhost/agentic-optimization:dev
```

Or use Podman:

```bash
podman run --rm -p 8501:8501 --env-file .env localhost/agentic-optimization:dev
```

Open <http://localhost:8501>. Press `Ctrl+C` to stop the container. `--rm` removes the stopped container; the built image remains available locally.

The Compose file references a published GHCR image. Building an image tagged `:dev` locally does not automatically change the image selected by Compose.

## How the Dockerfile works

### Builder stage

The builder starts from `ghcr.io/astral-sh/uv:0.13.6-trixie-slim`. It contains uv, which installs Python and the project environment.

| Setting | Purpose |
| --- | --- |
| `UV_COMPILE_BYTECODE=1` | Compiles Python bytecode during installation. |
| `UV_LINK_MODE=copy` | Copies installed files instead of linking them to the mounted cache. |
| `UV_NO_DEV=1` | Excludes the development dependency group from synchronization. |
| `UV_PYTHON_INSTALL_DIR=/python` | Places managed Python installations at a path accessible to the runtime user. |
| `UV_PYTHON_PREFERENCE=only-managed` | Uses uv-managed Python instead of a system interpreter. |

The Dockerfile first copies `.python-version` into `/app` and runs:

```bash
uv python install
```

Without an explicit version argument, uv reads `.python-version`. Python is installed separately from the application so changes to application source do not invalidate this installation layer.

Dependencies are installed before the application source:

```bash
uv sync --locked --no-install-project
```

`--locked` fails if the lockfile is missing or inconsistent with the project metadata, rather than updating it. `--no-install-project` installs the selected dependencies while skipping the application package itself. The environment is created at `/app/.venv`.

The build then copies the project into `/app` and runs `uv sync --locked` again to install the application itself. Development dependencies remain excluded through `UV_NO_DEV=1`. The project uses uv's default editable installation behavior; the source directory is retained in the runtime image.

### Build mounts and caching

The dependency installation uses two mount types:

- **Bind mounts:** Temporarily expose `uv.lock` and `pyproject.toml` from the build context during that `RUN` instruction. These mounts do not themselves add the files to the image. The later `COPY . /app` adds the project files, subject to `.dockerignore`.
- **Cache mount:** Makes `/root/.cache/uv` available as a reusable package cache. Cached downloads can accelerate subsequent builds, but cache contents are not included in the runtime image.

The installed `/app/.venv` is outside these mounts and remains in the build stage. Dependency installation can reuse its image layer when only application source changes. Cache mounts additionally help when the dependency layer must be rebuilt on the same builder. The workflow does not configure persistent caching across separate GitHub-hosted runner VMs.

### Runtime stage

The final stage starts from `debian:trixie-slim`. It copies `/python` and `/app` from the builder, without copying the builder's uv installation or package cache.

Python is copied to the same absolute path, `/python`, because the virtual environment refers to its interpreter there. Installing Python under `/root` would make that interpreter inaccessible to the non-root runtime user.

The runtime stage:

- Creates the `agentic-optimization` user and group with UID/GID `9999`.
- Assigns ownership of `/app` to that user.
- Adds `/app/.venv/bin` to `PATH` so the installed `streamlit` executable is available.
- Sets `PYTHONUNBUFFERED=1` so Python writes logs without its usual stream buffering.
- Switches to the non-root user and starts `streamlit run app.py` from `/app`.

Dependency synchronization happens during the build, not at application startup.

## Automatic publishing with GitHub Actions

The workflow in `.github/workflows/main.yml` runs on:

| Trigger | Image tag |
| --- | --- |
| Publication of a stable release | Release tag, with an optional leading lowercase `v` removed. |
| Publication of a prerelease | Prerelease tag, with an optional leading lowercase `v` removed. |
| Manual execution through `workflow_dispatch` | `test-<workflow-run-number>`. |

For example:

| GitHub event | Published image |
| --- | --- |
| Release `v1.0.0` | `ghcr.io/chrtall/agentic-optimization:1.0.0` |
| Prerelease `v1.1.0-rc.1` | `ghcr.io/chrtall/agentic-optimization:1.1.0-rc.1` |
| Manual run number `12` | `ghcr.io/chrtall/agentic-optimization:test-12` |

The job runs on a GitHub-hosted Ubuntu VM with Docker available. It checks out the selected source, logs in to GHCR, builds the image, and pushes it. The image namespace is fixed to `chrtall`; its name comes from the repository name and is converted to lowercase.

Authentication uses the automatically generated `GITHUB_TOKEN`. The workflow grants it `contents: read` for checkout and `packages: write` for publishing. No manually configured registry password is needed. Existing GHCR packages must allow this repository's workflow to write to them.

`github.actor` supplies the login username. It does not select the destination namespace. The workflow currently has no actor restriction, so a release published by a collaborator can trigger publication to the same package.

The build adds the `org.opencontainers.image.source` label pointing to the source repository. It publishes only the selected tag: it does not create `latest`, major-version aliases, or minor-version aliases. Release names must produce valid container tags after removing the optional `v` prefix. The current job builds a single platform, normally `linux/amd64` on the selected runner; it does not configure a multi-platform build.

## Publish a release image

1. Ensure the intended commit includes the Dockerfile, workflow, and required build inputs.
2. Create a GitHub release targeting that commit and choose a version tag, for example `v1.0.0`.
3. Publish the release. Saving a draft does not trigger this workflow.
4. Open **Actions → Publish container image** and inspect the workflow result.
5. Use the resulting image tag in `docker-compose.yaml`, for example `:1.0.0`.

New GHCR packages are private by default. If anonymous pulls are intended, configure the package's visibility accordingly. Publishing an image does not update running deployments or the Compose file automatically.

## Test the publishing workflow

The workflow must be present on the repository's default branch for the manual trigger to be available.

1. Open **Actions → Publish container image → Run workflow**.
2. Select the branch to test and start the workflow.
3. Inspect the checkout, login, build, and push logs.
4. Pull and run the resulting image, replacing `test-2` with the actual tag:

```bash
docker run --rm --pull=always -p 8501:8501 --env-file .env ghcr.io/chrtall/agentic-optimization:test-2
```

Verify that the UI opens and perform an optimization with the configured provider. A successful image build and push alone does not verify LLM access or solver behavior.

To test the release trigger itself, publish a prerelease such as `v1.0.0-rc.1`.

## Reproducibility and troubleshooting

The lockfile fixes resolved application dependency versions, and `--locked` prevents silent lockfile updates. This does not guarantee byte-identical images: a minor-only Python pin permits patch updates, base-image tags can move, the runner can change, and isolated build dependencies such as Hatchling are not necessarily fixed by the application lockfile. For tighter reproducibility, pin Python to a full version, pin base-image digests, and control build dependency versions.

| Symptom | What to check |
| --- | --- |
| Lockfile consistency error | Update `uv.lock` intentionally with uv and commit it alongside the corresponding project changes. |
| Source build fails because `cc` is missing | Check the Python version and wheel availability. Native source builds may require additional compiler tools and libraries. |
| `streamlit: permission denied` | Verify that `/app/.venv/bin/python` refers to an interpreter under `/python`, not `/root`. |
| GHCR push denied | Check `packages: write`, repository access to an existing package, and account or organization policies. |
| Compose cannot pull an image | Check the exact published tag and authenticate if the package is private. |
| Manual run button is missing | Ensure the workflow containing `workflow_dispatch` exists on the default branch. |

The Dockerfile currently repeats `WORKDIR /app` in the builder and labels its final command as FastAPI in a comment. These are cosmetic: the command actually starts Streamlit.

## References

- [uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/)
- [uv Python versions](https://docs.astral.sh/uv/concepts/python-versions/)
- [Docker build cache](https://docs.docker.com/build/cache/optimize/)
- [Publishing Docker images with GitHub Actions](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
