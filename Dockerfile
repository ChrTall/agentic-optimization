
# Adopted from https://github.com/astral-sh/uv-docker-example/blob/main/standalone.Dockerfile
# An example of using standalone Python builds with multistage images.

# First, build the application in the `/app` directory
FROM ghcr.io/astral-sh/uv:0.13.6-trixie-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Omit development dependencies
ENV UV_NO_DEV=1

# Configure the Python directory so it is consistent
ENV UV_PYTHON_INSTALL_DIR=/python

# Only use the managed Python version
ENV UV_PYTHON_PREFERENCE=only-managed

WORKDIR /app
COPY .python-version ./
# Install Python before the project for caching
RUN uv python install

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Then, use a final image without uv
FROM debian:trixie-slim

# Setup a non-root user
RUN groupadd --system --gid 9999 agentic-optimization \
 && useradd --system --gid 9999 --uid 9999 --create-home agentic-optimization

# Copy the Python version
COPY --from=builder /python /python

# Copy the application from the builder
COPY --from=builder --chown=agentic-optimization:agentic-optimization /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Use the non-root user to run our application
USER agentic-optimization

# Use `/app` as the working directory
WORKDIR /app

# Run the FastAPI application by default
CMD ["streamlit", "run", "app.py"]
