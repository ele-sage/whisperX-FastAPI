FROM nvidia/cuda:13.0.1-base-ubuntu22.04

ENV PYTHON_VERSION=3.11
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install dependencies and clean up in the same layer
# hadolint ignore=DL3008
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y update \
    && apt-get -y install --no-install-recommends \
    python3.11=3.11.0~rc1-1~22.04 \
    git \
    ffmpeg=7:4.4.2-0ubuntu0.22.04.1 \
    libcudnn9-cuda-12=9.8.0.87-1 \
    libatomic1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Install UV for package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Copy project definition files
COPY pyproject.toml .
COPY uv.lock .

# Install Python dependencies using UV (caching layer)
# This layer will be cached unless pyproject.toml or uv.lock changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project \
    && uv pip install --system ctranslate2==4.6.0 \
    && rm -rf /tmp/* /root/.uv /var/cache/*

# Copy project files
COPY app app/
COPY tests tests/
COPY app/gunicorn_logging.conf .

# Install the project itself and specific system packages
# UV automatically selects CUDA 12.8 wheels on Linux
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
    && find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type f -name '*.pyo' -delete

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Health check to verify the application is responsive
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1

ENTRYPOINT ["uv", "run", "--no-sync", "gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "0", "--log-config", "gunicorn_logging.conf", "app.main:app", "-k", "uvicorn.workers.UvicornWorker"]
