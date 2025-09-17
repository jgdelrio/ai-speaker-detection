# Multi-stage build for minimal Lambda image
# Stage 1: Build dependencies
FROM python:3.12-slim as builder

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project configuration
COPY pyproject.toml ./
RUN uv sync --no-dev

# Install PyTorch CPU-only first (largest dependency) - use latest version compatible with Python 3.12
RUN uv pip install --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    torch>=2.2.0 torchaudio>=2.2.0

# Install remaining dependencies with compatible versions
#RUN uv pip install --no-cache \
#    fastapi>=0.104.0 \
#    uvicorn>=0.24.0 \
#    mangum>=0.17.0 \
#    awslambdaric>=2.0.0 \
#    speechbrain>=1.0.0 \
#    numpy>=1.24.0 \
#    scipy>=1.11.0 \
#    pydantic>=2.5.0 \
#    python-multipart>=0.0.6

# Clean up unnecessary files in virtual environment
RUN find /.venv -name "*.pyc" -delete && \
    find /.venv -name "__pycache__" -type d -exec rm -rf {} + && \
    find /.venv -name "*.pyo" -delete && \
    find /.venv -name "tests" -type d -exec rm -rf {} + && \
    find /.venv -name "test" -type d -exec rm -rf {} + && \
    rm -rf /.venv/lib/python*/site-packages/torch/test && \
    rm -rf /.venv/lib/python*/site-packages/torchaudio/test*

# Stage 2: Runtime image
FROM public.ecr.aws/lambda/python:3.12

# Copy only the virtual environment from builder
COPY --from=builder /.venv /var/task/.venv

# Set PATH to use virtual environment
ENV PATH="/var/task/.venv/bin:$PATH"
ENV PYTHONPATH="/var/task/.venv/lib/python3.12/site-packages:${PYTHONPATH}"

# Copy only essential application code
COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY lambda_main.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["lambda_main.handler"]
