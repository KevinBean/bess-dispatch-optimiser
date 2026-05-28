# BESS Dispatch Optimiser demo image (Streamlit + agent + optimiser).
# Multi-stage isn't needed — single stage, CPU-only torch to keep size sane (~2GB).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# System deps: build tools for any wheels that need compiling, then trimmed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first (avoids pulling the multi-GB CUDA wheel).
RUN pip install --no-cache-dir torch==2.6.* --index-url https://download.pytorch.org/whl/cpu

COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

# Application code + baked artifacts (model, processed prices, Chroma store).
# Build artifacts locally first:  make artifacts
COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY docs ./docs
COPY models ./models
COPY data/processed ./data/processed
COPY chroma_store ./chroma_store
COPY pyproject.toml README.md ./

EXPOSE 8501

# Streamlit health endpoint for ECS/ALB checks (Cloud Run uses its own startup probe).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Shell form so ${PORT} expands — Cloud Run injects PORT (8080); local defaults to 8501.
CMD streamlit run app/streamlit_app.py \
    --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
