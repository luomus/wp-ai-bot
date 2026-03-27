# ---------------------------------------------------------------------------
# Stage 1 – build dependencies in a separate layer for better caching
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools needed by faiss-cpu and numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 – lean runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY main.py .

# The cache file is stored in a named volume mounted at /app/cache
ENV CACHE_DIR=/app/cache

# Expose the API port
EXPOSE 8000

# Healthcheck so Docker Compose knows when the service is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
