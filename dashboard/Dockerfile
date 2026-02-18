# Stage 1: Install Python dependencies
FROM python:3.13-slim AS builder
WORKDIR /app
COPY dashboard/requirements.txt .
ENV MAKEFLAGS="-j$(nproc)"
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Stage 2: Runtime image
FROM python:3.13-slim
WORKDIR /app

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Copy application code
COPY dashboard/ ./dashboard/
COPY quant/ ./quant/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Install quant package (no dev/live deps)
RUN pip install --no-cache-dir --no-deps -e .

# Create writable dirs for backtest runs and logs
RUN mkdir -p /app/logs /app/data/cache /app/data/backtest /app/checkpoints

# Non-root user
RUN useradd -m -u 1000 dashuser && chown -R dashuser:dashuser /app
USER dashuser

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

CMD ["gunicorn", "dashboard.app:server", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "2", \
     "--timeout", "120"]
