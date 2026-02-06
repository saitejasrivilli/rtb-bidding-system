# Multi-stage Dockerfile for RTB System
# Optimized for production deployment

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.10-slim

# Set labels
LABEL maintainer="rtb-team@example.com"
LABEL version="1.0.0"
LABEL description="Real-Time Bidding System with Budget Pacing"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app \
    WORKERS=4 \
    PORT=8000

# Create non-root user
RUN useradd -m -u 1000 rtbuser && \
    mkdir -p /app /app/logs /app/models && \
    chown -R rtbuser:rtbuser /app

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/rtbuser/.local

# Copy application code
COPY --chown=rtbuser:rtbuser . .

# Switch to non-root user
USER rtbuser

# Add local Python packages to PATH
ENV PATH=/home/rtbuser/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run application
CMD ["python", "-m", "uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
