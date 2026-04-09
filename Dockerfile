# Use Python 3.10 slim for compatibility with older package versions
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for numpy, pandas, matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser -g 1001 && \
    useradd -r -u 1001 -g appuser -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create directories for runtime files and set ownership
RUN mkdir -p /app/logs /app/temp /app/output && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port if needed (adjust based on your application)
# EXPOSE 8000

# Set the entrypoint
# Using main.py as the primary entry point based on repository structure
CMD ["python", "main.py"]