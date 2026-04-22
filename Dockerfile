# Production-ready Dockerfile for Python Air Quality ML Application
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies and AWS CLI in a single layer
RUN apt-get update && \
    apt-get install --no-install-recommends -y --no-install-recommends \
        curl \
        unzip \
        ca-certificates && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws && \
    apt-get remove -y curl unzip && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1001 appuser

# Copy requirements first for better layer caching
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and directories with proper ownership
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser startup.sh .
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser models ./models

# Make startup script executable
RUN chmod +x startup.sh

# Switch to non-root user
USER appuser

# Set entrypoint
ENTRYPOINT ["bash", "startup.sh"]