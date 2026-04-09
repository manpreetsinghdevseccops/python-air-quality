# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies required by startup.sh
# - awscli: Required for AWS SSM parameter retrieval in startup.sh
# - bash: Required to execute startup.sh
# - ca-certificates: For SSL/TLS connections to AWS
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    awscli \
    bash \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Copy and make startup script executable
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set proper ownership for runtime file creation (.env, logs, etc.)
# This is CRITICAL - the startup.sh script creates .env file at runtime
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check (optional - adjust port if needed)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#   CMD python -c "import sys; sys.exit(0)"

# Use startup.sh as entrypoint (it will fetch AWS SSM params and run main.py)
ENTRYPOINT ["sh","startup.sh"]