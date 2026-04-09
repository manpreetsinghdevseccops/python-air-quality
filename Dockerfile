# Use Python 3.10 slim - compatible with older pandas versions
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for:
# - Building Python packages (gcc, g++, etc.)
# - AWS CLI runtime
# - Shell script execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    curl \
    unzip \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2 (required by startup.sh)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Upgrade pip and install setuptools explicitly
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Copy and make startup script executable
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set ownership of application directory and ensure runtime file creation permissions
RUN chown -R appuser:appuser /app && \
    mkdir -p /app/logs /app/temp && \
    chown -R appuser:appuser /app/logs /app/temp

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Use startup script as entrypoint
ENTRYPOINT ["sh","/startup.sh"]