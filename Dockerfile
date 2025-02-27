FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data models reports logs

# Copy application code
COPY src/ src/
COPY config/ config/
COPY main.py .
COPY README.md .

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV MODELS_DIR=/app/models
ENV REPORTS_DIR=/app/reports
ENV CONFIG_PATH=/app/config/config.yaml

# Command to run the application
CMD ["python", "main.py"]