FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/arctic data/orders data/results data/monitoring data/alerts data/cache

# Expose ports
# 8000 - Admin API
# 8501 - Streamlit Dashboard
# 3000 - Dagster UI (optional)
EXPOSE 8000 8501

# Default command: run admin API
CMD ["python", "run_admin.py"]
