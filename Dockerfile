FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# alpaca-trade-api requires websockets<11 but yfinance>=1.2.1 requires websockets>=13.
# We only use alpaca REST (no streaming), so install alpaca without its deps.
COPY requirements*.txt ./
RUN sed -i '/alpaca-trade-api/d' requirements_api.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps 'alpaca-trade-api>=2.3.0'

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/arctic data/orders data/results data/monitoring data/alerts data/cache data/paper_trades

# Expose ports
# 8000 - Admin API
# 8501 - Streamlit Dashboard
# 3000 - Dagster UI (optional)
EXPOSE 8000 8501

# Default command: run admin API
CMD ["python", "run_admin.py"]
