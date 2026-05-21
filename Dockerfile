FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements_api.txt ./
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

ENV PYTHONPATH=/app \
    MLCOUNCIL_ENV_PROFILE=prod \
    MLCOUNCIL_USE_PRODUCTION_MANIFEST=true \
    MLCOUNCIL_OTEL_ENABLED=false

COPY . .

RUN mkdir -p \
    data/raw data/arctic data/orders data/results data/monitoring \
    data/alerts data/cache data/paper_trades data/operations data/dagster \
    models/checkpoints

EXPOSE 8000 8501 3000

CMD ["python", "run_admin.py"]
