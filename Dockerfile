FROM node:20-alpine AS frontend-build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ARM64 / CPU-only: force torch without CUDA to save disk space
COPY requirements.txt requirements_api.txt ./
RUN pip install --no-cache-dir --timeout 120 torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --timeout 120 -r requirements.txt

ENV PYTHONPATH=/app \
    MLCOUNCIL_ENV_PROFILE=prod \
    MLCOUNCIL_USE_PRODUCTION_MANIFEST=true \
    MLCOUNCIL_OTEL_ENABLED=false

COPY . .

COPY --from=frontend-build /app/dist /app/api/static/spa

RUN mkdir -p \
    data/raw data/arctic data/orders data/results data/monitoring \
    data/alerts data/cache data/paper_trades data/operations data/dagster \
    models/checkpoints

EXPOSE 8000 3000

CMD ["python", "run_admin.py"]
