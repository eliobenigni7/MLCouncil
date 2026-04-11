FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements_api.txt ./
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/arctic data/orders data/results data/monitoring data/alerts data/cache data/paper_trades

EXPOSE 8000 8501

CMD ["python", "run_admin.py"]
