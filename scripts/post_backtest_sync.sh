#!/bin/bash
# Post-backtest automation script
# Run after backtest completes to sync results, refresh dashboard, and verify

set -e

PROJECT=/mnt/data/MLCouncil
CONTAINER=mlcouncil-dagster-1

echo "=== Syncing backtest results from container ==="
docker cp $CONTAINER:/app/data/results/. $PROJECT/data/results/

echo "=== Results files ==="
ls -la $PROJECT/data/results/

echo "=== Cleaning dashboard cache ==="
docker exec $CONTAINER rm -rf /app/data/results/__pycache__ 2>/dev/null || true

echo "=== Restarting dashboard ==="
docker compose -f $PROJECT/docker-compose.yml restart dashboard

echo "=== Done ==="
echo "Dashboard available at http://localhost:8501"
