#!/bin/bash
# Post-backtest automation script
# Run after backtest completes to sync results from the Dagster container.

set -e

PROJECT=/mnt/data/MLCouncil
CONTAINER=mlcouncil-dagster-1

echo "=== Syncing backtest results from container ==="
docker cp $CONTAINER:/app/data/results/. $PROJECT/data/results/

echo "=== Results files ==="
ls -la $PROJECT/data/results/

echo "=== Cleaning results cache ==="
docker exec $CONTAINER rm -rf /app/data/results/__pycache__ 2>/dev/null || true

echo "=== Done ==="
