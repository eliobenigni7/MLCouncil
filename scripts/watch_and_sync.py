#!/usr/bin/env python3
"""Post-backtest automation: poll for results inside container, sync, restart dashboard.

Polls docker exec to check if equity_curve.parquet exists inside the
Dagster container. When stable, copies all results to host and restarts
the dashboard. Self-terminating after one successful sync.
"""

import subprocess
import time
import sys
from pathlib import Path

PROJECT = Path("/mnt/data/MLCouncil")
RESULTS = PROJECT / "data" / "results"
CONTAINER = "mlcouncil-dagster-1"
CONTAINER_RESULTS = "/app/data/results"
EQUITY_FILE = "equity_curve.parquet"
POLL_INTERVAL = 60  # seconds
MAX_WAIT = 8 * 3600  # 8 hours max

started = time.time()

def check_container_file(path):
    """Check if a file exists inside the container and return its age."""
    r = subprocess.run(
        ["docker", "exec", CONTAINER, "stat", "-c", "%Y", path],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        return None  # file doesn't exist
    try:
        mtime = int(r.stdout.strip())
        return time.time() - mtime
    except (ValueError, OSError):
        return None


print(f"[watcher] Waiting for backtest results inside container ({CONTAINER}:{CONTAINER_RESULTS}/{EQUITY_FILE})...")

while time.time() - started < MAX_WAIT:
    try:
        age = check_container_file(f"{CONTAINER_RESULTS}/{EQUITY_FILE}")

        if age is not None and age > 60:
            print(f"[watcher] Results detected (age={age:.0f}s). Syncing...")

            # Create local results dir
            RESULTS.mkdir(parents=True, exist_ok=True)

            # Copy results from container
            subprocess.run(
                ["docker", "cp", f"{CONTAINER}:{CONTAINER_RESULTS}/.", str(RESULTS)],
                check=False,
            )

            # Verify the copy
            if (RESULTS / EQUITY_FILE).exists():
                sz = (RESULTS / EQUITY_FILE).stat().st_size
                print(f"[watcher] Results synced ({sz} bytes). Restarting dashboard...")

            # Restart dashboard
            subprocess.run(
                ["docker", "compose", "-f", str(PROJECT / "docker-compose.yml"),
                 "restart", "dashboard"],
                check=False, cwd=str(PROJECT),
            )
            print("[watcher] Dashboard restarted. Done.")
            sys.exit(0)

        elif age is not None:
            print(f"[watcher] Results file too recent (age={age:.0f}s). Waiting...")
        else:
            elapsed = time.time() - started
            print(f"[watcher] No results yet ({elapsed/60:.0f}m elapsed). Waiting {POLL_INTERVAL}s...")

    except Exception as e:
        print(f"[watcher] Error: {e}")

    time.sleep(POLL_INTERVAL)

print("[watcher] Timed out after 8 hours. Backtest may have failed.")
sys.exit(1)
