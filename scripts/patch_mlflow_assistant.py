from __future__ import annotations

from pathlib import Path

import mlflow


PATCH_MARKER = "MLCOUNCIL_PRIVATE_ASSISTANT_PATCH"


def main() -> None:
    api_path = Path(mlflow.__file__).resolve().parent / "server" / "assistant" / "api.py"
    source = api_path.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        return

    original = """import ipaddress
import uuid
"""
    replacement = """import ipaddress
import os
import uuid
"""

    guard_original = """    if not ip.is_loopback:
        raise HTTPException(status_code=403, detail=_BLOCK_REMOTE_ACCESS_ERROR_MSG)
"""
    guard_replacement = """    allow_private_network = os.getenv(
        "MLFLOW_ASSISTANT_ALLOW_PRIVATE_NETWORK",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}

    # MLCOUNCIL_PRIVATE_ASSISTANT_PATCH
    if not ip.is_loopback and not (allow_private_network and ip.is_private):
        raise HTTPException(status_code=403, detail=_BLOCK_REMOTE_ACCESS_ERROR_MSG)
"""

    updated = source.replace(original, replacement, 1).replace(
        guard_original,
        guard_replacement,
        1,
    )
    api_path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
