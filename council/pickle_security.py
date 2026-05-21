"""Fail-closed pickle loading with mandatory SHA-256 sidecars for trusted artifacts."""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any


class PickleHashPolicyError(ValueError):
    """Raised when a trusted pickle artifact violates hash sidecar policy."""


def pickle_hash_path(path: Path | str) -> Path:
    artifact_path = Path(path)
    return artifact_path.with_suffix(artifact_path.suffix + ".hash")


def write_pickle_hash_sidecar(path: Path | str) -> str:
    """Write ``<artifact>.hash`` containing the SHA-256 hex digest of *path*."""
    artifact_path = Path(path)
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    pickle_hash_path(artifact_path).write_text(digest, encoding="utf-8")
    return digest


def verify_pickle_hash_sidecar(path: Path | str) -> str:
    """Verify ``<artifact>.hash`` exists and matches *path*."""
    artifact_path = Path(path)
    hash_path = pickle_hash_path(artifact_path)
    if not hash_path.exists():
        raise PickleHashPolicyError(
            f"Missing hash sidecar for trusted pickle artifact: {hash_path}. "
            "Write a sidecar with write_pickle_hash_sidecar() or use "
            "trusted_pickle_load(..., require_hash=False) only in local/test code."
        )
    expected = hash_path.read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if actual != expected:
        raise PickleHashPolicyError(
            f"Checkpoint hash mismatch for {artifact_path}: "
            f"expected {expected}, got {actual}. "
            "File may be corrupted or tampered with."
        )
    return actual


def trusted_pickle_load(
    path: Path | str,
    *,
    require_hash: bool = True,
) -> Any:
    """Load a pickle file after optional fail-closed hash verification.

    Parameters
    ----------
    path:
        Pickle artifact path.
    require_hash:
        When True (default), a matching ``.hash`` sidecar is mandatory.
        Set False only for clearly local/test escape hatches; setting
        ``MLCOUNCIL_ALLOW_UNHASHED_PICKLE=1`` also disables the requirement.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Pickle artifact not found: {artifact_path}")

    enforce = require_hash and os.getenv("MLCOUNCIL_ALLOW_UNHASHED_PICKLE", "").strip() not in {
        "1",
        "true",
        "True",
        "yes",
        "YES",
    }
    if enforce:
        verify_pickle_hash_sidecar(artifact_path)
    elif pickle_hash_path(artifact_path).exists():
        verify_pickle_hash_sidecar(artifact_path)

    with artifact_path.open("rb") as handle:
        return pickle.load(handle)
