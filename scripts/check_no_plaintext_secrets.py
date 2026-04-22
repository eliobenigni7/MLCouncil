from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub PAT", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("GitHub fine-grained PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
]

SKIP_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".parquet",
    ".zip",
    ".gz",
    ".pdf",
    ".mp4",
}


def _list_tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _is_scannable(path: Path) -> bool:
    return path.suffix.lower() not in SKIP_EXTENSIONS and path.is_file()


def main() -> int:
    findings: list[str] = []
    for rel_path in _list_tracked_files():
        if not _is_scannable(rel_path):
            continue

        try:
            text = rel_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append(f"{rel_path}:{line_no}: {label}")

    if findings:
        print("Potential plaintext secrets found:")
        for finding in findings:
            print(f"  - {finding}")
        return 1

    print("Secret scan passed (no known token patterns found).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
