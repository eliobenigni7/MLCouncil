"""News preprocessing utilities for FinBERT sentiment scoring.

Provides headline cleaning, deduplication, source weighting, and a persistent
SQLite cache to avoid re-scoring headlines already processed by FinBERT.
"""

from __future__ import annotations

import hashlib
import html
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Source weight table
# ---------------------------------------------------------------------------

_SOURCE_WEIGHTS: dict[str, float] = {
    "reuters": 1.0,
    "bloomberg": 1.0,
    "wsj": 1.0,
    "wall street journal": 1.0,
    "ft": 1.0,
    "financial times": 1.0,
    "benzinga": 0.7,
    "seeking alpha": 0.7,
    "motley fool": 0.7,
    "marketwatch": 0.7,
}


def assign_source_weight(source: str) -> float:
    """Return credibility weight for a news source.

    Reuters, Bloomberg, WSJ → 1.0
    Benzinga, Seeking Alpha → 0.7
    Unknown                 → 0.5
    """
    if not source:
        return 0.5
    s = source.lower().strip()
    for key, weight in _SOURCE_WEIGHTS.items():
        if key in s:
            return weight
    return 0.5


# ---------------------------------------------------------------------------
# Headline cleaning
# ---------------------------------------------------------------------------

# Mojibake patterns produced by latin-1 → UTF-8 double-encoding
_MOJIBAKE: list[tuple[str, str]] = [
    ("â€™", "'"),
    ("â€œ", "\u201c"),
    ("â€\x9d", "\u201d"),
    ("â€\x93", "\u2013"),   # en-dash
    ("â€\x94", "\u2014"),   # em-dash
    ("Ã©", "é"),
    ("Ã¨", "è"),
    ("Ã ", "à"),
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TICKER_BRACKET_RE = re.compile(r"\[[\w.]+\]")   # [AAPL], [BRK.B]
_TICKER_DOLLAR_RE = re.compile(r"\$[A-Z]{1,5}\b")  # $MSFT


def clean_headline(text: str) -> str:
    """Prepare a headline for FinBERT.

    Steps
    -----
    1. Remove HTML tags.
    2. Unescape HTML entities (``&amp;`` → ``&``).
    3. Fix common mojibake sequences.
    4. Remove inline ticker mentions (``[AAPL]``, ``$MSFT``).
    5. Collapse whitespace.
    6. Truncate to 512 characters (FinBERT token limit proxy).
    """
    if not text:
        return ""

    # 1. Strip HTML tags
    text = _HTML_TAG_RE.sub("", text)

    # 2. HTML entity decoding
    text = html.unescape(text)

    # 3. Mojibake repair
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)

    # 4. Remove ticker annotations
    text = _TICKER_BRACKET_RE.sub("", text)
    text = _TICKER_DOLLAR_RE.sub("", text)

    # 5. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Truncate
    return text[:512]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _title_hash(title: str) -> str:
    """Return a normalized hash for fuzzy title deduplication.

    Normalises to lowercase, strips non-alphanumeric characters (except
    spaces), collapses whitespace, then SHA-1 hashes the result.
    """
    normalized = re.sub(r"[^\w\s]", "", title.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha1(normalized.encode()).hexdigest()


def deduplicate_news(df: pl.DataFrame) -> pl.DataFrame:
    """Remove duplicate news rows.

    Deduplication is applied in two passes:

    1. **URL exact match** — keep first occurrence.
    2. **Fuzzy title hash** — normalised title SHA-1, keep the earliest
       record per date so older publication times win ties.

    Parameters
    ----------
    df:
        Polars DataFrame as produced by ``data.ingest.news.download_news``.
        Expected columns: ``url``, ``title``, ``valid_time``.

    Returns
    -------
    pl.DataFrame with duplicate rows removed, preserving the original schema.
    """
    date_col = "valid_time" if "valid_time" in df.columns else "date"

    # Pass 1: exact URL dedup
    if "url" in df.columns:
        df = df.unique(subset=["url"], keep="first")

    # Pass 2: fuzzy title dedup — sort by date first so "keep=first" retains
    # the earliest occurrence
    if "title" in df.columns:
        df = df.sort(date_col)
        hashes = [_title_hash(t or "") for t in df["title"].to_list()]
        df = df.with_columns(pl.Series("_title_hash", hashes))
        df = df.unique(subset=["_title_hash"], keep="first")
        df = df.drop("_title_hash")

    return df


# ---------------------------------------------------------------------------
# Sentiment cache
# ---------------------------------------------------------------------------

class SentimentCache:
    """Persistent SQLite cache for FinBERT sentiment scores.

    Keyed on ``SHA-256(headline_text)`` to avoid re-scoring identical
    headlines across days.  Pass ``db_path=":memory:"`` for in-memory
    operation (useful in tests).

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Parent directories are created
        automatically when a file path is given.
    """

    def __init__(self, db_path: str = "data/cache/sentiment.db") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sentiment_cache (
                headline_hash TEXT PRIMARY KEY,
                score         REAL    NOT NULL,
                created_at    TEXT    NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, headline: str) -> float | None:
        """Return cached score or ``None`` if not present."""
        row = self._conn.execute(
            "SELECT score FROM sentiment_cache WHERE headline_hash = ?",
            (self._hash(headline),),
        ).fetchone()
        return float(row[0]) if row is not None else None

    def set(self, headline: str, score: float) -> None:
        """Store a score.  Replaces any existing entry for the same headline."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO sentiment_cache (headline_hash, score, created_at)
            VALUES (?, ?, ?)
            """,
            (self._hash(headline), float(score), datetime.now().isoformat()),
        )
        self._conn.commit()

    def close(self) -> None:
        """Explicitly close the database connection."""
        self._conn.close()
