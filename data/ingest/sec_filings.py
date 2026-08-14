"""SEC EDGAR REST ingest skeleton for 10-K / 10-Q RAG (T2.2).

Uses the public EDGAR ``data.sec.gov`` JSON APIs. Full-text download and HTML
stripping are stubbed for v1; callers index ``summary`` / ``description`` fields
into :class:`data.store.vector_store.VectorStore`.

SEC fair-access: set ``SEC_EDGAR_USER_AGENT`` (or pass ``user_agent=``) to
``"YourName your@email.com"``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

_EDGAR_BASE = "https://data.sec.gov"
_TICKER_MAP_URL = f"{_EDGAR_BASE}/files/company_tickers.json"
def _submissions_url(cik: str) -> str:
    return f"{_EDGAR_BASE}/submissions/CIK{int(cik):010d}.json"
_DEFAULT_FORMS = frozenset({"10-K", "10-Q", "8-K"})
_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class FilingRecord:
    """Normalized filing metadata from EDGAR submissions."""

    ticker: str
    cik: str
    form_type: str
    filed_date: date
    accession_number: str
    primary_document: str
    description: str = ""

    @property
    def accession_no_dashes(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def filing_index_url(self) -> str:
        """URL to the filing index page (full text fetch is a follow-up step)."""
        cik_int = int(self.cik)
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{self.accession_no_dashes}/{self.primary_document}"
        )


def _default_user_agent() -> str:
    return os.getenv("SEC_EDGAR_USER_AGENT", "MLCouncil research@example.com")


def _http_get_json(url: str, *, user_agent: str, timeout: float = 30.0) -> Any:
    req = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_ticker_cik_map(*, user_agent: Optional[str] = None) -> dict[str, str]:
    """Return ``{TICKER: zero_padded_cik}`` from EDGAR company_tickers.json."""
    ua = user_agent or _default_user_agent()
    payload = _http_get_json(_TICKER_MAP_URL, user_agent=ua)
    mapping: dict[str, str] = {}
    for entry in payload.values():
        ticker = str(entry.get("ticker", "")).upper()
        cik = str(entry.get("cik_str", entry.get("cik", "")))
        if ticker and cik:
            mapping[ticker] = cik.zfill(10)
    return mapping


def resolve_cik(ticker: str, *, cik_map: Optional[dict[str, str]] = None) -> Optional[str]:
    """Resolve ticker to 10-digit CIK."""
    key = ticker.upper().strip()
    if cik_map is None:
        cik_map = load_ticker_cik_map()
    return cik_map.get(key)


def list_filings(
    ticker: str,
    *,
    form_types: Optional[set[str]] = None,
    limit: int = 20,
    user_agent: Optional[str] = None,
    cik_map: Optional[dict[str, str]] = None,
) -> list[FilingRecord]:
    """List recent filings for *ticker* from EDGAR submissions API."""
    forms = form_types or _DEFAULT_FORMS
    cik = resolve_cik(ticker, cik_map=cik_map)
    if not cik:
        logger.warning("SEC EDGAR: unknown ticker {}", ticker)
        return []

    ua = user_agent or _default_user_agent()
    url = _submissions_url(cik)
    try:
        data = _http_get_json(url, user_agent=ua)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("SEC EDGAR submissions failed for {}: {}", ticker, exc)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms_list = recent.get("form", [])
    dates_list = recent.get("filingDate", [])
    accession_list = recent.get("accessionNumber", [])
    primary_list = recent.get("primaryDocument", [])
    desc_list = recent.get("primaryDocDescription", [])

    records: list[FilingRecord] = []
    for i, form in enumerate(forms_list):
        if form not in forms:
            continue
        try:
            filed = date.fromisoformat(str(dates_list[i]))
        except (ValueError, IndexError):
            continue
        try:
            accession = str(accession_list[i])
            primary = str(primary_list[i])
        except IndexError:
            continue
        desc = ""
        if i < len(desc_list):
            desc = str(desc_list[i] or "")
        records.append(
            FilingRecord(
                ticker=ticker.upper(),
                cik=cik,
                form_type=str(form),
                filed_date=filed,
                accession_number=accession,
                primary_document=primary,
                description=desc,
            )
        )
        if len(records) >= limit:
            break

    return records


def strip_html(text: str) -> str:
    """Remove HTML tags for RAG chunking (skeleton; no BeautifulSoup dep)."""
    cleaned = _HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def fetch_filing_text(url: str, *, user_agent: Optional[str] = None, timeout: float = 60.0) -> str:
    """Download filing document text (best-effort; returns empty on failure)."""
    ua = user_agent or _default_user_agent()
    req = Request(url, headers={"User-Agent": ua, "Accept": "text/html,text/plain"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning("SEC EDGAR document fetch failed {}: {}", url, exc)
        return ""
    return strip_html(raw)


def ingest_filings_to_vector_store(
    ticker: str,
    store: Any,
    *,
    form_types: Optional[set[str]] = None,
    limit: int = 5,
    user_agent: Optional[str] = None,
    fetch_body: bool = False,
) -> int:
    """Index filing descriptions (and optionally bodies) into *store*.

    Parameters
    ----------
    store:
        :class:`data.store.vector_store.VectorStore` instance.
    fetch_body:
        When True, attempt ``fetch_filing_text`` per filing (slow; network).
    """
    filings = list_filings(
        ticker,
        form_types=form_types,
        limit=limit,
        user_agent=user_agent,
    )
    total = 0
    for rec in filings:
        passages: list[str] = []
        if rec.description:
            passages.append(rec.description)
        if fetch_body:
            body = fetch_filing_text(rec.filing_index_url, user_agent=user_agent)
            if body:
                # Chunk ~2k chars for RAG (simple fixed windows)
                step = 2000
                passages.extend(body[i : i + step] for i in range(0, min(len(body), 20000), step))
        if not passages:
            passages.append(
                f"{rec.ticker} {rec.form_type} filed {rec.filed_date.isoformat()} "
                f"accession {rec.accession_number}"
            )
        n = store.upsert_passages(
            rec.ticker,
            rec.form_type,
            rec.filed_date.isoformat(),
            passages,
            extra_metadata={"accession": rec.accession_number},
        )
        total += n
    return total
