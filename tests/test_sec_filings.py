"""Tests for data/ingest/sec_filings.py — mocked HTTP."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from data.ingest import sec_filings as sf
from data.store.vector_store import VectorStore


_TICKER_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}

_SUBMISSIONS_JSON = {
    "filings": {
        "recent": {
            "form": ["10-K", "8-K", "10-Q"],
            "filingDate": ["2024-02-01", "2024-01-15", "2023-11-01"],
            "accessionNumber": [
                "0000320193-24-000001",
                "0000320193-24-000002",
                "0000320193-23-000099",
            ],
            "primaryDocument": ["aapl-20240201.htm", "aapl-8k.htm", "aapl-10q.htm"],
            "primaryDocDescription": ["Annual report", "Current report", "Quarterly"],
        }
    }
}


def test_load_ticker_cik_map():
    def fake_get(url, *, user_agent, timeout=30.0):
        return _TICKER_JSON

    with patch.object(sf, "_http_get_json", side_effect=fake_get):
        m = sf.load_ticker_cik_map(user_agent="Test test@example.com")
    assert m["AAPL"] == "0000320193"


def test_list_filings_filters_forms():
    cik_map = {"AAPL": "0000320193"}

    def fake_get(url, *, user_agent, timeout=30.0):
        return _SUBMISSIONS_JSON

    with patch.object(sf, "_http_get_json", side_effect=fake_get):
        records = sf.list_filings(
            "AAPL",
            form_types={"10-K", "10-Q"},
            limit=10,
            cik_map=cik_map,
            user_agent="Test test@example.com",
        )
    assert len(records) == 2
    assert all(r.form_type in {"10-K", "10-Q"} for r in records)
    assert records[0].filed_date == date(2024, 2, 1)
    assert "sec.gov" in records[0].filing_index_url


def test_list_filings_unknown_ticker():
    with patch.object(sf, "load_ticker_cik_map", return_value={}):
        assert sf.list_filings("UNKNOWN") == []


def test_strip_html():
    assert sf.strip_html("<p>Revenue <b>up</b></p>") == "Revenue up"


def test_fetch_filing_text_empty_on_error():
    with patch("urllib.request.urlopen", side_effect=OSError("network")):
        assert sf.fetch_filing_text("https://www.sec.gov/example") == ""


def test_ingest_filings_to_vector_store():
    store = VectorStore(force_mock=True)
    rec = sf.FilingRecord(
        ticker="AAPL",
        cik="0000320193",
        form_type="10-K",
        filed_date=date(2024, 2, 1),
        accession_number="0000320193-24-000001",
        primary_document="aapl.htm",
        description="Annual report narrative",
    )
    with patch.object(sf, "list_filings", return_value=[rec]):
        n = sf.ingest_filings_to_vector_store("AAPL", store, fetch_body=False)
    assert n >= 1
    assert store.count() >= 1
