"""Tests for scripts/backfill_fill_log.py."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import sys
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import backfill_fill_log as bf  # noqa: E402
from execution import fill_log as fl_module
from execution.fill_log import read_fills


def _write_log(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records))


@pytest.fixture
def paper_trades_dir(tmp_path: Path) -> Path:
    d = tmp_path / "paper_trades"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def isolate_fill_log(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(fl_module, "FILL_LOG_DIR", tmp_path / "fills")
    yield


def test_skips_records_without_fill_price(paper_trades_dir: Path):
    _write_log(
        paper_trades_dir / "2026-05-21.json",
        [
            {"order_id": "o1", "symbol": "AAPL", "qty": 10, "side": "buy",
             "submitted_at": "2026-05-21T14:30:00+00:00"},
        ],
    )
    summary = bf.backfill(paper_trades_dir=paper_trades_dir)
    assert summary["records_in"] == 1
    assert summary["records_out"] == 0
    assert summary["skipped"] == 1


def test_emits_record_when_filled_avg_price_present(paper_trades_dir: Path):
    _write_log(
        paper_trades_dir / "2026-05-21.json",
        [
            {
                "order_id": "o1",
                "symbol": "AAPL",
                "qty": 10,
                "side": "buy",
                "filled_avg_price": 201.5,
                "submitted_at": "2026-05-21T14:30:00+00:00",
                "filled_at": "2026-05-21T14:30:15+00:00",
            },
        ],
    )
    summary = bf.backfill(paper_trades_dir=paper_trades_dir)
    assert summary["records_out"] == 1

    df = read_fills()
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["ticker"] == "AAPL"
    assert row["fill_price"] == pytest.approx(201.5)
    # decision_price falls back to fill_price → 0 IS
    assert row["decision_price"] == pytest.approx(201.5)


def test_include_submissions_uses_limit_price_as_fill(paper_trades_dir: Path):
    _write_log(
        paper_trades_dir / "2026-05-21.json",
        [
            {
                "order_id": "o2",
                "symbol": "MSFT",
                "qty": 5,
                "side": "sell",
                "limit_price": 420.0,
                "submitted_at": "2026-05-21T15:00:00+00:00",
            },
        ],
    )
    summary = bf.backfill(paper_trades_dir=paper_trades_dir, include_submissions=True)
    assert summary["records_out"] == 1
    df = read_fills()
    assert df["fill_price"][0] == pytest.approx(420.0)


def test_dry_run_does_not_write(paper_trades_dir: Path, tmp_path: Path):
    _write_log(
        paper_trades_dir / "2026-05-21.json",
        [
            {
                "order_id": "o3",
                "symbol": "NVDA",
                "qty": 1,
                "side": "buy",
                "filled_avg_price": 900.0,
                "submitted_at": "2026-05-21T16:00:00+00:00",
            },
        ],
    )
    summary = bf.backfill(paper_trades_dir=paper_trades_dir, dry_run=True)
    assert summary["records_out"] == 1
    df = read_fills()
    assert df.height == 0  # nothing written


def test_missing_dir_returns_zeros(tmp_path: Path):
    summary = bf.backfill(paper_trades_dir=tmp_path / "does-not-exist")
    assert summary == {"files": 0, "records_in": 0, "records_out": 0, "skipped": 0}


def test_idempotent_across_runs(paper_trades_dir: Path):
    _write_log(
        paper_trades_dir / "2026-05-21.json",
        [
            {
                "order_id": "o4",
                "symbol": "GOOGL",
                "qty": 7,
                "side": "buy",
                "filled_avg_price": 150.5,
                "submitted_at": "2026-05-21T17:00:00+00:00",
            },
        ],
    )
    bf.backfill(paper_trades_dir=paper_trades_dir)
    bf.backfill(paper_trades_dir=paper_trades_dir)
    df = read_fills()
    # de-dup keeps a single row even after a second pass
    assert df.height == 1


def test_cross_month_writes_separate_partitions(paper_trades_dir: Path, tmp_path: Path):
    _write_log(
        paper_trades_dir / "2026-05-30.json",
        [
            {"order_id": "o5", "symbol": "AAPL", "qty": 1, "side": "buy",
             "filled_avg_price": 200.0,
             "submitted_at": "2026-05-30T15:00:00+00:00"},
        ],
    )
    _write_log(
        paper_trades_dir / "2026-06-01.json",
        [
            {"order_id": "o6", "symbol": "MSFT", "qty": 1, "side": "sell",
             "filled_avg_price": 420.0,
             "submitted_at": "2026-06-01T15:00:00+00:00"},
        ],
    )
    bf.backfill(paper_trades_dir=paper_trades_dir)
    fills_dir = tmp_path / "fills"
    assert (fills_dir / "2026-05.parquet").exists()
    assert (fills_dir / "2026-06.parquet").exists()
