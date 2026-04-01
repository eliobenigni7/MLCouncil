"""Tests for data/ingest modules — all network calls are mocked."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cfg(tmp_path):
    """Write a minimal universe.yaml and return its path."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    cfg_data = {
        "universe": {"tickers": ["AAPL", "MSFT"]},
        "settings": {
            "data_dir": str(tmp_path / "data" / "raw"),
            "transaction_timezone": "America/New_York",
            "transaction_time_hour": 20,
            "transaction_time_minute": 30,
            "forward_fill_max_days": 2,
        },
        "macro": {
            "fred_series": {
                "vix": "VIXCLS",
                "treasury_10y": "DGS10",
                "treasury_2y": "DGS2",
                "sp500": "SP500",
            },
            "sp500_rolling_windows": [5, 10],
        },
    }
    (config_dir / "universe.yaml").write_text(yaml.dump(cfg_data))
    return tmp_path


def _make_ohlcv_pandas(n: int = 5, ticker: str = "AAPL") -> pd.DataFrame:
    """Synthetic daily OHLCV DataFrame as yfinance would return it."""
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": [150.0 + i for i in range(n)],
            "High": [155.0 + i for i in range(n)],
            "Low": [148.0 + i for i in range(n)],
            "Close": [152.0 + i for i in range(n)],
            "Adj Close": [152.0 + i for i in range(n)],
            "Volume": [1_000_000 + i * 1000 for i in range(n)],
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


# ---------------------------------------------------------------------------
# market_data tests
# ---------------------------------------------------------------------------

class TestMarketData:
    def _patch_config(self, cfg_root):
        """Context manager: redirect config load to tmp config."""
        import data.ingest.market_data as md
        orig = md._ROOT
        md._ROOT = cfg_root
        yield
        md._ROOT = orig

    def test_download_daily_schema(self, cfg, monkeypatch):
        import data.ingest.market_data as md

        monkeypatch.setattr(md, "_ROOT", cfg)

        fake_df = _make_ohlcv_pandas(n=1)
        with patch("yfinance.download", return_value=fake_df):
            result = md.download_daily(tickers=["AAPL"], date="2024-01-02")

        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()

        expected_cols = {"ticker", "valid_time", "transaction_time",
                         "open", "high", "low", "close", "adj_close", "volume"}
        assert expected_cols == set(result.columns)

    def test_download_daily_column_types(self, cfg, monkeypatch):
        import data.ingest.market_data as md

        monkeypatch.setattr(md, "_ROOT", cfg)
        fake_df = _make_ohlcv_pandas(n=1)

        with patch("yfinance.download", return_value=fake_df):
            result = md.download_daily(tickers=["AAPL"], date="2024-01-02")

        assert result["ticker"].dtype == pl.Utf8
        assert result["valid_time"].dtype == pl.Date
        assert result["transaction_time"].dtype == pl.Datetime
        assert result["close"].dtype == pl.Float64
        assert result["volume"].dtype == pl.Int64

    def test_download_daily_bitemporal_not_null(self, cfg, monkeypatch):
        import data.ingest.market_data as md

        monkeypatch.setattr(md, "_ROOT", cfg)
        fake_df = _make_ohlcv_pandas(n=3)

        with patch("yfinance.download", return_value=fake_df):
            result = md.download_daily(tickers=["AAPL"], date="2024-01-02")

        assert result["valid_time"].null_count() == 0
        assert result["transaction_time"].null_count() == 0

    def test_download_daily_empty_on_missing_ticker(self, cfg, monkeypatch):
        import data.ingest.market_data as md

        monkeypatch.setattr(md, "_ROOT", cfg)
        empty_df = pd.DataFrame()

        with patch("yfinance.download", return_value=empty_df):
            result = md.download_daily(tickers=["FAKE999"], date="2024-01-02")

        assert result.is_empty()

    def test_download_daily_no_nan_on_key_columns(self, cfg, monkeypatch):
        import data.ingest.market_data as md

        monkeypatch.setattr(md, "_ROOT", cfg)
        fake_df = _make_ohlcv_pandas(n=5)
        # Inject one NaN in Close — should be forward-filled
        fake_df.loc[fake_df.index[2], "Close"] = float("nan")
        fake_df.loc[fake_df.index[2], "Adj Close"] = float("nan")

        with patch("yfinance.download", return_value=fake_df):
            result = md.download_daily(tickers=["AAPL"], date="2024-01-02")

        # The previously null row should be filled (or dropped if ffill still null)
        assert result.filter(pl.col("close").is_null()).is_empty()

    def test_download_universe_saves_parquet(self, cfg, monkeypatch, tmp_path):
        import data.ingest.market_data as md

        monkeypatch.setattr(md, "_ROOT", cfg)
        fake_df = _make_ohlcv_pandas(n=5)

        with patch("yfinance.download", return_value=fake_df):
            md.download_universe(
                tickers=["AAPL"],
                start="2024-01-01",
                end="2024-01-10",
                data_dir=cfg / "data" / "raw",
            )

        parquet_files = list((cfg / "data" / "raw" / "ohlcv" / "AAPL").glob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_download_daily_multiple_tickers(self, cfg, monkeypatch):
        import data.ingest.market_data as md

        monkeypatch.setattr(md, "_ROOT", cfg)

        call_count = 0

        def fake_download(ticker, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_ohlcv_pandas(n=1, ticker=ticker)

        with patch("yfinance.download", side_effect=fake_download):
            result = md.download_daily(tickers=["AAPL", "MSFT"], date="2024-01-02")

        assert call_count == 2
        assert len(result) == 2
        assert set(result["ticker"].to_list()) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# fundamentals tests
# ---------------------------------------------------------------------------

class TestFundamentals:
    def _make_balance_sheet(self) -> pd.DataFrame:
        dates = pd.to_datetime(["2023-09-30", "2022-09-30", "2021-09-30"])
        return pd.DataFrame(
            {
                dates[0]: [1e11, 5e10, 2e10],
                dates[1]: [9e10, 4e10, 1.5e10],
                dates[2]: [8e10, 3.5e10, 1e10],
            },
            index=["Total Assets", "Total Liabilities", "Cash"],
        )

    def test_fundamentals_schema(self, cfg, monkeypatch):
        import data.ingest.fundamentals as fd

        monkeypatch.setattr(fd, "_ROOT", cfg)

        mock_ticker = MagicMock()
        mock_ticker.balance_sheet = self._make_balance_sheet()
        mock_ticker.financials = self._make_balance_sheet()
        mock_ticker.cashflow = self._make_balance_sheet()
        mock_ticker.earnings_dates = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fd.download_fundamentals(
                tickers=["AAPL"],
                data_dir=cfg / "data" / "raw",
            )

        out_dir = cfg / "data" / "raw" / "fundamentals" / "AAPL"
        for name in ("balance_sheet", "income_statement", "cash_flow"):
            parquet = out_dir / f"{name}.parquet"
            assert parquet.exists(), f"Missing {name}.parquet"
            df = pl.read_parquet(parquet)
            assert "ticker" in df.columns
            assert "valid_time" in df.columns
            assert "transaction_time" in df.columns
            assert "statement" in df.columns

    def test_fundamentals_bitemporal_not_null(self, cfg, monkeypatch):
        import data.ingest.fundamentals as fd

        monkeypatch.setattr(fd, "_ROOT", cfg)

        mock_ticker = MagicMock()
        mock_ticker.balance_sheet = self._make_balance_sheet()
        mock_ticker.financials = pd.DataFrame()
        mock_ticker.cashflow = pd.DataFrame()
        mock_ticker.earnings_dates = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fd.download_fundamentals(
                tickers=["AAPL"],
                data_dir=cfg / "data" / "raw",
            )

        df = pl.read_parquet(cfg / "data" / "raw" / "fundamentals" / "AAPL" / "balance_sheet.parquet")
        assert df["ticker"].null_count() == 0
        assert df["valid_time"].null_count() == 0
        assert df["transaction_time"].null_count() == 0

    def test_fundamentals_ticker_column(self, cfg, monkeypatch):
        import data.ingest.fundamentals as fd

        monkeypatch.setattr(fd, "_ROOT", cfg)

        mock_ticker = MagicMock()
        mock_ticker.balance_sheet = self._make_balance_sheet()
        mock_ticker.financials = pd.DataFrame()
        mock_ticker.cashflow = pd.DataFrame()
        mock_ticker.earnings_dates = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fd.download_fundamentals(
                tickers=["MSFT"],
                data_dir=cfg / "data" / "raw",
            )

        df = pl.read_parquet(cfg / "data" / "raw" / "fundamentals" / "MSFT" / "balance_sheet.parquet")
        assert (df["ticker"] == "MSFT").all()


# ---------------------------------------------------------------------------
# news tests
# ---------------------------------------------------------------------------

class TestNews:
    def _make_feed(self, tickers: list[str]) -> MagicMock:
        """Synthetic feedparser result."""
        entries = []
        import time as _time

        for t in tickers:
            for i in range(3):
                entry = SimpleNamespace(
                    title=f"{t} headline {i}",
                    published_parsed=_time.strptime("2024-01-15 10:00:00", "%Y-%m-%d %H:%M:%S"),
                    source={"title": "Reuters"},
                    link=f"https://finance.yahoo.com/{t}/news/{i}",
                )
                entries.append(entry)

        mock_feed = MagicMock()
        mock_feed.entries = entries
        mock_feed.bozo = False
        return mock_feed

    def test_news_schema(self, cfg, monkeypatch):
        import data.ingest.news as nw

        monkeypatch.setattr(nw, "_ROOT", cfg)

        with patch("feedparser.parse", return_value=self._make_feed(["AAPL"])):
            result = nw.download_news(
                tickers=["AAPL"],
                date="2024-01-15",
                data_dir=cfg / "data" / "raw",
            )

        assert isinstance(result, pl.DataFrame)
        expected_cols = {"ticker", "valid_time", "transaction_time", "title", "published", "source", "url"}
        assert expected_cols == set(result.columns)

    def test_news_bitemporal_not_null(self, cfg, monkeypatch):
        import data.ingest.news as nw

        monkeypatch.setattr(nw, "_ROOT", cfg)

        with patch("feedparser.parse", return_value=self._make_feed(["AAPL"])):
            result = nw.download_news(
                tickers=["AAPL"],
                date="2024-01-15",
                data_dir=cfg / "data" / "raw",
            )

        assert result["valid_time"].null_count() == 0
        assert result["transaction_time"].null_count() == 0

    def test_news_deduplication(self, cfg, monkeypatch):
        import data.ingest.news as nw

        monkeypatch.setattr(nw, "_ROOT", cfg)

        # Same feed returned twice (simulating same URL from two calls)
        with patch("feedparser.parse", return_value=self._make_feed(["AAPL"])):
            result = nw.download_news(
                tickers=["AAPL", "AAPL"],  # duplicate ticker to force duplicate URLs
                date="2024-01-15",
                data_dir=cfg / "data" / "raw",
            )

        # All URLs should be unique
        assert result["url"].n_unique() == len(result)

    def test_news_saved_parquet(self, cfg, monkeypatch):
        import data.ingest.news as nw

        monkeypatch.setattr(nw, "_ROOT", cfg)

        with patch("feedparser.parse", return_value=self._make_feed(["AAPL", "MSFT"])):
            nw.download_news(
                tickers=["AAPL", "MSFT"],
                date="2024-01-15",
                data_dir=cfg / "data" / "raw",
            )

        parquet = cfg / "data" / "raw" / "news" / "2024-01-15.parquet"
        assert parquet.exists()

    def test_news_column_types(self, cfg, monkeypatch):
        import data.ingest.news as nw

        monkeypatch.setattr(nw, "_ROOT", cfg)

        with patch("feedparser.parse", return_value=self._make_feed(["AAPL"])):
            result = nw.download_news(
                tickers=["AAPL"],
                date="2024-01-15",
                data_dir=cfg / "data" / "raw",
            )

        assert result["ticker"].dtype == pl.Utf8
        assert result["valid_time"].dtype == pl.Date
        assert result["transaction_time"].dtype == pl.Datetime
        assert result["title"].dtype == pl.Utf8
        assert result["url"].dtype == pl.Utf8


# ---------------------------------------------------------------------------
# macro tests
# ---------------------------------------------------------------------------

class TestMacro:
    _VIX_CSV = "DATE,VIXCLS\n2024-01-02,13.40\n2024-01-03,12.90\n2024-01-04,13.10\n"
    _T10_CSV = "DATE,DGS10\n2024-01-02,3.95\n2024-01-03,3.97\n2024-01-04,3.93\n"
    _T2_CSV = "DATE,DGS2\n2024-01-02,4.32\n2024-01-03,4.30\n2024-01-04,4.28\n"
    _SP_CSV = (
        "DATE,SP500\n"
        "2023-12-29,4769.83\n2024-01-02,4742.83\n2024-01-03,4704.81\n"
        "2024-01-04,4697.24\n2024-01-05,4688.68\n"
    )

    def _make_response(self, text: str) -> MagicMock:
        resp = MagicMock()
        resp.text = text
        resp.raise_for_status = MagicMock()
        return resp

    def test_macro_vix_saved(self, cfg, monkeypatch):
        import data.ingest.macro as mc

        monkeypatch.setattr(mc, "_ROOT", cfg)

        responses = [
            self._make_response(self._VIX_CSV),
            self._make_response(self._T10_CSV),
            self._make_response(self._T2_CSV),
            self._make_response(self._SP_CSV),
        ]

        with patch("requests.get", side_effect=responses):
            mc.download_macro(
                start="2024-01-01",
                end="2024-01-05",
                data_dir=cfg / "data" / "raw",
            )

        vix_path = cfg / "data" / "raw" / "macro" / "vix.parquet"
        assert vix_path.exists()
        df = pl.read_parquet(vix_path)
        assert "valid_time" in df.columns
        assert "transaction_time" in df.columns
        assert "value" in df.columns

    def test_macro_treasuries_spread(self, cfg, monkeypatch):
        import data.ingest.macro as mc

        monkeypatch.setattr(mc, "_ROOT", cfg)

        responses = [
            self._make_response(self._VIX_CSV),
            self._make_response(self._T10_CSV),
            self._make_response(self._T2_CSV),
            self._make_response(self._SP_CSV),
        ]

        with patch("requests.get", side_effect=responses):
            mc.download_macro(
                start="2024-01-01",
                end="2024-01-05",
                data_dir=cfg / "data" / "raw",
            )

        treas_path = cfg / "data" / "raw" / "macro" / "treasuries.parquet"
        assert treas_path.exists()
        df = pl.read_parquet(treas_path)
        assert "dgs10" in df.columns
        assert "dgs2" in df.columns
        assert "yield_spread" in df.columns
        assert "transaction_time" in df.columns

        # Spread = 10Y - 2Y (should be negative in 2024 inverted curve)
        spread = df["yield_spread"][0]
        assert abs(spread - (df["dgs10"][0] - df["dgs2"][0])) < 1e-9

    def test_macro_sp500_rolling_returns(self, cfg, monkeypatch):
        import data.ingest.macro as mc

        monkeypatch.setattr(mc, "_ROOT", cfg)

        responses = [
            self._make_response(self._VIX_CSV),
            self._make_response(self._T10_CSV),
            self._make_response(self._T2_CSV),
            self._make_response(self._SP_CSV),
        ]

        with patch("requests.get", side_effect=responses):
            mc.download_macro(
                start="2023-12-01",
                end="2024-01-05",
                data_dir=cfg / "data" / "raw",
            )

        sp_path = cfg / "data" / "raw" / "macro" / "sp500.parquet"
        assert sp_path.exists()
        df = pl.read_parquet(sp_path)
        assert "sp500_price" in df.columns
        assert "daily_return" in df.columns
        # Config sets rolling windows [5, 10]
        assert "return_5d" in df.columns
        assert "return_10d" in df.columns
        assert "transaction_time" in df.columns

    def test_macro_bitemporal_not_null(self, cfg, monkeypatch):
        import data.ingest.macro as mc

        monkeypatch.setattr(mc, "_ROOT", cfg)

        responses = [
            self._make_response(self._VIX_CSV),
            self._make_response(self._T10_CSV),
            self._make_response(self._T2_CSV),
            self._make_response(self._SP_CSV),
        ]

        with patch("requests.get", side_effect=responses):
            mc.download_macro(
                start="2024-01-01",
                end="2024-01-05",
                data_dir=cfg / "data" / "raw",
            )

        for fname in ("vix.parquet", "treasuries.parquet"):
            df = pl.read_parquet(cfg / "data" / "raw" / "macro" / fname)
            assert df["valid_time"].null_count() == 0
            assert df["transaction_time"].null_count() == 0
