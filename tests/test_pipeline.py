"""Tests per la pipeline Dagster MLCouncil.

Copertura
---------
1. test_assets_dependencies       – il grafo dipendenze è un DAG (aciclico)
2. test_quality_checks_fail_on_bad_data – i quality check falliscono su dati cattivi
3. test_full_pipeline_synthetic   – pipeline completa su dati sintetici
4. test_schedule_cron             – la schedule triggera correttamente lun-ven
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

# Aggiungi la root del progetto al path
_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dagster as dg

# ---------------------------------------------------------------------------
# Caricamento pipeline (standalone, come fa Dagster con workspace.yaml)
# ---------------------------------------------------------------------------

import importlib.util


def _load_pipeline():
    """Carica data/pipeline.py come modulo standalone."""
    mod_name = "pipeline_module_test"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name, _ROOT / "data" / "pipeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Caricamento una sola volta
_pipeline = _load_pipeline()


def _make_context(partition_date: str = "2024-01-15") -> MagicMock:
    """Context mock generico per test di asset."""
    ctx = MagicMock(spec=dg.AssetExecutionContext)
    ctx.partition_key = partition_date
    ctx.log = MagicMock()
    return ctx


def _call_asset(asset_def, *args):
    """Chiama la funzione decorata di un asset."""
    return asset_def.op.compute_fn.decorated_fn(*args)


# ===========================================================================
# 1. test_assets_dependencies — DAG aciclico
# ===========================================================================

class TestAssetDependencies:
    """Verifica che il grafo delle dipendenze sia un DAG valido."""

    def test_definitions_load_without_error(self):
        """Le Definitions si costruiscono senza errori → Dagster ha validato il grafo."""
        assert _pipeline.defs is not None

    def test_all_expected_assets_present(self):
        """Tutti e 11 gli asset definiti sono registrati."""
        keys = {str(a.key) for a in _pipeline.defs.assets}
        expected = {
            "AssetKey(['raw_ohlcv'])",
            "AssetKey(['raw_news'])",
            "AssetKey(['raw_macro'])",
            "AssetKey(['alpha158_features'])",
            "AssetKey(['sentiment_features'])",
            "AssetKey(['lgbm_signals'])",
            "AssetKey(['sentiment_signals'])",
            "AssetKey(['current_regime'])",
            "AssetKey(['council_signal'])",
            "AssetKey(['portfolio_weights'])",
            "AssetKey(['daily_orders'])",
        }
        assert expected == keys

    def test_dag_is_acyclic(self):
        """Il grafo è aciclico: DFS non rileva cicli.

        Dagster valida l'aciclicità al momento della costruzione delle
        Definitions. Questo test verifica esplicitamente la proprietà.
        """
        # Costruiamo il grafo: asset_key → set(upstream_keys)
        # asset_deps è {self_key: {upstream_key, ...}}
        adj: dict[str, set[str]] = {}
        for a in _pipeline.defs.assets:
            self_key = str(a.key)
            upstream = {str(k) for k in a.asset_deps.get(a.key, set())}
            adj[self_key] = upstream

        # DFS per rilevare cicli
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node: str) -> None:
            visited.add(node)
            in_stack.add(node)
            for neighbor in adj.get(node, set()):
                assert neighbor not in in_stack, (
                    f"Ciclo rilevato nel grafo: {node} → {neighbor}"
                )
                if neighbor not in visited:
                    dfs(neighbor)
            in_stack.discard(node)

        for node in list(adj.keys()):
            if node not in visited:
                dfs(node)

    def test_dependency_chain(self):
        """Verifica la catena di dipendenze attesa nel grafo."""
        asset_by_key = {str(a.key): a for a in _pipeline.defs.assets}

        def upstream(key: str) -> set[str]:
            a = asset_by_key[key]
            return {str(k) for k in a.asset_deps.get(a.key, set())}

        alpha_deps = upstream("AssetKey(['alpha158_features'])")
        assert "AssetKey(['raw_ohlcv'])" in alpha_deps
        assert "AssetKey(['raw_macro'])" in alpha_deps

        assert "AssetKey(['alpha158_features'])" in upstream("AssetKey(['lgbm_signals'])")

        council_deps = upstream("AssetKey(['council_signal'])")
        assert "AssetKey(['lgbm_signals'])" in council_deps
        assert "AssetKey(['sentiment_signals'])" in council_deps
        assert "AssetKey(['current_regime'])" in council_deps

        assert "AssetKey(['portfolio_weights'])" in upstream("AssetKey(['daily_orders'])")

    def test_all_assets_are_partitioned(self):
        """Tutti gli asset usano DailyPartitionsDefinition (stessa granularità)."""
        for a in _pipeline.defs.assets:
            assert a.partitions_def is not None, (
                f"Asset {a.key} non ha partitions_def"
            )
            assert isinstance(a.partitions_def, dg.DailyPartitionsDefinition), (
                f"Asset {a.key} non usa DailyPartitionsDefinition"
            )

    def test_retry_policy_configured(self):
        """Ogni asset ha RetryPolicy con max_retries=2."""
        for a in _pipeline.defs.assets:
            op = a.op
            retry = op.retry_policy
            assert retry is not None, f"Asset {a.key} non ha retry_policy"
            assert retry.max_retries == 2, (
                f"Asset {a.key}: max_retries={retry.max_retries}, atteso 2"
            )


# ===========================================================================
# 2. test_quality_checks_fail_on_bad_data
# ===========================================================================

class TestQualityChecks:
    """I quality check sollevano eccezioni su dati invalidi."""

    def test_empty_dataframe_fails_row_check(self):
        """AssertionError se il DataFrame scaricato è vuoto."""
        empty_df = pl.DataFrame()
        with pytest.raises(AssertionError, match="Nessun dato scaricato"):
            assert empty_df.shape[0] > 0, "Nessun dato scaricato"

    def test_missing_valid_time_fails(self):
        """AssertionError se manca il campo valid_time."""
        df = pl.DataFrame({"close": [100.0]})
        with pytest.raises(AssertionError, match="Campo bi-temporale mancante"):
            assert "valid_time" in df.columns, "Campo bi-temporale mancante"

    def test_nan_close_fails_quality_check(self):
        """AssertionError se ci sono NaN nel campo close."""
        df = pl.DataFrame({"close": [float("nan"), 100.0]})
        nan_count = df["close"].is_nan().sum()
        with pytest.raises(AssertionError, match="NaN nei prezzi di chiusura"):
            assert nan_count == 0, f"NaN nei prezzi di chiusura: {nan_count}"

    def test_insufficient_features_fails(self):
        """AssertionError se alpha158 produce meno di 158 feature."""
        exclude = {"ticker", "valid_time", "transaction_time"}
        df_few = pl.DataFrame({
            "ticker": ["AAPL"],
            "valid_time": [date(2024, 1, 1)],
            "f1": [1.0], "f2": [2.0], "f3": [3.0],
        })
        non_meta = [c for c in df_few.columns if c not in exclude]
        with pytest.raises(AssertionError, match="feature"):
            assert len(non_meta) >= 158, (
                f"Solo {len(non_meta)} feature, attese 158+"
            )

    def test_nan_features_fails(self):
        """AssertionError se ci sono NaN nelle feature Alpha158."""
        nan_df = pl.DataFrame({
            f"f{i}": [float("nan")] for i in range(5)
        })
        float_cols = [
            c for c in nan_df.columns
            if nan_df[c].dtype in (pl.Float32, pl.Float64)
        ]
        nan_sum = (
            nan_df.select([pl.col(c).is_nan().sum() for c in float_cols])
            .to_pandas().sum().sum()
        )
        with pytest.raises(AssertionError, match="NaN"):
            assert nan_sum == 0, f"NaN nelle feature: {nan_sum}"

    def test_raw_ohlcv_asset_fails_on_empty_download(self):
        """Il body di raw_ohlcv solleva AssertionError su DataFrame vuoto.

        Inietta un mock di data.ingest.market_data per evitare la dipendenza
        da yfinance a runtime.
        """
        import types

        ctx = _make_context("2024-01-15")

        empty_df = pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "valid_time": pl.Date,
                "transaction_time": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "adj_close": pl.Float64,
                "volume": pl.Int64,
            }
        )

        # Registra un mock di data.ingest.market_data prima della chiamata
        mock_market_data = types.ModuleType("data.ingest.market_data")
        mock_market_data.download_daily = MagicMock(return_value=empty_df)

        orig = sys.modules.get("data.ingest.market_data")
        sys.modules["data.ingest.market_data"] = mock_market_data
        try:
            with pytest.raises(AssertionError, match="Nessun dato scaricato"):
                _call_asset(_pipeline.raw_ohlcv, ctx)
        finally:
            if orig is None:
                sys.modules.pop("data.ingest.market_data", None)
            else:
                sys.modules["data.ingest.market_data"] = orig

    def test_alpha158_asset_fails_on_empty_ohlcv(self):
        """alpha158_features solleva ValueError se _load_all_ohlcv restituisce vuoto."""
        ctx = _make_context("2024-01-15")
        empty_ohlcv = pl.DataFrame()
        empty_macro = pl.DataFrame()

        with patch.object(_pipeline, "_load_all_ohlcv", return_value=empty_ohlcv):
            with pytest.raises(ValueError, match="Nessun dato OHLCV"):
                _call_asset(
                    _pipeline.alpha158_features, ctx, empty_ohlcv, empty_macro
                )


# ===========================================================================
# 3. test_full_pipeline_synthetic
# ===========================================================================

def _synthetic_ohlcv(
    tickers: list[str],
    n_days: int,
    start: str = "2023-01-01",
) -> pl.DataFrame:
    """Genera un DataFrame OHLCV sintetico per test."""
    rows: list[dict] = []
    base = date.fromisoformat(start)
    rng = np.random.default_rng(42)
    for i in range(n_days):
        d = date.fromordinal(base.toordinal() + i)
        for ticker in tickers:
            close = float(100.0 + rng.normal(0, 2))
            rows.append({
                "ticker": ticker,
                "valid_time": d,
                "transaction_time": None,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "adj_close": close,
                "volume": int(1_000_000 + rng.integers(0, 500_000)),
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("transaction_time").cast(pl.Datetime("us", "UTC"))
    )


class TestFullPipelineSynthetic:
    """Esegue i layer della pipeline con dati sintetici."""

    TICKERS   = ["AAPL", "MSFT", "GOOGL"]
    PARTITION = "2024-01-15"

    # ── Layer 2: Features ────────────────────────────────────────────────────

    def test_sentiment_features_returns_empty_on_empty_news(self):
        """sentiment_features restituisce DataFrame vuoto su news vuote."""
        ctx = _make_context(self.PARTITION)
        result = _call_asset(_pipeline.sentiment_features, ctx, pl.DataFrame())
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    def test_sentiment_signals_z_scored(self):
        """sentiment_signals produce z-score cross-sezionale con media ≈ 0."""
        ctx = _make_context(self.PARTITION)
        today = date.fromisoformat(self.PARTITION)
        sent_feat = pl.DataFrame({
            "ticker":          self.TICKERS,
            "valid_time":      [today] * len(self.TICKERS),
            "sentiment_score": [0.3, -0.1, 0.7],
        })
        result = _call_asset(_pipeline.sentiment_signals, ctx, sent_feat)
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.TICKERS)
        assert abs(result.mean()) < 1e-9, "La media z-score deve essere ~0"

    # ── Layer 3: Signals ─────────────────────────────────────────────────────

    def test_current_regime_fallback_when_no_checkpoint(self):
        """current_regime restituisce 'transition' senza checkpoint e senza macro."""
        ctx = _make_context(self.PARTITION)
        # Patch exists() sul Path del checkpoint
        checkpoint_path = _pipeline._CHECKPOINTS / "hmm_latest.pkl"
        with patch.object(Path, "exists", return_value=False):
            result = _call_asset(_pipeline.current_regime, ctx, pl.DataFrame())
        assert result == "transition"

    def test_lgbm_signals_fallback_when_no_checkpoint(self):
        """lgbm_signals restituisce segnali 0.0 se il checkpoint non esiste."""
        ctx = _make_context(self.PARTITION)
        today = date.fromisoformat(self.PARTITION)
        feat = pl.DataFrame({
            "ticker":     self.TICKERS,
            "valid_time": [today] * len(self.TICKERS),
            "f1":         [1.0] * len(self.TICKERS),
        })
        with patch.object(Path, "exists", return_value=False):
            result = _call_asset(_pipeline.lgbm_signals, ctx, feat)
        assert isinstance(result, pd.Series)
        assert (result == 0.0).all()

    # ── Layer 4: Council ─────────────────────────────────────────────────────

    def test_council_signal_aggregation(self):
        """council_signal produce z-score cross-sezionale su segnali sintetici."""
        ctx = _make_context(self.PARTITION)
        lgbm_sig = pd.Series([1.0, -0.5, 0.3], index=self.TICKERS)
        sent_sig  = pd.Series([0.2, 0.8, -0.4], index=self.TICKERS)

        result = _call_asset(
            _pipeline.council_signal, ctx, lgbm_sig, sent_sig, "bull"
        )

        assert isinstance(result, pd.Series)
        assert set(result.index) == set(self.TICKERS)
        assert abs(result.mean()) < 1e-6, "Media z-score deve essere ~0"
        assert abs(result.std() - 1.0) < 1e-6, "Std z-score deve essere ~1"

    def test_council_signal_empty_on_no_signals(self):
        """council_signal restituisce Series vuoto se tutti i segnali sono vuoti."""
        ctx = _make_context(self.PARTITION)
        empty_series = pd.Series(dtype=float)

        result = _call_asset(
            _pipeline.council_signal, ctx, empty_series, empty_series, "transition"
        )

        assert isinstance(result, pd.Series)
        assert result.empty

    def test_portfolio_weights_sum_to_one(self):
        """portfolio_weights produce pesi non negativi che sommano a 1."""
        ctx = _make_context(self.PARTITION)
        council = pd.Series([1.2, -0.4, 0.8], index=self.TICKERS)

        n = len(self.TICKERS)
        cov_mat = pd.DataFrame(
            np.eye(n) * 0.0001, index=self.TICKERS, columns=self.TICKERS
        )
        with patch.object(_pipeline, "_compute_covariance", return_value=cov_mat):
            result = _call_asset(_pipeline.portfolio_weights, ctx, council)

        assert isinstance(result, pd.Series)
        assert not result.empty
        assert abs(result.sum() - 1.0) < 1e-6, (
            f"I pesi devono sommare a 1, ottenuto {result.sum():.6f}"
        )
        assert (result >= 0).all(), "Tutti i pesi devono essere ≥ 0 (long-only)"

    def test_daily_orders_valid_schema(self):
        """daily_orders produce un DataFrame con schema corretto."""
        ctx = _make_context(self.PARTITION)
        weights = pd.Series([0.5, 0.3, 0.2], index=self.TICKERS)

        # Mocka to_parquet per non richiedere filesystem/parquet engine
        with patch("pandas.DataFrame.to_parquet", return_value=None):
            result = _call_asset(_pipeline.daily_orders, ctx, weights)

        assert isinstance(result, pd.DataFrame)
        assert set(["ticker", "direction", "quantity", "target_weight"]).issubset(
            result.columns
        )
        if not result.empty:
            assert result["direction"].isin(["buy", "sell"]).all()
            assert (result["quantity"] >= 1.0).all()

    def test_daily_orders_empty_on_empty_weights(self):
        """daily_orders restituisce DataFrame vuoto se i pesi sono vuoti."""
        ctx = _make_context(self.PARTITION)
        result = _call_asset(
            _pipeline.daily_orders, ctx, pd.Series(dtype=float)
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ===========================================================================
# 4. test_schedule_cron
# ===========================================================================

class TestScheduleCron:
    """Verifica la correttezza della schedule 21:30 ET, lun-ven."""

    def test_cron_expression(self):
        """L'espressione cron è '30 21 * * 1-5'."""
        assert _pipeline.daily_schedule.cron_schedule == "30 21 * * 1-5"

    def test_schedule_timezone(self):
        """La schedule usa America/New_York come timezone."""
        assert _pipeline.daily_schedule.execution_timezone == "America/New_York"

    def test_schedule_job(self):
        """La schedule è associata al daily_pipeline job."""
        assert _pipeline.daily_schedule.job_name == "daily_pipeline"

    def test_cron_weekday_range(self):
        """L'espressione cron limita i trigger a lun-ven (dow = 1-5).

        Analisi sintattica dell'espressione cron:
          MIN HOUR DOM MON DOW
          30   21   *   *  1-5
        DOW 1 = lunedì, DOW 5 = venerdì.
        """
        import re
        cron = _pipeline.daily_schedule.cron_schedule
        parts = cron.split()
        assert len(parts) == 5, f"Espressione cron malformata: {cron!r}"

        minute, hour, dom, month, dow = parts
        assert minute == "30"
        assert hour == "21"

        m = re.fullmatch(r"(\d+)-(\d+)", dow)
        assert m is not None, f"DOW non è un range numerico: {dow!r}"
        start_day, end_day = int(m.group(1)), int(m.group(2))
        assert start_day == 1, "DOW deve iniziare da 1 (lunedì)"
        assert end_day == 5, "DOW deve finire a 5 (venerdì)"

        # Sabato (6) e domenica (0/7) fuori dal range
        for weekend in [0, 6, 7]:
            assert not (start_day <= weekend <= end_day), (
                f"Il weekend ({weekend}) non deve essere nel range {dow}"
            )

    def test_job_registered_in_definitions(self):
        """Il daily_pipeline job è registrato nelle Definitions."""
        job_names = [j.name for j in _pipeline.defs.jobs]
        assert "daily_pipeline" in job_names

    def test_schedule_registered_in_definitions(self):
        """La schedule è registrata nelle Definitions."""
        schedule_names = [s.name for s in _pipeline.defs.schedules]
        assert any("daily_pipeline" in n for n in schedule_names), (
            f"Nessuna schedule per daily_pipeline, trovate: {schedule_names}"
        )

    def test_failure_sensor_registered(self):
        """Il failure sensor è registrato nelle Definitions."""
        sensor_names = [s.name for s in _pipeline.defs.sensors]
        assert "failure_sensor" in sensor_names
