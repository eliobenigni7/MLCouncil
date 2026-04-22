"""Backtest runner e paper-trading launcher per MLCouncil.

Backtest
--------
Utilizza NautilusTrader ``BacktestEngine`` per simulare la CouncilStrategy
su dati storici OHLCV con:
  - Fill model: next-open (ordine EOD → fill su apertura T+1)
  - Slippage:   3 bps probabilistico + aggiustamento diretto in strategy
  - Commission: default 0 bps (configurabile da env)
  - Venue:      SimulatedExchange (SIM) cash account, long-only

Paper Trading
-------------
Stessa strategy, venue adapter Alpaca sandbox.
Richiede le variabili d'ambiente ALPACA_API_KEY, ALPACA_SECRET_KEY.

Utilizzo
--------
    from backtest.runner import run_backtest, run_paper_trading

    result = run_backtest("2020-01-01", "2023-12-31", initial_capital=100_000)
    print(result.stats)

    # Avvia paper trading (bloccante):
    run_paper_trading()

Nota PoC
--------
- I dati OHLCV vengono letti dai parquet in ``data/raw/ohlcv/`` (scritti dalla
  pipeline Dagster).  Se non presenti, si usa yfinance direttamente.
- Per il paper trading Alpaca è richiesto il pacchetto
  ``nautilus_trader_adapters_alpaca`` (non incluso in questa PoC).
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
try:
    import polars as pl
except ImportError:  # pragma: no cover
    class _PolarsStub:
        class DataFrame:  # minimal type placeholder for postponed runtime access
            pass

        def __getattr__(self, name):
            raise ImportError(
                "polars is required for OHLCV loading. Install with: pip install polars"
            )

    pl = _PolarsStub()

try:
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.backtest.models import FillModel
    from nautilus_trader.backtest.config import BacktestVenueConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import Bar, BarType, BarSpecification
    from nautilus_trader.model.enums import (
        AggregationSource,
        BarAggregation,
        OmsType,
        AccountType,
        PriceType,
    )
    from nautilus_trader.model.identifiers import (
        InstrumentId,
        Symbol,
        TraderId,
        Venue,
    )
    from nautilus_trader.model.instruments import Equity
    from nautilus_trader.model.objects import Money, Price, Quantity
    from nautilus_trader.config import LoggingConfig

    _NT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NT_AVAILABLE = False

_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA_DIR = _ROOT / "data" / "raw"

from council.transaction_costs import (
    get_default_commission_bps,
    get_default_slippage_bps,
)


# ===========================================================================
# BacktestResult
# ===========================================================================

@dataclass
class BacktestResult:
    """Risultati del backtest.

    Attributes
    ----------
    fills : pd.DataFrame
        Fill report da NautilusTrader (ogni riga = un trade eseguito).
    positions : pd.DataFrame
        Position report.
    equity_curve : pd.Series
        Equity curve giornaliera (index=date, values=portfolio_value_usd).
    stats : dict
        Metriche aggregate: sharpe, max_dd, calmar, turnover, n_trades.
    strategy_fills : pd.DataFrame
        Fill tracker interno alla CouncilStrategy (con slippage attribution).
    """

    fills: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    gross_equity_curve: pd.Series = field(default_factory=pd.Series)
    stats: dict = field(default_factory=dict)
    strategy_fills: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_trades(self) -> int:
        return len(self.fills)

    @property
    def sharpe(self) -> float:
        return float(self.stats.get("sharpe", float("nan")))

    @property
    def max_drawdown(self) -> float:
        return float(self.stats.get("max_drawdown", float("nan")))

    def __repr__(self) -> str:
        return (
            f"BacktestResult("
            f"n_trades={self.n_trades}, "
            f"sharpe={self.sharpe:.3f}, "
            f"max_dd={self.max_drawdown:.1%}"
            f")"
        )


# ===========================================================================
# Data helpers
# ===========================================================================

def _load_ohlcv_polars(
    ticker: str,
    start: str,
    end: str,
) -> pl.DataFrame:
    """Carica OHLCV da parquet locali, con fallback yfinance."""
    ohlcv_dir = _DATA_DIR / "ohlcv" / ticker
    frames: list[pl.DataFrame] = []

    if ohlcv_dir.exists():
        for pq in sorted(ohlcv_dir.glob("*.parquet")):
            try:
                frames.append(pl.read_parquet(pq))
            except Exception:
                pass

    if frames:
        df = (
            pl.concat(frames)
            .filter(
                (pl.col("valid_time") >= pl.lit(start).str.to_date())
                & (pl.col("valid_time") <= pl.lit(end).str.to_date())
            )
            .sort("valid_time")
        )
        if not df.is_empty():
            return df

    # Fallback: yfinance
    try:
        import yfinance as yf
        import pandas as pd_yf

        raw = yf.download(
            ticker, start=start, end=end,
            auto_adjust=False, progress=False, actions=False,
        )
        if raw is None or raw.empty:
            return pl.DataFrame()

        if isinstance(raw.columns, pd_yf.MultiIndex):
            raw.columns = [col[0] for col in raw.columns]
        raw = raw.rename(columns={"Adj Close": "adj_close"})
        raw.columns = [c.lower() for c in raw.columns]
        raw = raw.reset_index().rename(columns={"date": "valid_time", "Date": "valid_time"})
        df = pl.from_pandas(raw)
        if "valid_time" in df.columns and df["valid_time"].dtype != pl.Date:
            df = df.with_columns(pl.col("valid_time").cast(pl.Date))
        return df.sort("valid_time")
    except Exception:
        return pl.DataFrame()


def _ohlcv_to_nautilus_bars(
    df: pl.DataFrame,
    ticker: str,
    venue: "Venue",
    bar_type: "BarType",
) -> list["Bar"]:
    """Converte un DataFrame Polars OHLCV in una lista di Bar NautilusTrader."""
    if not _NT_AVAILABLE:
        return []

    bars: list[Bar] = []
    df_pd = df.to_pandas()

    for _, row in df_pd.iterrows():
        # Converti la data in timestamp nanosecondo UTC
        ts_date = row["valid_time"]
        if hasattr(ts_date, "to_pydatetime"):
            ts_date = ts_date.to_pydatetime()
        if hasattr(ts_date, "date"):
            ts_date = ts_date.date()

        # EOD bar: timestamp = chiusura del mercato (21:00 ET = 02:00 UTC+1 giorno)
        ts_ns = int(
            datetime(ts_date.year, ts_date.month, ts_date.day, 21, 0, 0, tzinfo=timezone.utc)
            .timestamp() * 1_000_000_000
        )

        try:
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{float(row.get('open', row.get('close', 0))):.4f}"),
                high=Price.from_str(f"{float(row.get('high', row.get('close', 0))):.4f}"),
                low=Price.from_str(f"{float(row.get('low', row.get('close', 0))):.4f}"),
                close=Price.from_str(f"{float(row.get('adj_close', row.get('close', 0))):.4f}"),
                volume=Quantity.from_int(max(1, int(row.get("volume", 1_000_000)))),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
            bars.append(bar)
        except Exception:
            continue

    return bars


def _make_equity(ticker: str, venue: "Venue") -> "Equity":
    """Crea un oggetto Equity NautilusTrader per un ticker."""
    instrument_id = InstrumentId(Symbol(ticker), venue)
    return Equity(
        instrument_id=instrument_id,
        raw_symbol=Symbol(ticker),
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
        maker_fee=Decimal("0.0001"),   # 1 bps maker
        taker_fee=Decimal("0.0003"),   # 3 bps taker
    )


# ===========================================================================
# run_backtest
# ===========================================================================

def _run_lookahead_preflight(start_date: str, end_date: str) -> None:
    """Load available feature parquets and run lookahead-bias checks.

    Warns on any suspicious findings but does not block the backtest.
    """
    import logging as _logging

    _logger = _logging.getLogger(__name__)
    features_dir = _ROOT / "data" / "features" / "alpha158"

    if not features_dir.exists():
        _logger.debug("No alpha158 feature dir found — skipping lookahead check.")
        return

    # Sample up to 5 parquet files from the backtest period
    parquets = sorted(features_dir.glob("*.parquet"))
    if not parquets:
        return

    sample_files = [
        p for p in parquets
        if start_date <= p.stem <= end_date
    ][:5]

    if not sample_files:
        sample_files = parquets[:3]

    try:
        from backtest.validation import validate_no_lookahead

        frames = []
        for pq in sample_files:
            try:
                frames.append(pd.read_parquet(pq))
            except Exception:
                continue
        if not frames:
            return

        df = pd.concat(frames, ignore_index=True)
        warnings = validate_no_lookahead(
            features_df=df,
            target_col="forward_return",
            date_col="valid_time",
        )
        if warnings:
            _logger.warning(
                "Lookahead bias pre-flight found %d warning(s) — "
                "review feature pipeline before trusting backtest results.",
                len(warnings),
            )
    except Exception as exc:
        _logger.debug("Lookahead pre-flight skipped: %s", exc)


def run_backtest(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_capital: float = 100_000.0,
    universe: Optional[list[str]] = None,
    slippage_bps: float | None = None,
    commission_bps: float | None = None,
    verbose: bool = False,
    skip_lookahead_check: bool = False,
) -> BacktestResult:
    """Configura e lancia il backtest NautilusTrader.

    Parametri
    ---------
    start_date : str
        Data di inizio backtest (ISO 8601).
    end_date : str
        Data di fine backtest (ISO 8601).
    initial_capital : float
        Capitale iniziale in USD.
    universe : list[str], optional
        Override lista ticker. Default da config/universe.yaml.
    slippage_bps : float
        Slippage in basis-point per side (default da MLCOUNCIL_SLIPPAGE_BPS).
    commission_bps : float
        Commissioni in basis-point (default da MLCOUNCIL_COMMISSION_BPS).
    verbose : bool
        Se True, abilita il logging NautilusTrader.
    skip_lookahead_check : bool
        Se True, salta la validazione di lookahead bias pre-backtest.

    Returns
    -------
    BacktestResult
    """
    if not _NT_AVAILABLE:
        raise ImportError(
            "NautilusTrader non è installato. "
            "Esegui: pip install nautilus_trader"
        )

    if slippage_bps is None:
        slippage_bps = get_default_slippage_bps()
    if commission_bps is None:
        commission_bps = get_default_commission_bps()

    # Carica universo (survivorship-bias-aware when history file exists)
    if universe is None:
        try:
            from data.pipeline import load_universe_as_of
            universe = load_universe_as_of(as_of_date=start_date)
        except Exception:
            try:
                from data.pipeline import _load_universe
                universe = _load_universe()
            except Exception:
                universe = ["AAPL", "MSFT", "GOOGL"]

    # ------------------------------------------------------------------
    # 0. Pre-flight: lookahead-bias validation on available features
    # ------------------------------------------------------------------
    if not skip_lookahead_check:
        _run_lookahead_preflight(start_date, end_date)

    # ------------------------------------------------------------------
    # 1. Configura BacktestEngine
    # ------------------------------------------------------------------
    logging_cfg = LoggingConfig(log_level="ERROR" if not verbose else "INFO")
    engine_cfg = BacktestEngineConfig(
        trader_id=TraderId("COUNCIL-001"),
        logging=logging_cfg,
        run_analysis=True,
    )
    engine = BacktestEngine(engine_cfg)

    # ------------------------------------------------------------------
    # 2. Aggiungi venue SIM (SimulatedExchange)
    # ------------------------------------------------------------------
    sim_venue = Venue("SIM")

    # FillModel: prob_slippage applica uno slippage casuale di 1 tick
    # in aggiunta allo slippage esplicito nella strategy
    fill_model = FillModel(
        prob_fill_on_limit=1.0,
        prob_fill_on_stop=1.0,
        prob_slippage=slippage_bps / 100.0,   # ~3 bps ≈ 3% prob 1-tick slip
        random_seed=42,
    )

    engine.add_venue(
        sim_venue,
        OmsType.NETTING,
        AccountType.CASH,
        starting_balances=[Money(initial_capital, USD)],
        base_currency=USD,
        fill_model=fill_model,
        bar_execution=True,            # fill su barra successiva (next-open)
        bar_adaptive_high_low_ordering=True,
    )

    # ------------------------------------------------------------------
    # 3. Carica dati e crea strumenti
    # ------------------------------------------------------------------
    loaded_tickers: list[str] = []

    for ticker in universe:
        df = _load_ohlcv_polars(ticker, start_date, end_date)
        if df.is_empty():
            continue

        equity = _make_equity(ticker, sim_venue)
        engine.add_instrument(equity)

        bar_spec = BarSpecification(
            step=1,
            aggregation=BarAggregation.DAY,
            price_type=PriceType.LAST,
        )
        bar_type = BarType(
            instrument_id=equity.id,
            spec=bar_spec,
            aggregation_source=AggregationSource.EXTERNAL,
        )
        bars = _ohlcv_to_nautilus_bars(df, ticker, sim_venue, bar_type)
        if bars:
            engine.add_data(bars)
            loaded_tickers.append(ticker)

    if not loaded_tickers:
        raise ValueError(
            f"Nessun dato OHLCV trovato per il periodo {start_date}..{end_date}. "
            "Assicurarsi che la pipeline Dagster abbia già scaricato i dati "
            "oppure che yfinance sia accessibile."
        )

    # ------------------------------------------------------------------
    # 4. Aggiungi CouncilStrategy
    # ------------------------------------------------------------------
    from backtest.strategy import CouncilStrategy, CouncilStrategyConfig

    strategy_cfg = CouncilStrategyConfig(
        strategy_id="COUNCIL-001",
        universe=loaded_tickers,
        venue_name="SIM",
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
        portfolio_value=initial_capital,
    )
    strategy = CouncilStrategy(strategy_cfg)
    engine.add_strategy(strategy)

    # ------------------------------------------------------------------
    # 5. Esegui il backtest
    # ------------------------------------------------------------------
    engine.run()

    # ------------------------------------------------------------------
    # 6. Raccogli i risultati
    # ------------------------------------------------------------------
    fills_df    = engine.trader.generate_fills_report()
    positions_df = engine.trader.generate_positions_report()
    strategy_fills_df = strategy.get_fill_report()

    gross_equity_curve = _compute_equity_curve(fills_df, initial_capital)
    stats = _compute_stats(
        gross_equity_curve,
        fills_df,
        initial_capital,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )

    engine.dispose()

    result = BacktestResult(
        fills=fills_df,
        positions=positions_df,
        equity_curve=stats.pop("_net_equity_curve", gross_equity_curve),
        gross_equity_curve=gross_equity_curve,
        stats=stats,
        strategy_fills=strategy_fills_df,
    )
    result.start_date = start_date
    result.end_date = end_date
    result.initial_capital = initial_capital

    try:
        from council.mlflow_utils import log_backtest_result

        log_backtest_result(
            result,
            pipeline_run_id=f"backtest-{start_date}-{end_date}",
            data_version=f"backtest-data-{len(loaded_tickers)}",
            feature_version="backtest-sim",
            environment="paper",
            model_name="council",
        )
    except Exception:
        pass

    return result


def _collect_environment_metadata() -> dict[str, str]:
    """Capture reproducibility metadata for retrospective validation runs."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "runtime_profile": os.getenv("MLCOUNCIL_ENV_PROFILE", "unknown"),
        "slippage_bps_default": str(get_default_slippage_bps()),
        "commission_bps_default": str(get_default_commission_bps()),
    }


def run_walk_forward_backtest(
    *,
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    train_window: int = 252,
    test_window: int = 63,
    step: int | None = None,
    purge_period: int = 1,
    embargo_period: int = 1,
    benchmark_returns: dict[str, pd.Series] | None = None,
    regime_labels: pd.Series | None = None,
    component_signals: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Run deterministic walk-forward validation with benchmark/regime diagnostics."""
    from backtest.validation import run_walk_forward_analysis

    result = run_walk_forward_analysis(
        signals=signals,
        forward_returns=forward_returns,
        train_window=train_window,
        test_window=test_window,
        step=step,
        purge_period=purge_period,
        embargo_period=embargo_period,
        benchmark_returns=benchmark_returns,
        regime_labels=regime_labels,
        component_signals=component_signals,
    )
    result["environment_metadata"] = _collect_environment_metadata()
    return result


# ===========================================================================
# run_paper_trading
# ===========================================================================

def run_paper_trading(
    universe: Optional[list[str]] = None,
    initial_capital: float = 100_000.0,
    trading_mode: str = "paper",
) -> None:
    """Launch live/paper trading via Alpaca.

    Prerequisites
    ------------
    1. Environment variables:
       - ALPACA_PAPER_KEY, ALPACA_PAPER_SECRET (for paper trading)
       - ALPACA_LIVE_KEY, ALPACA_LIVE_SECRET (for live trading)
       - TRADING_MODE=paper or TRADING_MODE=live
    2. Package: alpaca-trade-api
    3. Dagster pipeline running in parallel producing ``data/orders/{today}.parquet``

    Parameters
    ----------
    universe : list[str], optional
        Override ticker list. Defaults to config/universe.yaml.
    initial_capital : float
        Starting capital (informational only for paper trading).
    trading_mode : str
        "paper" (default) or "live". Can also set TRADING_MODE env var.
    """
    try:
        from execution.alpaca_adapter import AlpacaLiveNode, AlpacaConfig, TradingMode
    except ImportError:
        raise ImportError(
            "execution.alpaca_adapter not found or alpaca-trade-api not installed.\n"
            "Install with: pip install alpaca-trade-api"
        )

    mode = TradingMode(trading_mode) if trading_mode else TradingMode(os.getenv("TRADING_MODE", "paper"))

    if universe is None:
        try:
            import yaml
            with open(_ROOT / "config" / "universe.yaml") as f:
                cfg = yaml.safe_load(f)
            all_tickers = cfg["universe"].get("large_cap", []) + cfg["universe"].get("mid_cap", [])
            universe = all_tickers if all_tickers else ["AAPL", "MSFT", "GOOGL"]
        except Exception:
            universe = ["AAPL", "MSFT", "GOOGL"]

    config = AlpacaConfig.from_env()
    config.mode = mode

    try:
        node = AlpacaLiveNode(config)
    except EnvironmentError as e:
        raise EnvironmentError(
            f"Alpaca configuration error: {e}\n"
            "Set ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET (or ALPACA_LIVE_KEY/LIVE_SECRET) "
            "in your environment or .env file."
        )

    account_info = node.get_account_info()
    print(
        f"[paper-trading] Alpaca node initialized.\n"
        f"  Mode         : {mode.value}\n"
        f"  Account      : ${account_info['portfolio_value']:,.2f}\n"
        f"  Buying Power : ${account_info['buying_power']:,.2f}\n"
        f"  Status       : {account_info['status']}\n"
        f"  Universe     : {len(universe)} tickers"
    )

    positions = node.get_all_positions()
    if not positions.empty:
        print(f"\n  Current positions:")
        for _, pos in positions.iterrows():
            print(f"    {pos['symbol']}: {pos['qty']} shares @ ${pos['avg_price']:.2f}")
    else:
        print("\n  No open positions.")

    print(
        f"\n  Orders will be read from: data/orders/{{date}}.parquet\n"
        f"  Trade logs will be saved to: data/paper_trades/{{date}}.json\n"
        f"\n  To submit today's orders, call:\n"
        f"    from backtest.runner import submit_daily_orders\n"
        f"    submit_daily_orders(node)"
    )


def submit_daily_orders(
    node: "AlpacaLiveNode",
    date: Optional[str] = None,
    max_position_size: Optional[float] = None,
) -> list[dict]:
    """Submit today's orders from parquet to Alpaca.

    Parameters
    ----------
    node : AlpacaLiveNode
        Configured Alpaca trading node.
    date : str, optional
        Date string YYYY-MM-DD. Defaults to today.
    max_position_size : float, optional
        Maximum position size in USD. Defaults to MAX_POSITION_SIZE env var.

    Returns
    -------
    list[dict]
        List of submitted order records.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    orders_file = _ROOT / "data" / "orders" / f"{date}.parquet"
    if not orders_file.exists():
        print(f"No orders found for {date} at {orders_file}")
        return []

    try:
        orders_df = pd.read_parquet(orders_file)
    except Exception as e:
        print(f"Error reading orders file: {e}")
        return []

    if orders_df.empty:
        print(f"No orders generated for {date}")
        return []

    print(f"Found {len(orders_df)} orders for {date}")

    if "direction" not in orders_df.columns or "ticker" not in orders_df.columns:
        print("Orders file missing required columns: direction, ticker")
        return []

    submitted = []
    for _, order in orders_df.iterrows():
        symbol = order["ticker"]
        side = order["direction"].lower()
        qty = int(order.get("quantity", order.get("qty", 0)))

        if qty <= 0:
            continue

        max_pos = max_position_size or float(os.getenv("MAX_POSITION_SIZE", "50000"))
        limit_ok, limit_msg = node.check_position_limits(symbol, qty, max_pos)
        if not limit_ok:
            print(f"Skipping {symbol} {side} {qty}: {limit_msg}")
            continue

        try:
            result = node.submit_order(symbol, qty, side)
            print(f"Submitted: {symbol} {side} {qty} -> order_id={result.get('order_id', '?')}")
            submitted.append(result)
        except Exception as e:
            print(f"Error submitting {symbol} {side} {qty}: {e}")

    print(f"\nSubmitted {len(submitted)} orders")
    return submitted


def get_paper_trade_status(date: Optional[str] = None) -> dict:
    """Get status of paper trades for a given date.

    Parameters
    ----------
    date : str, optional
        Date string YYYY-MM-DD. Defaults to today.

    Returns
    -------
    dict
        Summary of orders and their fill status.
    """
    try:
        from execution.alpaca_adapter import AlpacaLiveNode, AlpacaConfig
    except ImportError:
        return {"error": "Alpaca adapter not available"}

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    config = AlpacaConfig.from_env()
    try:
        node = AlpacaLiveNode(config)
    except Exception as e:
        return {"error": str(e)}

    trades = node.get_trade_log(date)
    positions = node.get_all_positions()
    account = node.get_account_info()

    return {
        "date": date,
        "mode": config.mode.value,
        "submitted_orders": len(trades),
        "open_positions": len(positions),
        "account_value": account["portfolio_value"],
        "buying_power": account["buying_power"],
        "trades": trades,
        "positions": positions.to_dict(orient="records") if not positions.empty else [],
    }


# ===========================================================================
# Helpers statistiche
# ===========================================================================

def _compute_equity_curve(
    fills_df: pd.DataFrame,
    initial_capital: float,
) -> pd.Series:
    """Ricostruisce l'equity curve dai fill.

    Approccio semplificato: ogni fill modifica il P&L basandosi su
    realized_pnl (se disponibile) o su prezzo × quantità.
    """
    if fills_df.empty:
        return pd.Series(
            [initial_capital], index=[pd.Timestamp("today").date()], name="equity"
        )

    df = fills_df.copy()

    # Normalizza colonne — Nautilus usa nomi variabili per versione
    col_map = {
        "ts_event":       ["ts_event", "timestamp", "ts_init"],
        "last_px":        ["last_px", "fill_price", "avg_px"],
        "last_qty":       ["last_qty", "quantity", "qty"],
        "side":           ["order_side", "side"],
        "realized_pnl":   ["realized_pnl", "pnl"],
    }

    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    ts_col    = _find_col(df, col_map["ts_event"])
    pnl_col   = _find_col(df, col_map["realized_pnl"])

    if ts_col is None:
        return pd.Series([initial_capital], name="equity")

    df["_date"] = pd.to_datetime(df[ts_col], unit="ns", utc=True).dt.date

    if pnl_col and df[pnl_col].notna().any():
        daily_pnl = df.groupby("_date")[pnl_col].sum()
        equity = initial_capital + daily_pnl.cumsum()
    else:
        # Proxy: equity = initial_capital (nessun P&L disponibile)
        unique_dates = sorted(df["_date"].unique())
        equity = pd.Series(initial_capital, index=unique_dates, name="equity")

    equity.index = pd.to_datetime(equity.index)
    equity.name = "equity"
    return equity


def _compute_stats(
    equity: pd.Series,
    fills_df: pd.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.05,
    commission_bps: float | None = None,
    slippage_bps: float | None = None,
) -> dict:
    """Calcola metriche nette e lorde del backtest da un'unica reportistica."""
    stats: dict = {}

    if equity.empty or len(equity) < 2:
        return stats

    resolved_commission_bps = (
        get_default_commission_bps() if commission_bps is None else float(commission_bps)
    )
    resolved_slippage_bps = (
        get_default_slippage_bps() if slippage_bps is None else float(slippage_bps)
    )

    from backtest.report import BacktestReport

    report = BacktestReport(
        fills=fills_df,
        prices=pd.DataFrame(),
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        equity_curve=equity,
        commission_bps=resolved_commission_bps,
        slippage_bps=resolved_slippage_bps,
    )
    turnover_df = report.turnover_analysis()
    net_equity = report.equity_curve
    gross_equity = report.gross_equity_curve
    n_years = len(net_equity) / 252 if len(net_equity) else 0.0

    gross_returns = report.gross_returns
    if not gross_returns.empty and gross_returns.std() > 0:
        gross_excess = gross_returns - risk_free_rate / 252
        gross_sharpe = float(gross_excess.mean() / gross_returns.std() * np.sqrt(252))
    else:
        gross_sharpe = 0.0

    gross_rolling_max = gross_equity.cummax()
    gross_drawdown = (gross_equity - gross_rolling_max) / gross_rolling_max if not gross_equity.empty else pd.Series(dtype=float)
    gross_max_drawdown = float(gross_drawdown.min()) if not gross_drawdown.empty else 0.0

    if n_years > 0 and initial_capital > 0:
        gross_final_equity = float(gross_equity.iloc[-1])
        net_final_equity = float(net_equity.iloc[-1])
        gross_cagr = (gross_final_equity / initial_capital) ** (1 / n_years) - 1
        net_cagr = (net_final_equity / initial_capital) ** (1 / n_years) - 1
    else:
        gross_final_equity = float(gross_equity.iloc[-1]) if not gross_equity.empty else initial_capital
        net_final_equity = float(net_equity.iloc[-1]) if not net_equity.empty else initial_capital
        gross_cagr = 0.0
        net_cagr = 0.0

    stats["sharpe"] = float(report.sharpe_ratio())
    stats["gross_sharpe"] = gross_sharpe
    stats["max_drawdown"] = float(report.max_drawdown())
    stats["gross_max_drawdown"] = gross_max_drawdown
    stats["cagr"] = float(net_cagr)
    stats["gross_cagr"] = float(gross_cagr)
    stats["calmar"] = float(report.calmar_ratio())
    stats["gross_calmar"] = (
        float(gross_cagr / abs(gross_max_drawdown))
        if abs(gross_max_drawdown) > 1e-9
        else float("inf")
    )
    stats["n_trades"] = len(fills_df)
    stats["n_years"] = round(n_years, 2)
    stats["final_equity"] = net_final_equity
    stats["gross_final_equity"] = gross_final_equity
    stats["estimated_costs_usd"] = float(report.total_estimated_cost_usd)
    stats["turnover"] = (
        float(turnover_df["turnover_pct"].mean())
        if not turnover_df.empty and "turnover_pct" in turnover_df.columns
        else 0.0
    )
    stats["_net_equity_curve"] = net_equity

    return stats
