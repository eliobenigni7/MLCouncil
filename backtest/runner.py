"""Backtest runner e paper-trading launcher per MLCouncil.

Backtest
--------
Utilizza NautilusTrader ``BacktestEngine`` per simulare la CouncilStrategy
su dati storici OHLCV con:
  - Fill model: next-open (ordine EOD → fill su apertura T+1)
  - Slippage:   3 bps probabilistico + aggiustamento diretto in strategy
  - Commission: 1 bps fee-model
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
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

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

def run_backtest(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_capital: float = 100_000.0,
    universe: Optional[list[str]] = None,
    slippage_bps: float = 3.0,
    commission_bps: float = 1.0,
    verbose: bool = False,
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
        Slippage in basis-point per side (default 3 bps).
    commission_bps : float
        Commissioni in basis-point (default 1 bps).
    verbose : bool
        Se True, abilita il logging NautilusTrader.

    Returns
    -------
    BacktestResult
    """
    if not _NT_AVAILABLE:
        raise ImportError(
            "NautilusTrader non è installato. "
            "Esegui: pip install nautilus_trader"
        )

    # Carica universo
    if universe is None:
        try:
            import yaml
            with open(_ROOT / "config" / "universe.yaml") as f:
                cfg = yaml.safe_load(f)
            universe = cfg["universe"]["tickers"]
        except Exception:
            universe = ["AAPL", "MSFT", "GOOGL"]

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

    equity_curve = _compute_equity_curve(fills_df, initial_capital)
    stats = _compute_stats(equity_curve, fills_df, initial_capital)

    engine.dispose()

    return BacktestResult(
        fills=fills_df,
        positions=positions_df,
        equity_curve=equity_curve,
        stats=stats,
        strategy_fills=strategy_fills_df,
    )


# ===========================================================================
# run_paper_trading
# ===========================================================================

def run_paper_trading(
    universe: Optional[list[str]] = None,
    initial_capital: float = 100_000.0,
) -> None:
    """Avvia il paper trading live tramite Alpaca sandbox.

    Prerequisiti
    ------------
    1. Variabili d'ambiente: ALPACA_API_KEY, ALPACA_SECRET_KEY
    2. Pacchetto: nautilus_trader_adapters_alpaca (oppure adattatore custom)
    3. La pipeline Dagster deve girare in parallelo producendo
       ``data/orders/{today}.parquet`` ogni sera alle 21:30 ET.

    Note
    ----
    Il paper trading usa lo stesso codice del backtest (CouncilStrategy).
    L'unica differenza è il venue adapter.  In questa PoC implementiamo
    lo stub — in produzione connettere un LiveNode NautilusTrader.
    """
    if not _NT_AVAILABLE:
        raise ImportError("NautilusTrader non installato.")

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        raise EnvironmentError(
            "Variabili ALPACA_API_KEY e ALPACA_SECRET_KEY non impostate.\n"
            "Configura il file .env con le credenziali Alpaca sandbox."
        )

    if universe is None:
        try:
            import yaml
            with open(_ROOT / "config" / "universe.yaml") as f:
                cfg = yaml.safe_load(f)
            universe = cfg["universe"]["tickers"]
        except Exception:
            universe = ["AAPL", "MSFT", "GOOGL"]

    # ─── Stub: in produzione usare TradingNode + AlpacaAdapterConfig ───────
    # from nautilus_trader.live.node import TradingNode, TradingNodeConfig
    # from nautilus_trader_adapters_alpaca import AlpacaDataClientConfig, AlpacaExecClientConfig
    #
    # config = TradingNodeConfig(
    #     trader_id="COUNCIL-PAPER-001",
    #     data_clients={"ALPACA": AlpacaDataClientConfig(api_key=api_key, api_secret=secret_key, base_url=base_url)},
    #     exec_clients={"ALPACA": AlpacaExecClientConfig(api_key=api_key, api_secret=secret_key, base_url=base_url, account_type="paper")},
    # )
    # node = TradingNode(config=config)
    # strategy = CouncilStrategy(CouncilStrategyConfig(universe=universe, venue_name="ALPACA", ...))
    # node.trader.add_strategy(strategy)
    # node.run()

    print(
        f"[paper-trading] Stub attivo.\n"
        f"  API Key: {api_key[:4]}***\n"
        f"  Venue  : {base_url}\n"
        f"  Universe: {universe}\n"
        f"\nPer la produzione: connettere TradingNode + AlpacaAdapter."
    )


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
) -> dict:
    """Calcola le metriche principali del backtest."""
    stats: dict = {}

    if equity.empty or len(equity) < 2:
        return stats

    returns = equity.pct_change().dropna()

    # Sharpe annualizzato
    if returns.std() > 0:
        excess = returns - risk_free_rate / 252
        stats["sharpe"] = float(excess.mean() / returns.std() * np.sqrt(252))
    else:
        stats["sharpe"] = 0.0

    # Max drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    stats["max_drawdown"] = float(drawdown.min())

    # CAGR
    n_years = len(equity) / 252
    if n_years > 0 and initial_capital > 0:
        final_equity = float(equity.iloc[-1])
        cagr = (final_equity / initial_capital) ** (1 / n_years) - 1
        stats["cagr"] = float(cagr)
    else:
        stats["cagr"] = 0.0

    # Calmar
    if abs(stats.get("max_drawdown", 0)) > 1e-9:
        stats["calmar"] = stats["cagr"] / abs(stats["max_drawdown"])
    else:
        stats["calmar"] = float("inf")

    # Numero di trade
    stats["n_trades"] = len(fills_df)

    # Annate simulate
    stats["n_years"] = round(n_years, 2)
    stats["final_equity"] = float(equity.iloc[-1])

    return stats
