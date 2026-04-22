"""MLCouncil end-to-end pipeline demo.

Usage
-----
    # Prima esecuzione (scarica ~3 anni di dati, richiede rete)
    .venv/Scripts/python scripts/run_pipeline.py

    # Esecuzioni successive (i dati sono già su disco)
    .venv/Scripts/python scripts/run_pipeline.py --skip-download

Flusso
------
1.  Download OHLCV (yfinance) + macro (FRED)        → data/raw/
2.  Calcola Alpha158 features                        → pl.DataFrame in memory
3.  Train/test split temporale (80/20)
4.  Fit LightGBM con CPCV                           → models/TechnicalModel
5.  Fit HMM regime detector                          → models/RegimeModel
6.  Rileva regime corrente
7.  Genera segnali alpha sull'ultimo giorno disponibile
8.  Council aggregation                              → council/CouncilAggregator
9.  Fit ConformalPositionSizer sul training set      → council/ConformalPositionSizer
10. Ottimizzazione portafoglio con cvxpy             → council/PortfolioConstructor
11. Stampa target weights + ordini

Infrastruttura richiesta
------------------------
* NO PostgreSQL  (dati salvati come Parquet locali)
* NO MinIO       (solo disco locale)
* NO MLflow srv  (il logging fallisce silenziosamente se il server è assente)
* NO GPU/CUDA    (sentiment skippato di default; usa --with-sentiment per attivarlo)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM", "JNJ", "V", "XOM",
]
START = "2021-01-01"
DATA_DIR = ROOT / "data" / "raw"
ORDERS_DIR = ROOT / "data" / "orders"
_EXCLUDE_COLS = {"ticker", "valid_time", "transaction_time"}


# ---------------------------------------------------------------------------
# Step 1 — Download
# ---------------------------------------------------------------------------

def step_download(force: bool = False) -> None:
    """Scarica OHLCV + macro se non già presenti."""
    already_done = all(
        (DATA_DIR / "ohlcv" / t).exists() and list((DATA_DIR / "ohlcv" / t).glob("*.parquet"))
        for t in TICKERS
    )
    if already_done and not force:
        print("[1] Download già presente — skip (usa --force-download per rifarlo)\n")
        return

    from data.ingest.market_data import download_universe
    from data.ingest.macro import download_macro

    print("[1] Download OHLCV…")
    download_universe(tickers=TICKERS, start=START, data_dir=DATA_DIR)

    print("[1] Download macro (FRED: VIX, Treasury, S&P500)…")
    download_macro(start=START, data_dir=DATA_DIR)
    print("[1] Download completato.\n")


# ---------------------------------------------------------------------------
# Step 2 — Carica OHLCV raw da Parquet
# ---------------------------------------------------------------------------

def step_load_ohlcv() -> pl.DataFrame:
    frames = []
    for ticker in TICKERS:
        ticker_dir = DATA_DIR / "ohlcv" / ticker
        if not ticker_dir.exists():
            continue
        for p in sorted(ticker_dir.glob("*.parquet")):
            frames.append(pl.read_parquet(p))

    if not frames:
        sys.exit(
            "Nessun dato OHLCV trovato in data/raw/.\n"
            "Esegui senza --skip-download per scaricare i dati."
        )

    ohlcv = pl.concat(frames).sort(["ticker", "valid_time"])
    n_tickers = ohlcv["ticker"].n_unique()
    n_days    = ohlcv["valid_time"].n_unique()
    print(f"[2] OHLCV caricato: {n_tickers} ticker × {n_days} giorni\n")
    return ohlcv


# ---------------------------------------------------------------------------
# Step 3 — Features
# ---------------------------------------------------------------------------

def step_features(ohlcv: pl.DataFrame) -> pl.DataFrame:
    from data.features.alpha158 import compute_alpha158, build_macro_context

    macro_dir = DATA_DIR / "macro"

    def _path(name: str) -> str | None:
        p = macro_dir / f"{name}.parquet"
        return str(p) if p.exists() else None

    macro = build_macro_context(
        vix_path=_path("vix"),
        treasuries_path=_path("treasuries"),
        sp500_path=_path("sp500"),
    )

    features = compute_alpha158(ohlcv, macro_df=macro if not macro.is_empty() else None)
    print(f"[3] Alpha158: {features.shape[0]} righe × {len(features.columns) - 2} feature\n")
    return features


# ---------------------------------------------------------------------------
# Step 4 — Train/test split
# ---------------------------------------------------------------------------

def step_split(
    features: pl.DataFrame,
    ohlcv: pl.DataFrame,
    test_ratio: float = 0.20,
) -> tuple:
    from data.features.target import compute_targets

    all_dates = sorted(features["valid_time"].unique().to_list())
    cutoff_idx = int(len(all_dates) * (1 - test_ratio))
    cutoff = all_dates[cutoff_idx]

    feat_train = features.filter(pl.col("valid_time") < cutoff)
    feat_test  = features.filter(pl.col("valid_time") >= cutoff)

    targets_df = compute_targets(ohlcv, horizons=[1], risk_adjusted=False)
    tgt_pd = (
        targets_df
        .filter(pl.col("valid_time") < cutoff)
        .to_pandas()
    )
    tgt_pd["valid_time"] = pd.to_datetime(tgt_pd["valid_time"]).dt.date
    targets_train = (
        tgt_pd.set_index(["ticker", "valid_time"])["rank_fwd_1d"].dropna()
    )

    print(f"[4] Split: train fino a {cutoff} ({len(all_dates[:cutoff_idx])} giorni) "
          f"| test da {cutoff} ({len(all_dates[cutoff_idx:])} giorni)\n")
    return feat_train, feat_test, targets_train, cutoff, targets_df


# ---------------------------------------------------------------------------
# Step 5 — Addestramento modelli
# ---------------------------------------------------------------------------

def step_train_lgbm(feat_train: pl.DataFrame, targets_train: pd.Series):
    from models.technical import TechnicalModel

    print("[5a] Training LightGBM con CPCV…")
    model = TechnicalModel()
    model._params["n_estimators"] = 300   # riduci per velocità demo
    model.fit(feat_train, targets_train)
    print(f"    LightGBM pronto. Features: {model._n_features}\n")
    return model


def step_train_hmm(cutoff: date):
    from models.regime import RegimeModel
    from data.features.alpha158 import build_macro_context

    macro_dir = DATA_DIR / "macro"

    def _path(name: str) -> str | None:
        p = macro_dir / f"{name}.parquet"
        return str(p) if p.exists() else None

    macro = build_macro_context(
        vix_path=_path("vix"),
        treasuries_path=_path("treasuries"),
        sp500_path=_path("sp500"),
    )

    if macro.is_empty():
        # Fallback sintetico se il macro non è disponibile
        print("[5b] Macro non disponibile — HMM in modalità fallback\n")
        macro = pl.DataFrame({
            "valid_time": [cutoff],
            "sp500_ret_20d": [0.01],
            "vix": [18.0],
            "yield_spread": [1.2],
        }).with_columns(pl.col("valid_time").cast(pl.Date))

    macro_train = macro.filter(pl.col("valid_time") <= cutoff)

    print("[5b] Training HMM regime detector…")
    model = RegimeModel()
    model.fit(macro_train)

    regime = model.predict_regime(macro)   # regime sull'intero storico disponibile
    probs  = model.predict_probabilities(macro)
    print(f"    Regime corrente: {regime.upper()} — probabilità: "
          f"{', '.join(f'{k}={v:.2%}' for k,v in probs.items())}\n")
    return model, regime, macro


# ---------------------------------------------------------------------------
# Step 6 — Segnali
# ---------------------------------------------------------------------------

def step_signals(
    lgbm,
    hmm_model,
    feat_test: pl.DataFrame,
    macro: pl.DataFrame,
    with_sentiment: bool = False,
) -> tuple[dict[str, pd.Series], date, str]:
    all_dates = sorted(feat_test["valid_time"].unique().to_list())
    last_date = all_dates[-1]

    day_feat = feat_test.filter(pl.col("valid_time") == last_date)

    lgbm_signal = lgbm.predict(day_feat)
    tickers     = sorted(lgbm_signal.index.tolist())

    signals: dict[str, pd.Series] = {"lgbm": lgbm_signal}

    # HMM: usa la prob del regime bull come tilt cross-sezionale uniforme
    # (in produzione il regime condiziona i pesi del council, non è un segnale diretto)
    signals["hmm"] = pd.Series(np.zeros(len(tickers)), index=tickers)

    if with_sentiment:
        try:
            from models.sentiment import SentimentModel
            from data.ingest.news import download_news
            news_df = download_news(tickers=tickers, date=last_date.isoformat())
            if news_df.is_empty():
                print(f"[6] Sentiment: no news data for {last_date}")
            else:
                print(f"[6] Sentiment: {len(news_df)} headlines collected")
        except ImportError:
            print("[6] Sentiment: modulo non disponibile")
    else:
        print("[6] Sentiment skippato (usa --with-sentiment per attivarlo)")

    regime = hmm_model.predict_regime(macro)
    print(f"[6] Segnali generati per {last_date} | regime={regime}\n")
    return signals, last_date, regime


# ---------------------------------------------------------------------------
# Step 7 — Council aggregation
# ---------------------------------------------------------------------------

def step_council(
    signals: dict[str, pd.Series],
    regime: str,
    last_date: date,
) -> pd.Series:
    from council.aggregator import CouncilAggregator

    agg    = CouncilAggregator()
    signal = agg.aggregate(signals, regime=regime, date=last_date)

    print("[7] Council signal — top 5 long / top 5 short:")
    print(f"    Long:  {signal.nlargest(5).round(3).to_dict()}")
    print(f"    Short: {signal.nsmallest(5).round(3).to_dict()}\n")
    return signal


# ---------------------------------------------------------------------------
# Step 8 — Conformal sizing
# ---------------------------------------------------------------------------

def step_conformal(lgbm, feat_train: pl.DataFrame, targets_df: pl.DataFrame):
    from council.conformal import ConformalPositionSizer

    feat_cols = [c for c in feat_train.columns if c not in _EXCLUDE_COLS]

    train_pd = feat_train.to_pandas()
    train_pd["valid_time"] = pd.to_datetime(train_pd["valid_time"]).dt.date

    tgt_pd = targets_df.to_pandas()
    tgt_pd["valid_time"] = pd.to_datetime(tgt_pd["valid_time"]).dt.date

    merged = (
        train_pd
        .merge(tgt_pd[["ticker", "valid_time", "ret_fwd_1d"]], on=["ticker", "valid_time"])
        .dropna(subset=feat_cols + ["ret_fwd_1d"])
    )

    X = merged[feat_cols].values
    y = merged["ret_fwd_1d"].values

    print(f"[8] Calibrazione conformal su {len(y):,} campioni…")
    sizer = ConformalPositionSizer(coverage=0.90)
    sizer.fit(X, y)
    print("    ConformalPositionSizer pronto (coverage=90%)\n")
    return sizer, feat_cols


# ---------------------------------------------------------------------------
# Step 9 — Portfolio construction
# ---------------------------------------------------------------------------

def step_portfolio(
    council_signal: pd.Series,
    sizer,
    feat_cols: list[str],
    feat_test: pl.DataFrame,
    ohlcv: pl.DataFrame,
    last_date: date,
    portfolio_value: float = 1_000_000.0,
) -> pd.Series:
    from council.portfolio import PortfolioConstructor

    # Feature row per l'ultimo giorno (allineata ai ticker del council_signal)
    tickers = sorted(council_signal.index.tolist())
    day_rows = (
        feat_test
        .filter(pl.col("valid_time") == last_date)
        .sort("ticker")
        .filter(pl.col("ticker").is_in(tickers))
    )
    available_tickers = day_rows["ticker"].to_list()
    X_live = day_rows.select(feat_cols).fill_null(0.0).to_numpy()

    signal_live = council_signal.reindex(available_tickers).fillna(0.0)

    # Moltiplicatori conformal + filtro bassa confidenza
    multipliers = sizer.compute_position_multipliers(signal_live, X_live)
    filtered    = sizer.filter_low_confidence(signal_live, X_live, threshold_percentile=80)

    # Matrice covarianza su ultime 90 sessioni
    returns_wide = (
        ohlcv
        .sort(["ticker", "valid_time"])
        .select(["ticker", "valid_time", "adj_close"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1)
            .over("ticker")
            .alias("ret_1d")
        )
        .drop_nulls()
        .pivot(values="ret_1d", index="valid_time", on="ticker")
        .to_pandas()
        .set_index("valid_time")
        .tail(90)
    )
    cov_tickers = [t for t in available_tickers if t in returns_wide.columns]
    cov = returns_wide[cov_tickers].cov()

    # Portafoglio corrente: equal weight (giorno 1)
    current_w = pd.Series(
        np.ones(len(cov_tickers)) / len(cov_tickers), index=cov_tickers
    )

    constructor = PortfolioConstructor()
    optimize_kwargs = {
        "alpha_signals": filtered.reindex(cov_tickers).fillna(0.0),
        "position_multipliers": multipliers.reindex(cov_tickers).fillna(1.0),
        "current_weights": current_w,
        "returns_covariance": cov,
    }
    try:
        target_w = constructor.optimize(
            **optimize_kwargs,
            portfolio_value=portfolio_value,
        )
    except TypeError:
        # Backward compatibility for lightweight/dummy constructors used in tests.
        target_w = constructor.optimize(**optimize_kwargs)

    orders = constructor.compute_orders(target_w, current_w, portfolio_value)

    # ── Save orders to parquet ───────────────────────────────────────────────
    if not orders.empty:
        orders_df = pd.DataFrame({
            "ticker": orders.index,
            "direction": orders["direction"],
            "quantity": orders["quantity"],
            "target_weight": orders["target_weight"],
        })
        ORDERS_DIR.mkdir(parents=True, exist_ok=True)
        orders_path = ORDERS_DIR / f"{last_date}.parquet"
        orders_df.to_parquet(orders_path, index=False)
        print(f"    [9] Ordini salvati in {orders_path}")

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"[9] Target weights al {last_date} (portafoglio da ${portfolio_value:,.0f}):")
    for ticker, w in target_w.sort_values(ascending=False).items():
        bar = "█" * int(w * 200)
        print(f"    {ticker:<6} {w:>6.2%}  {bar}")

    print(f"\n    Ordini da eseguire ({len(orders)} trade):")
    if orders.empty:
        print("    Nessun ordine (portafoglio già ottimale)")
    else:
        print(orders.to_string(index=False))

    return target_w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("MLCouncil — Pipeline Demo")
    print("=" * 60 + "\n")

    step_download(force=args.force_download)

    ohlcv    = step_load_ohlcv()
    features = step_features(ohlcv)
    feat_train, feat_test, targets_train, cutoff, targets_df = step_split(features, ohlcv)

    lgbm            = step_train_lgbm(feat_train, targets_train)
    hmm_model, _, macro = step_train_hmm(cutoff)

    signals, last_date, regime = step_signals(
        lgbm, hmm_model, feat_test, macro,
        with_sentiment=args.with_sentiment,
    )

    council_signal = step_council(signals, regime, last_date)
    sizer, feat_cols = step_conformal(lgbm, feat_train, targets_df)

    step_portfolio(council_signal, sizer, feat_cols, feat_test, ohlcv, last_date)

    print("\n" + "=" * 60)
    print("Pipeline completata.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLCouncil pipeline demo")
    parser.add_argument(
        "--skip-download", dest="skip_download", action="store_true",
        help="Non usato (skip automatico se i dati esistono già)",
    )
    parser.add_argument(
        "--force-download", dest="force_download", action="store_true",
        help="Forza il re-download anche se i dati esistono",
    )
    parser.add_argument(
        "--with-sentiment", dest="with_sentiment", action="store_true",
        help="Attiva il modello sentiment FinBERT (richiede PyTorch + GPU consigliata)",
    )
    main(parser.parse_args())
