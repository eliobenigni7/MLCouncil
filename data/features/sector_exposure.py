"""Sector exposure computation for portfolio constraints."""

from __future__ import annotations

import pandas as pd

SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary",
    "META": "Technology",
    "NVDA": "Technology",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "XOM": "Energy",
    "WMT": "Consumer Staples",
    "PG": "Consumer Staples",
    "ETSY": "Consumer Discretionary",
    "DOCU": "Technology",
    "UBER": "Consumer Discretionary",
    "ABNB": "Consumer Discretionary",
    "PLTR": "Technology",
    "SNOW": "Technology",
    "CRWD": "Technology",
    "NET": "Technology",
    "SQ": "Financials",
    "SHOP": "Technology",
    "FVRR": "Technology",
    "ROKU": "Communication Services",
    "DDOG": "Technology",
}

UNIQUE_SECTORS = sorted(set(SECTOR_MAP.values()))


def compute_sector_exposures(weights: pd.Series) -> pd.Series:
    sector_weights = pd.Series(0.0, index=UNIQUE_SECTORS, dtype=float)
    for ticker, w in weights.items():
        sector = SECTOR_MAP.get(ticker, "Other")
        sector_weights[sector] += w
    return sector_weights


def get_ticker_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker, "Other")


def compute_beta_vector(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    import numpy as np

    betas = {}
    for ticker in returns.columns:
        ticker_ret = returns[ticker].dropna()
        mkt_ret = market_returns.reindex(ticker_ret.index).dropna()
        common_idx = ticker_ret.index.intersection(mkt_ret.index)
        if len(common_idx) < 20:
            betas[ticker] = 1.0
            continue

        ticker_aligned = ticker_ret.loc[common_idx].tail(lookback)
        mkt_aligned = mkt_ret.loc[common_idx].tail(lookback)

        cov = np.cov(ticker_aligned, mkt_aligned)[0, 1]
        var = np.var(mkt_aligned)
        betas[ticker] = cov / var if var > 1e-10 else 1.0

    return pd.Series(betas)
