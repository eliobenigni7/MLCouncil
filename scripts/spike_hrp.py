"""Mini-spike: HRP (López de Prado) vs equal-weight / MV proxy on recent returns."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "data" / "results" / "spike_hrp.json"


def _load_returns(tickers: list[str], lookback: int = 90) -> pd.DataFrame:
    ohlcv_dir = ROOT / "data" / "raw" / "ohlcv"
    frames = []
    for ticker in tickers:
        path = ohlcv_dir / ticker
        if not path.exists():
            continue
        files = sorted(path.glob("*.parquet"))
        if not files:
            continue
        df = pd.read_parquet(files[-1])
        if "adj_close" not in df.columns:
            continue
        s = df.set_index("valid_time")["adj_close"].astype(float).pct_change().dropna()
        frames.append(s.rename(ticker))
    if not frames:
        rng = np.random.default_rng(42)
        idx = pd.bdate_range(end=pd.Timestamp.today(), periods=lookback)
        data = rng.standard_normal((lookback, len(tickers))) * 0.01
        return pd.DataFrame(data, index=idx, columns=tickers)
    out = pd.concat(frames, axis=1).dropna(how="any").tail(lookback)
    return out


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    from council.hrp import hrp_weights_from_covariance

    return hrp_weights_from_covariance(returns.cov())


def main() -> None:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    returns = _load_returns(tickers)
    from council.hrp import covariance_condition_number

    cond = covariance_condition_number(returns.cov())

    hrp_w = hrp_weights(returns)
    ew_w = pd.Series(1.0 / len(tickers), index=tickers)

    sample_dates = returns.index[[0, len(returns) // 2, -1]]
    comparison = []
    for d in sample_dates:
        r = returns.loc[d]
        day = d.date() if hasattr(d, "date") and callable(getattr(d, "date")) else d
        comparison.append(
            {
                "date": str(day),
                "hrp_return": float((hrp_w * r).sum()),
                "ew_return": float((ew_w.reindex(r.index).fillna(0) * r).sum()),
            }
        )

    payload = {
        "tickers": tickers,
        "lookback_days": len(returns),
        "covariance_condition_number": cond,
        "hrp_weights": hrp_w.to_dict(),
        "equal_weight": ew_w.to_dict(),
        "sample_day_comparison": comparison,
        "recommendation": "go" if cond < 1e4 else "no-go",
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
