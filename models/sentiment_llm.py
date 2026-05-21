"""FinMA / FinGPT LLM sentiment challenger with RAG (T2.2 shadow mode).

Production daily path keeps :class:`models.sentiment.SentimentModel` (FinBERT).
This module scores headlines and filing context via an LLM encoder in
**shadow mode** only — results are logged for walk-forward comparison, not
wired into ``sentiment_signals``.

Hallucination guard: any unparseable LLM output maps to score ``0.0``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger

_SCORE_RE = re.compile(
    r"sentiment\s*[=:]\s*([-+]?(?:\d+\.?\d*|\.\d+))",
    re.IGNORECASE,
)
_JSON_SCORE_RE = re.compile(r'"sentiment(?:_score)?"\s*:\s*([-+]?\d+\.?\d*)', re.IGNORECASE)
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_DEFAULT_MODEL = "FinMA-7B"
_SHADOW_DIR = Path("data/results/shadow_sentiment_llm")


def llm_sentiment_shadow_enabled() -> bool:
    """True when Dagster/scripts should log LLM challenger scores alongside FinBERT."""
    return os.getenv("MLCOUNCIL_LLM_SENTIMENT_SHADOW", "").strip().lower() in _TRUTHY


def use_local_llm_backend() -> bool:
    """Force keyword/mock backend even if GPU packages are present."""
    return os.getenv("MLCOUNCIL_LLM_SENTIMENT_MOCK", "").strip().lower() in _TRUTHY


@dataclass
class LLMScoreResult:
    """Outcome for one scoring request."""

    score: float
    parseable: bool
    backend: str
    raw_response: str = ""
    retrieved_chunks: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_sentiment_score(text: str) -> Optional[float]:
    """Parse LLM output into [-1, +1]. Returns None if unparseable."""
    if not text or not str(text).strip():
        return None

    body = str(text).strip()
    for pattern in (_SCORE_RE, _JSON_SCORE_RE):
        m = pattern.search(body)
        if m:
            try:
                val = float(m.group(1))
                return max(-1.0, min(1.0, val))
            except ValueError:
                continue

    # Bare float on last non-empty line
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if lines:
        try:
            val = float(lines[-1])
            if -1.0 <= val <= 1.0:
                return val
        except ValueError:
            pass

    return None


def apply_hallucination_guard(parsed: Optional[float]) -> tuple[float, bool]:
    """Map unparseable output to 0.0 per T2.2 gating."""
    if parsed is None:
        return 0.0, False
    return float(parsed), True


class _KeywordMockBackend:
    """CPU-only stub: lexicon sentiment when no GPU / llama-cpp."""

    _POS = frozenset(
        {
            "beat", "beats", "growth", "profit", "surge", "upgrade", "bullish",
            "strong", "record", "outperform", "raises", "raised",
        }
    )
    _NEG = frozenset(
        {
            "miss", "misses", "loss", "decline", "downgrade", "bearish", "weak",
            "cut", "cuts", "lawsuit", "probe", "warning",
        }
    )

    def generate(self, prompt: str) -> str:
        tokens = re.findall(r"[a-z]+", prompt.lower())
        if not tokens:
            return "sentiment=0.0"
        pos = sum(1 for t in tokens if t in self._POS)
        neg = sum(1 for t in tokens if t in self._NEG)
        if pos == neg == 0:
            return "sentiment=0.0"
        raw = (pos - neg) / max(pos + neg, 1)
        score = max(-1.0, min(1.0, raw))
        return f"sentiment={score:.4f}"


class LLMSentimentScorer:
    """Finance-tuned LLM sentiment with optional RAG over SEC filings.

    Parameters
    ----------
    model:
        Logical model id (``FinMA-7B``, ``FinGPT``, etc.).
    vector_store:
        Optional :class:`data.store.vector_store.VectorStore` for retrieval.
    llm_fn:
        Injectable ``(prompt) -> str`` for tests; when None uses mock or
        llama-cpp if ``MLCOUNCIL_LLM_GGUF_PATH`` is set.
    top_k:
        RAG passages to inject into the prompt.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        vector_store: Any = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        top_k: int = 3,
    ) -> None:
        self.model = model
        self._vector_store = vector_store
        self._top_k = top_k
        self._llm_fn = llm_fn
        self._backend = "mock"

    def _resolve_backend(self) -> Callable[[str], str]:
        if self._llm_fn is not None:
            self._backend = "injected"
            return self._llm_fn

        if use_local_llm_backend():
            self._backend = "keyword_mock"
            return _KeywordMockBackend().generate

        gguf = os.getenv("MLCOUNCIL_LLM_GGUF_PATH", "").strip()
        if gguf and Path(gguf).is_file():
            try:
                from llama_cpp import Llama

                llama = Llama(model_path=gguf, n_ctx=2048, verbose=False)

                def _llama_generate(prompt: str) -> str:
                    out = llama(
                        prompt,
                        max_tokens=64,
                        temperature=0.0,
                    )
                    return out["choices"][0]["text"]

                self._backend = "llama_cpp"
                return _llama_generate
            except Exception as exc:
                logger.warning("llama-cpp load failed ({}); using keyword mock", exc)

        self._backend = "keyword_mock"
        return _KeywordMockBackend().generate

    def build_rag_prompt(
        self,
        headline: str,
        *,
        ticker: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> str:
        """Assemble prompt with top-K retrieved SEC passages."""
        passages: list[str] = []
        if self._vector_store is not None:
            chunks = self._vector_store.retrieve(
                headline,
                top_k=self._top_k,
                ticker=ticker,
            )
            passages = [c.text for c in chunks if c.text]

        if extra_context:
            passages.append(extra_context)

        ctx_block = ""
        if passages:
            joined = "\n---\n".join(passages[: self._top_k])
            ctx_block = f"Context from SEC filings:\n{joined}\n\n"

        return (
            f"You are a financial sentiment analyst ({self.model}).\n"
            f"{ctx_block}"
            f"Headline: {headline}\n"
            "Reply with exactly one line: sentiment=<float between -1 and 1>\n"
            "Negative=-1, neutral=0, positive=+1."
        )

    def score_text(
        self,
        text: str,
        *,
        ticker: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> LLMScoreResult:
        """Score a single headline or passage."""
        generate = self._resolve_backend()
        prompt = self.build_rag_prompt(
            text,
            ticker=ticker,
            extra_context=extra_context,
        )
        raw = generate(prompt)
        parsed = parse_sentiment_score(raw)
        score, parseable = apply_hallucination_guard(parsed)
        chunks = 0
        if self._vector_store is not None:
            chunks = len(
                self._vector_store.retrieve(text, top_k=self._top_k, ticker=ticker)
            )
        return LLMScoreResult(
            score=score,
            parseable=parseable,
            backend=self._backend,
            raw_response=raw,
            retrieved_chunks=chunks,
            metadata={"model": self.model, "ticker": ticker},
        )

    def score_headlines(
        self,
        headlines: list[str],
        *,
        ticker: Optional[str] = None,
    ) -> list[float]:
        """Batch score; unparseable → 0.0."""
        return [self.score_text(h, ticker=ticker).score for h in headlines]

    def predict_shadow(
        self,
        features: pd.DataFrame,
        *,
        headline_col: str = "title",
        ticker_col: str = "ticker",
    ) -> pd.Series:
        """Cross-sectional z-scored shadow signals (mirrors FinBERT predict shape)."""
        if features.empty:
            return pd.Series(dtype=float, name="llm_sentiment_shadow")

        scores: dict[str, list[float]] = {}
        for row in features.to_dict("records"):
            ticker = str(row.get(ticker_col, ""))
            text = str(row.get(headline_col, "") or "")
            if not text.strip():
                continue
            s = self.score_text(text, ticker=ticker or None).score
            scores.setdefault(ticker, []).append(s)

        agg = {t: sum(v) / len(v) for t, v in scores.items() if v}
        raw = pd.Series(agg, dtype=float)
        if len(raw) < 2:
            return raw.rename("llm_sentiment_shadow")

        std = raw.std()
        if std > 1e-9:
            raw = (raw - raw.mean()) / std
        else:
            raw = pd.Series(0.0, index=raw.index)
        raw.name = "llm_sentiment_shadow"
        return raw


def log_shadow_scores(
    partition_date: str,
    finbert_scores: pd.Series,
    llm_scores: pd.Series,
    *,
    output_dir: Path | str | None = None,
) -> Path:
    """Persist side-by-side FinBERT vs LLM scores for offline IC studies."""
    out_dir = Path(output_dir or _SHADOW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{partition_date}.parquet"

    tickers = sorted(set(finbert_scores.index) | set(llm_scores.index))
    df = pd.DataFrame(
        {
            "ticker": tickers,
            "partition_date": partition_date,
            "finbert_score": [float(finbert_scores.get(t, 0.0)) for t in tickers],
            "llm_score": [float(llm_scores.get(t, 0.0)) for t in tickers],
        }
    )
    df.to_parquet(path, index=False)
    logger.info("shadow LLM sentiment logged → {}", path)
    return path


def run_shadow_comparison(
    news_df: pd.DataFrame,
    finbert_series: pd.Series,
    *,
    scorer: Optional[LLMSentimentScorer] = None,
    partition_date: str,
) -> dict[str, Any]:
    """Score news with LLM challenger and log comparison parquet."""
    scorer = scorer or LLMSentimentScorer()
    llm_series = scorer.predict_shadow(news_df)
    path = log_shadow_scores(partition_date, finbert_series, llm_series)
    return {
        "shadow_path": str(path),
        "llm_backend": scorer._backend,
        "ticker_count": len(llm_series),
    }
