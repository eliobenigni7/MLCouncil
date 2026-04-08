"""SEC EDGAR filing ingestor for earnings transcripts and filings.

Fetches 8-K, 10-Q, 10-K filings from SEC EDGAR.
Extracts earnings call content, management discussion, and guidance.

Data source: SEC EDGAR (free, no API key required)
Rate limit: Respect 10 requests/second max

Usage:
    from data.ingest.edgar_ingest import EdgarIngestor

    ingester = EdgarIngestor()
    filing = ingester.fetch_filing("AAPL", form_type="10-K", year=2024)
    sections = ingester.split_transcript(filing["text"])
    sentiment = ingester.analyze_sentiment(sections)
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

_ROOT = Path(__file__).parents[2]
CACHE_DIR = _ROOT / "data" / "raw" / "edgar"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CIK_MAP_CACHE = CACHE_DIR / "cik_map.json"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


class EdgarIngestor:
    BASE_URL = "https://data.sec.gov/submissions"

    HEADERS = {
        "User-Agent": "MLCouncil Research mlcouncil@example.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }

    def __init__(self, rate_limit_delay: float = 0.1):
        self.rate_limit_delay = rate_limit_delay
        self._cik_map = self._load_or_fetch_cik_map()

    def _load_or_fetch_cik_map(self) -> dict:
        if CIK_MAP_CACHE.exists():
            try:
                return json.loads(CIK_MAP_CACHE.read_text())
            except Exception:
                pass

        print("Fetching SEC company tickers (one-time, cached)...")
        response = requests.get(
            COMPANY_TICKERS_URL,
            headers=self.headers() if False else self.HEADERS,
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            ticker_to_cik = {}
            for entry in data.values():
                ticker = entry.get("ticker", "").upper()
                cik = str(entry.get("cik_str", "")).zfill(10)
                if ticker and cik:
                    ticker_to_cik[ticker] = cik
            CIK_MAP_CACHE.write_text(json.dumps(ticker_to_cik))
            return ticker_to_cik
        return {}

    @staticmethod
    def headers() -> dict:
        return {
            "User-Agent": "MLCouncil Research mlcouncil@example.com",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }

    def ticker_to_cik(self, ticker: str) -> Optional[str]:
        ticker = ticker.upper()
        if ticker in self._cik_map:
            return self._cik_map[ticker]
        return self._search_cik(ticker)

    def _search_cik(self, ticker: str) -> Optional[str]:
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker={ticker}&type=&dateb=&owner=include&count=1"
        response = requests.get(url, headers=self.HEADERS, timeout=30)
        if response.status_code == 200:
            match = re.search(r"CIK=(\d+)", response.text)
            if match:
                cik = match.group(1).zfill(10)
                self._cik_map[ticker] = cik
                CIK_MAP_CACHE.write_text(json.dumps(self._cik_map))
                return cik
        return None

    def fetch_submissions(self, cik: str) -> dict:
        url = f"{self.BASE_URL}/CIK{cik}.json"
        for attempt in range(3):
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, headers=self.HEADERS, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(1 * (attempt + 1))
            else:
                break
        return {}

    def find_filing(
        self,
        ticker: str,
        form_type: str,
        year: Optional[int] = None,
        quarter: Optional[int] = None,
    ) -> Optional[dict]:
        cik = self.ticker_to_cik(ticker)
        if not cik:
            return None

        submissions = self.fetch_submissions(cik)
        filings = submissions.get("filings", {}).get("recent", {})

        if not filings or "accessionNumber" not in filings:
            return None

        accession_numbers = filings.get("accessionNumber", [])
        filing_dates = filings.get("filingDate", [])
        form_types = filings.get("form", [])
        primary_documents = filings.get("primaryDocument", [])

        for i in range(len(accession_numbers)):
            form = form_types[i]
            if form != form_type:
                continue

            filing_date = filing_dates[i]
            if year and not filing_date.startswith(str(year)):
                continue

            accession = accession_numbers[i].replace("-", "")

            if quarter:
                month = int(filing_date[5:7])
                filing_quarter = (month - 1) // 3 + 1
                if filing_quarter != quarter:
                    continue

            return {
                "ticker": ticker,
                "cik": cik,
                "accession_number": accession_numbers[i],
                "accession_number_raw": accession,
                "filing_date": filing_date,
                "form_type": form,
                "primary_document": primary_documents[i],
            }

        return None

    def fetch_filing_text(self, ticker: str, filing: dict) -> str:
        cik = filing["cik"]
        accession = filing["accession_number_raw"]
        doc = filing["primary_document"]

        base_url = f"https://www.sec.gov/Archives/edgar/full-index/{cik[:4]}/{cik[4:7]}/{cik[7:10]}/{accession[-6:]}/{doc}"

        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
        time.sleep(self.rate_limit_delay)

        response = requests.get(url, headers=self.HEADERS, timeout=30)
        if response.status_code == 200:
            return self._clean_html(response.text)

        fallback_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
        response = requests.get(fallback_url, headers=self.HEADERS, timeout=30)
        if response.status_code == 200:
            return self._clean_html(response.text)

        return ""

    def _clean_html(self, html: str) -> str:
        import html as html_module
        text = html_module.unescape(html)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def split_transcript(self, text: str) -> dict:
        text_upper = text.upper()

        qa_markers = [
            "QUESTION AND ANSWER",
            "Q&A SESSION",
            "OPERATOR:",
            "THE QUESTION-AND-ANSWER SESSION",
        ]

        qa_split_point = len(text)
        for marker in qa_markers:
            pos = text_upper.find(marker)
            if pos != -1 and pos < qa_split_point:
                qa_split_point = pos

        opening = text[:qa_split_point].strip()

        guidance_markers = [
            "OUTLOOK",
            "GUIDANCE",
            "FULL-YEAR OUTLOOK",
            "NEXT QUARTER",
            "FISCAL YEAR OUTLOOK",
            "RAISES",
            "LOWERS",
        ]

        guidance_split_point = qa_split_point
        for marker in guidance_markers:
            pos = text_upper.find(marker, qa_split_point)
            if pos != -1 and pos < guidance_split_point + 5000:
                guidance_split_point = pos

        qa_section = text[qa_split_point:guidance_split_point].strip()
        guidance_section = text[guidance_split_point:].strip()

        return {
            "opening": opening,
            "qa": qa_section,
            "guidance": guidance_section,
            "full_text": text,
        }

    def extract_key_sentences(self, text: str, n: int = 10) -> list[str]:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        return sentences[:n]

    def compute_change_metrics(
        self,
        current_text: str,
        prior_text: Optional[str] = None,
    ) -> dict:
        metrics = {
            "text_length": len(current_text),
            "sentence_count": len(re.split(r"[.!?]+", current_text)),
        }

        if prior_text:
            length_change = (len(current_text) - len(prior_text)) / (len(prior_text) + 1)
            metrics["length_change_pct"] = length_change

            prior_sentiment = self.analyze_sentiment({"full_text": prior_text})
            current_sentiment = self.analyze_sentiment({"full_text": current_text})
            sentiment_change = (
                current_sentiment.get("overall_tone", 0) - prior_sentiment.get("overall_tone", 0)
            )
            metrics["sentiment_change"] = sentiment_change

        return metrics

    def fetch_and_process(
        self,
        ticker: str,
        form_type: str = "8-K",
        year: Optional[int] = None,
        quarter: Optional[int] = None,
    ) -> Optional[dict]:
        filing = self.find_filing(ticker, form_type, year, quarter)
        if not filing:
            return None

        text = self.fetch_filing_text(ticker, filing)
        if not text:
            return None

        sections = self.split_transcript(text)
        sentiment = self.analyze_sentiment(sections)
        metrics = self.compute_change_metrics(text)

        return {
            "ticker": ticker,
            "form_type": form_type,
            "filing_date": filing["filing_date"],
            "accession_number": filing["accession_number"],
            "sections": sections,
            "sentiment": sentiment,
            "metrics": metrics,
            "raw_text": text,
        }

    def analyze_sentiment(self, sections: dict) -> dict:
        try:
            from transformers import pipeline

            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1,
                truncation=True,
                max_length=512,
            )
        except Exception:
            try:
                sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="yiyanghkust/finbert-tone",
                    device=-1,
                    truncation=True,
                    max_length=512,
                )
            except Exception:
                return self._default_sentiment(sections)

        results = {}

        for section_name, section_text in sections.items():
            if not section_text or len(section_text) < 50:
                results[section_name] = {"label": "neutral", "score": 0.5, "tone": 0.0}
                continue

            text_chunks = [section_text[i : i + 512] for i in range(0, min(len(section_text), 2048), 512)]

            sentiments = []
            for chunk in text_chunks[:4]:
                if len(chunk.strip()) > 10:
                    try:
                        result = sentiment_analyzer(chunk[:512])[0]
                        sentiments.append(result)
                    except Exception:
                        continue

            if sentiments:
                positive_count = sum(1 for s in sentiments if s["label"].lower() == "positive")
                negative_count = sum(1 for s in sentiments if s["label"].lower() == "negative")
                avg_score = sum(s["score"] for s in sentiments) / len(sentiments)

                tone = (positive_count - negative_count) / len(sentiments)
                results[section_name] = {
                    "label": "positive" if tone > 0.1 else "negative" if tone < -0.1 else "neutral",
                    "score": avg_score,
                    "tone": tone,
                    "positive_ratio": positive_count / len(sentiments),
                    "negative_ratio": negative_count / len(sentiments),
                }
            else:
                results[section_name] = {"label": "neutral", "score": 0.5, "tone": 0.0}

        all_tones = [r["tone"] for r in results.values() if isinstance(r, dict)]
        results["overall_tone"] = sum(all_tones) / len(all_tones) if all_tones else 0.0

        return results

    def _default_sentiment(self, sections: dict) -> dict:
        return {
            section_name: {"label": "neutral", "score": 0.5, "tone": 0.0}
            for section_name in sections.keys()
        }


class EarningsSignalProcessor:
    def __init__(self, ingester: Optional[EdgarIngestor] = None):
        self.ingester = ingester or EdgarIngestor()

    def compute_earnings_features(self, ticker: str, lookback_quarters: int = 4) -> pd.DataFrame:
        features = []
        current_year = datetime.now().year

        for year_offset in range(lookback_quarters):
            year = current_year - year_offset // 4
            quarter = 4 - (year_offset % 4)

            for form_type in ["8-K", "10-Q", "10-K"]:
                result = self.ingester.fetch_and_process(ticker, form_type, year, quarter)
                if not result:
                    continue

                sentiment = result["sentiment"]

                features.append({
                    "ticker": ticker,
                    "filing_date": result["filing_date"],
                    "form_type": result["form_type"],
                    "overall_tone": sentiment.get("overall_tone", 0),
                    "opening_tone": sentiment.get("opening", {}).get("tone", 0),
                    "qa_tone": sentiment.get("qa", {}).get("tone", 0),
                    "guidance_tone": sentiment.get("guidance", {}).get("tone", 0),
                    "positive_ratio": sentiment.get("opening", {}).get("positive_ratio", 0),
                    "negative_ratio": sentiment.get("opening", {}).get("negative_ratio", 0),
                    "sentiment_change": result["metrics"].get("sentiment_change", 0),
                    "length_change_pct": result["metrics"].get("length_change_pct", 0),
                })
                break

        if not features:
            return pd.DataFrame()

        df = pd.DataFrame(features)

        for col in ["overall_tone", "opening_tone", "qa_tone", "guidance_tone"]:
            df[f"{col}_z"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

        return df

    def compute_event_return(
        self,
        ticker: str,
        filing_date: str,
        prices: pd.Series,
        event_window: tuple = (-1, 3),
    ) -> float:
        try:
            filing_dt = pd.Timestamp(filing_date)
            start_dt = filing_dt + pd.Timedelta(days=event_window[0])
            end_dt = filing_dt + pd.Timedelta(days=event_window[1])

            event_prices = prices[(prices.index >= start_dt) & (prices.index <= end_dt)]
            if len(event_prices) < 2:
                return 0.0

            ret = (event_prices.iloc[-1] / event_prices.iloc[0]) - 1
            return ret
        except Exception:
            return 0.0


def get_earnings_signals(
    ticker: str,
    lookback_quarters: int = 4,
) -> pd.DataFrame:
    ingester = EdgarIngestor()
    processor = EarningsSignalProcessor(ingester)
    return processor.compute_earnings_features(ticker, lookback_quarters)
