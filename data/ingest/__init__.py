from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "download_universe": ("data.ingest.market_data", "download_universe"),
    "download_daily": ("data.ingest.market_data", "download_daily"),
    "download_fundamentals": ("data.ingest.fundamentals", "download_fundamentals"),
    "download_news": ("data.ingest.news", "download_news"),
    "download_macro": ("data.ingest.macro", "download_macro"),
    "clean_headline": ("data.ingest.news_processor", "clean_headline"),
    "deduplicate_news": ("data.ingest.news_processor", "deduplicate_news"),
    "assign_source_weight": ("data.ingest.news_processor", "assign_source_weight"),
    "SentimentCache": ("data.ingest.news_processor", "SentimentCache"),
}

__all__ = [
    "download_universe",
    "download_daily",
    "download_fundamentals",
    "download_news",
    "download_macro",
    "clean_headline",
    "deduplicate_news",
    "assign_source_weight",
    "SentimentCache",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
