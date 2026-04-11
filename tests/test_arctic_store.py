from __future__ import annotations

import importlib
import os
import sys
import types
from datetime import date, datetime, timezone

import pandas as pd
import polars as pl
import pytest


def _load_store_module(monkeypatch):
    class FakeVersionedItem:
        def __init__(self, data):
            self.data = data

    class FakeLibrary:
        def __init__(self):
            self._store = {}

        def write(self, symbol, pd_df, metadata=None):
            self._store.setdefault(symbol, []).append(
                {"data": pd_df.copy(), "metadata": metadata or {}}
            )

        def has_symbol(self, symbol):
            return symbol in self._store

        def read(self, symbol, date_range=None):
            frame = self._store[symbol][-1]["data"].copy()
            if date_range is not None:
                start, end = date_range
                if start is not None:
                    frame = frame.loc[frame.index >= start]
                if end is not None:
                    frame = frame.loc[frame.index <= end]
            return FakeVersionedItem(frame)

        def list_symbols(self):
            return list(self._store.keys())

        def list_versions(self, symbol):
            return list(range(len(self._store.get(symbol, []))))

    class FakeArctic:
        def __init__(self, uri):
            self.uri = uri
            self._library = FakeLibrary()

        def get_library(self, library, create_if_missing=True):
            return self._library

    monkeypatch.setitem(sys.modules, "arcticdb", types.SimpleNamespace(Arctic=FakeArctic))

    import data.store.arctic_store as store_module

    return importlib.reload(store_module)


def test_feature_store_write_adds_transaction_time(monkeypatch):
    store_module = _load_store_module(monkeypatch)
    store = store_module.FeatureStore(uri="lmdb://test/")

    df = pl.DataFrame(
        {
            "valid_time": [date(2026, 4, 7)],
            "feature": [1.5],
        }
    )
    store.write("AAPL", df)

    loaded = store.read("AAPL")
    assert "transaction_time" in loaded.columns
    assert loaded.height == 1


def test_feature_store_read_respects_as_of_transaction_time(monkeypatch):
    store_module = _load_store_module(monkeypatch)
    store = store_module.FeatureStore(uri="lmdb://test/")

    old_ts = datetime(2026, 4, 7, 12, 0, tzinfo=timezone.utc)
    new_ts = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)

    first = pl.DataFrame(
        {
            "valid_time": [date(2026, 4, 7)],
            "feature": [1.0],
            "transaction_time": [old_ts],
        }
    )
    second = pl.DataFrame(
        {
            "valid_time": [date(2026, 4, 8)],
            "feature": [2.0],
            "transaction_time": [new_ts],
        }
    )

    store.write("AAPL", first)
    store.write("AAPL", second)

    current = store.read("AAPL")
    historical = store.read("AAPL", as_of_transaction_time=old_ts)

    assert current.height == 1
    assert historical.height == 0 or historical["transaction_time"].max() <= old_ts


def test_feature_store_list_symbols_strips_prefix(monkeypatch):
    store_module = _load_store_module(monkeypatch)
    store = store_module.FeatureStore(uri="lmdb://test/")

    df = pl.DataFrame(
        {
            "valid_time": [date(2026, 4, 7)],
            "feature": [1.5],
            "transaction_time": [datetime(2026, 4, 7, tzinfo=timezone.utc)],
        }
    )
    store.write("AAPL", df)
    store.write("MSFT", df)

    assert sorted(store.list_symbols()) == ["AAPL", "MSFT"]


def test_feature_store_parquet_backend_writes_via_tmp_then_replace(monkeypatch, tmp_path):
    import data.store.arctic_store as store_module

    store_module = importlib.reload(store_module)
    monkeypatch.setattr(store_module, "_arcticdb_available", False)

    store = store_module.FeatureStore(uri=str(tmp_path), library="features")
    path = store._symbol_path("AAPL")
    replace_calls: list[tuple[str, str]] = []
    original_replace = os.replace

    def spy_replace(src, dst):
        replace_calls.append((str(src), str(dst)))
        assert str(src).endswith(".tmp")
        assert os.path.exists(src)
        return original_replace(src, dst)

    monkeypatch.setattr(store_module.os, "replace", spy_replace)

    df = pl.DataFrame({"valid_time": [date(2026, 4, 7)], "feature": [1.5]})
    store.write("AAPL", df)

    assert path.exists()
    assert not path.with_name(f"{path.name}.tmp").exists()
    assert replace_calls == [(str(path.with_name(f"{path.name}.tmp")), str(path))]


def test_feature_store_parquet_backend_preserves_original_when_replace_fails(
    monkeypatch, tmp_path
):
    import data.store.arctic_store as store_module

    store_module = importlib.reload(store_module)
    monkeypatch.setattr(store_module, "_arcticdb_available", False)

    store = store_module.FeatureStore(uri=str(tmp_path), library="features")
    original = pl.DataFrame({"valid_time": [date(2026, 4, 7)], "feature": [1.0]})
    updated = pl.DataFrame({"valid_time": [date(2026, 4, 8)], "feature": [2.0]})
    path = store._symbol_path("AAPL")
    original.write_parquet(path)

    def fail_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(store_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.write("AAPL", updated)

    restored = pl.read_parquet(path)
    assert restored.to_dicts() == original.to_dicts()
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_feature_store_parquet_backend_read_universe_and_list_versions(
    monkeypatch, tmp_path
):
    import data.store.arctic_store as store_module

    store_module = importlib.reload(store_module)
    monkeypatch.setattr(store_module, "_arcticdb_available", False)

    store = store_module.FeatureStore(uri=str(tmp_path), library="features")
    old_ts = datetime(2026, 4, 7, 9, 0, tzinfo=timezone.utc)
    new_ts = datetime(2026, 4, 8, 9, 0, tzinfo=timezone.utc)

    store.write(
        "AAPL",
        pl.DataFrame(
            {
                "valid_time": [date(2026, 4, 7)],
                "feature": [1.0],
                "transaction_time": [old_ts],
            }
        ),
    )
    store.write(
        "MSFT",
        pl.DataFrame(
            {
                "valid_time": [date(2026, 4, 7)],
                "feature": [2.0],
                "transaction_time": [new_ts],
            }
        ),
    )

    universe = store.read_universe(
        ["AAPL", "MSFT"],
        as_of_date=date(2026, 4, 7),
        as_of_transaction_time=old_ts,
    )
    versions = store.list_versions("AAPL")

    assert sorted(universe["feature"].to_list()) == [1.0]
    assert len(versions) == 1
    assert versions[0]["version"] == 1
    assert "timestamp" in versions[0]
