"""Vector store wrapper for SEC filing / transcript retrieval (T2.2 RAG).

Uses Chroma when installed; otherwise an in-memory mock backend suitable for
tests and CPU-only dev machines without GPU or vector DB deps.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_CHROMA_AVAILABLE = False
try:
    import chromadb  # noqa: F401

    _CHROMA_AVAILABLE = True
except ImportError:
    pass


@dataclass
class RetrievedChunk:
    """Single passage returned by similarity search."""

    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class _InMemoryVectorBackend:
    """Deterministic mock store: bag-of-words overlap scoring."""

    def __init__(self) -> None:
        self._docs: dict[str, tuple[str, dict[str, Any]]] = {}

    def upsert(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        self._docs[doc_id] = (text, metadata)

    def query(self, query_text: str, top_k: int) -> list[RetrievedChunk]:
        q_tokens = set(query_text.lower().split())
        if not q_tokens:
            return []

        scored: list[tuple[float, str, str, dict[str, Any]]] = []
        for doc_id, (text, meta) in self._docs.items():
            t_tokens = set(text.lower().split())
            overlap = len(q_tokens & t_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(q_tokens), 1)
            scored.append((score, doc_id, text, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(doc_id=did, text=txt, score=sc, metadata=meta)
            for sc, did, txt, meta in scored[:top_k]
        ]

    def count(self) -> int:
        return len(self._docs)


class VectorStore:
    """Chroma-backed collection with mock fallback.

    Parameters
    ----------
    collection_name:
        Logical collection (e.g. ``sec_filings``).
    persist_directory:
        Chroma persist path; ignored for mock backend.
    """

    def __init__(
        self,
        collection_name: str = "sec_filings",
        persist_directory: str | Path | None = None,
        *,
        force_mock: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = Path(
            persist_directory or os.getenv("MLCOUNCIL_VECTOR_STORE_PATH", "data/vector_store")
        )
        self._backend: str = "mock"
        self._mock = _InMemoryVectorBackend()
        self._collection: Any = None

        use_chroma = (
            not force_mock
            and _CHROMA_AVAILABLE
            and os.getenv("MLCOUNCIL_VECTOR_STORE_MOCK", "").strip().lower()
            not in {"1", "true", "yes", "on"}
        )
        if use_chroma:
            self._init_chroma()
        else:
            self._backend = "mock"

    def _init_chroma(self) -> None:
        import chromadb

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection = client.get_or_create_collection(self.collection_name)
        self._backend = "chroma"

    @property
    def backend(self) -> str:
        return self._backend

    @staticmethod
    def _doc_id(ticker: str, filing_type: str, filed_date: str, chunk_index: int = 0) -> str:
        raw = f"{ticker}|{filing_type}|{filed_date}|{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def upsert_passages(
        self,
        ticker: str,
        filing_type: str,
        filed_date: str,
        passages: list[str],
        *,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """Index text passages for a filing. Returns number of chunks stored."""
        if not passages:
            return 0

        extra = extra_metadata or {}
        if self._backend == "chroma" and self._collection is not None:
            ids: list[str] = []
            docs: list[str] = []
            metas: list[dict[str, Any]] = []
            for i, text in enumerate(passages):
                if not text.strip():
                    continue
                doc_id = self._doc_id(ticker, filing_type, filed_date, i)
                ids.append(doc_id)
                docs.append(text)
                metas.append(
                    {
                        "ticker": ticker,
                        "filing_type": filing_type,
                        "filed_date": filed_date,
                        "chunk_index": i,
                        **extra,
                    }
                )
            if ids:
                self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
            return len(ids)

        n = 0
        for i, text in enumerate(passages):
            if not text.strip():
                continue
            doc_id = self._doc_id(ticker, filing_type, filed_date, i)
            self._mock.upsert(
                doc_id,
                text,
                {
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "filed_date": filed_date,
                    "chunk_index": i,
                    **extra,
                },
            )
            n += 1
        return n

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        ticker: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """Return top-K passages for RAG prompting."""
        if self._backend == "chroma" and self._collection is not None:
            where = {"ticker": ticker} if ticker else None
            result = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
            )
            chunks: list[RetrievedChunk] = []
            ids = (result.get("ids") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            dists = (result.get("distances") or [[]])[0]
            for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
                score = 1.0 / (1.0 + float(dist)) if dist is not None else 1.0
                chunks.append(
                    RetrievedChunk(
                        doc_id=str(doc_id),
                        text=str(text),
                        score=score,
                        metadata=dict(meta or {}),
                    )
                )
            return chunks

        hits = self._mock.query(query, top_k=top_k)
        if ticker:
            hits = [h for h in hits if h.metadata.get("ticker") == ticker]
        return hits[:top_k]

    def count(self) -> int:
        if self._backend == "chroma" and self._collection is not None:
            return int(self._collection.count())
        return self._mock.count()
