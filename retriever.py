"""
retriever.py - Hybrid dense + sparse retrieval with RRF fusion.
GPU-optimised: embedding model runs on CUDA if available.
"""

import logging
import os
import pickle
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger("retriever")

EMBED_MODEL      = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_STORE_DIR = Path("vector_store")
DB_PATH          = "metadata.db"
RRF_K            = 60
HNSW_EF_SEARCH   = 128


def _detect_device() -> str:
    env = os.environ.get("RETRIEVER_DEVICE", "")
    if env:
        return env
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class HybridRetriever:
    def __init__(self, device: str = "auto"):
        self.device = _detect_device() if device == "auto" else device
        self._faiss_index = None
        self._doc_map     = []
        self._bm25_data   = {}
        self._conn        = None
        self.embed_model  = None
        self._load()

    def _load(self):
        faiss_path  = VECTOR_STORE_DIR / "faiss.index"
        docmap_path = VECTOR_STORE_DIR / "doc_map.pkl"
        bm25_path   = VECTOR_STORE_DIR / "bm25_index.pkl"

        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {faiss_path}. Run build_index.py first."
            )

        log.info("Loading FAISS index...")
        self._faiss_index = faiss.read_index(str(faiss_path))
        self._faiss_index.hnsw.efSearch = HNSW_EF_SEARCH

        with open(docmap_path, "rb") as f:
            self._doc_map = pickle.load(f)

        log.info("Loading BM25 index...")
        with open(bm25_path, "rb") as f:
            self._bm25_data = pickle.load(f)

        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        log.info("Loading embedding model: %s on %s", EMBED_MODEL, self.device)
        self.embed_model = SentenceTransformer(EMBED_MODEL, device=self.device)
        log.info("HybridRetriever ready (%d vectors).", self._faiss_index.ntotal)

    def _dense_search(self, q_emb: np.ndarray, top_k: int) -> list:
        vec = q_emb.reshape(1, -1).astype(np.float32)
        scores, indices = self._faiss_index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._doc_map):
                results.append((self._doc_map[idx], float(score)))
        return results

    def _sparse_search(self, query: str, top_k: int) -> list:
        bm25      = self._bm25_data["bm25"]
        chunk_ids = self._bm25_data["chunk_ids"]
        tokens    = query.lower().split()
        scores    = bm25.get_scores(tokens)
        top_idx   = np.argsort(scores)[::-1][:top_k]
        return [(chunk_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    @staticmethod
    def _rrf_fuse(dense: list, sparse: list, k: int = RRF_K) -> list:
        rrf = {}
        for rank, (cid, _) in enumerate(dense, 1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
        for rank, (cid, _) in enumerate(sparse, 1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    def _enrich(self, chunk_ids: list, rrf_scores: dict) -> list:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"""SELECT c.id AS chunk_id, c.chunk_index, c.content AS text,
                       d.filename, d.source_path, d.form_type, d.filing_date
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE c.id IN ({placeholders})""",
            chunk_ids,
        ).fetchall()
        order = {cid: i for i, cid in enumerate(chunk_ids)}
        rows  = sorted(rows, key=lambda r: order.get(r["chunk_id"], 9999))
        return [{
            "chunk_id":    r["chunk_id"],
            "chunk_index": r["chunk_index"],
            "text":        r["text"],
            "filename":    r["filename"],
            "source_path": r["source_path"],
            "form_type":   r["form_type"],
            "filing_date": r["filing_date"],
            "rrf_score":   rrf_scores.get(r["chunk_id"], 0.0),
        } for r in rows]

    def retrieve(self, query: str, top_k: int = 10) -> tuple:
        q_emb = self.embed_model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        fetch_k     = top_k * 3
        dense_hits  = self._dense_search(q_emb, fetch_k)
        sparse_hits = self._sparse_search(query, fetch_k)
        fused       = self._rrf_fuse(dense_hits, sparse_hits)[:top_k]

        chunk_ids  = [cid for cid, _ in fused]
        rrf_scores = dict(fused)
        candidates = self._enrich(chunk_ids, rrf_scores)
        return candidates, q_emb