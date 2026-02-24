"""
compression.py - Query-aware context compression, GPU-optimised.

GPU path: uses sentence embeddings (cosine dedup) for best accuracy
CPU path: uses keyword scoring + Jaccard dedup for speed

Pipeline:
  Step 1 - Split reranked chunks into sentences
  Step 2 - Score by relevance (GPU: cosine sim | CPU: keyword overlap)
  Step 3 - Deduplicate (GPU: cosine threshold | CPU: Jaccard)
  Step 4 - Keep within token budget
  Step 5 - Re-order in document order for coherent LLM input
"""

import logging
import os
import re

import numpy as np

log = logging.getLogger("compression")

RELEVANCE_THRESHOLD  = 0.20   # min cosine sim to keep sentence (GPU path)
DEDUP_COSINE         = 0.85   # cosine dedup threshold (GPU path)
DEDUP_JACCARD        = 0.60   # Jaccard dedup threshold (CPU path)
MIN_SENT_WORDS       = 6
WORDS_PER_TOKEN      = 0.75

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "what","was","is","the","a","an","in","of","for","to","and","or",
    "how","did","does","were","are","when","where","who","which","its",
    "their","by","with","at","from","this","that","these","those","be",
    "has","have","had","will","would","could","should","may","might","do"
}


def _detect_device() -> str:
    env = os.environ.get("COMPRESSOR_DEVICE", "")
    if env:
        return env
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ContextCompressor:
    """
    Dual-path context compressor.
    Automatically uses best algorithm for available hardware.
    """

    def __init__(
        self,
        embed_model=None,
        max_tokens:        int   = 4096,
        compression_ratio: float = 0.50,
    ):
        self.embed_model       = embed_model
        self.max_tokens        = max_tokens
        self.compression_ratio = compression_ratio
        self._max_words        = int(max_tokens * WORDS_PER_TOKEN)
        self._device           = _detect_device()
        log.info("Compressor using %s path", self._device.upper())

    def _split_sentences(self, text: str) -> list:
        parts = _SENT_SPLIT.split(text.strip())
        return [s.strip() for s in parts if len(s.strip().split()) >= MIN_SENT_WORDS]

    # ── GPU path (cosine similarity) ──────────────────────────────────

    def _compress_gpu(self, query: str, chunks: list, q_emb=None) -> tuple:
        texts = []
        meta  = []
        original_words = 0

        for ci, chunk in enumerate(chunks):
            for si, sent in enumerate(self._split_sentences(chunk)):
                texts.append(sent)
                meta.append((ci, si))
                original_words += len(sent.split())

        if not texts:
            return " ".join(chunks), 0.0

        if q_emb is None:
            q_emb = self.embed_model.encode(
                query, normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)

        sent_embs = self.embed_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False,
        ).astype(np.float32)

        q_scores = sent_embs @ q_emb

        scored = sorted(
            [(q_scores[i], meta[i][0], meta[i][1], texts[i], sent_embs[i])
             for i in range(len(texts)) if q_scores[i] >= RELEVANCE_THRESHOLD],
            key=lambda x: x[0], reverse=True
        )

        kept_embs  = []
        kept_sents = []
        total_words = 0

        for q_score, ci, si, text, emb in scored:
            words = len(text.split())
            if total_words + words > self._max_words:
                continue
            is_dup = any(float(np.dot(emb, ke)) >= DEDUP_COSINE for ke in kept_embs)
            if is_dup:
                continue
            kept_embs.append(emb)
            kept_sents.append((ci, si, text))
            total_words += words

        if not kept_sents:
            return chunks[0], 0.0

        kept_sents.sort(key=lambda x: (x[0], x[1]))
        compressed  = " ".join(t for _, _, t in kept_sents)
        kept_words  = sum(len(t.split()) for _, _, t in kept_sents)
        ratio       = max(0.0, 1.0 - kept_words / max(original_words, 1))
        return compressed, ratio

    # ── CPU path (keyword + Jaccard) ──────────────────────────────────

    def _compress_cpu(self, query: str, chunks: list, q_emb=None) -> tuple:
        query_keywords = set(query.lower().split()) - STOPWORDS
        original_words = 0
        all_sents      = []

        for ci, chunk in enumerate(chunks):
            sents = self._split_sentences(chunk)
            n     = len(sents)
            original_words += len(chunk.split())
            chunk_score = 1.0 - (ci / max(len(chunks), 1)) * 0.5

            for si, sent in enumerate(sents):
                word_set = set(w.lower() for w in sent.split())
                pos_score = 1.0 - (si / max(n, 1)) * 0.3
                kw_score  = min(len(query_keywords & word_set) * 0.15, 0.45)
                score     = chunk_score * pos_score + kw_score
                all_sents.append((score, ci, si, sent, word_set))

        if not all_sents:
            return " ".join(chunks), 0.0

        all_sents.sort(key=lambda x: x[0], reverse=True)

        kept       = []
        kept_sets  = []
        total_words = 0

        for score, ci, si, text, word_set in all_sents:
            words = len(text.split())
            if total_words + words > self._max_words:
                continue
            is_dup = any(_jaccard(word_set, ks) >= DEDUP_JACCARD for ks in kept_sets)
            if is_dup:
                continue
            kept.append((ci, si, text))
            kept_sets.append(word_set)
            total_words += words

        if not kept:
            return chunks[0], 0.0

        kept.sort(key=lambda x: (x[0], x[1]))
        compressed  = " ".join(t for _, _, t in kept)
        kept_words  = sum(len(t.split()) for _, _, t in kept)
        ratio       = max(0.0, 1.0 - kept_words / max(original_words, 1))
        return compressed, ratio

    # ── Public API ─────────────────────────────────────────────────────

    def compress(self, query: str, chunks: list, query_embedding=None) -> tuple:
        if not chunks:
            return "", 0.0

        if self._device == "cuda" and self.embed_model is not None:
            compressed, ratio = self._compress_gpu(query, chunks, query_embedding)
        else:
            compressed, ratio = self._compress_cpu(query, chunks, query_embedding)

        log.info("Compressed %.0f%% | device=%s", ratio * 100, self._device)
        return compressed, ratio