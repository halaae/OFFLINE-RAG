"""
reranker.py - Cross-encoder reranking, GPU-optimised.

Auto-detects GPU. Uses best available model based on device.

GPU: cross-encoder/ms-marco-MiniLM-L-6-v2  (best quality, ~50ms on GPU)
CPU: cross-encoder/ms-marco-TinyBERT-L-2-v2 (fastest on CPU, ~300ms)
"""

import logging
import os

log = logging.getLogger("reranker")

# Override via env var: RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_MODEL_GPU = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_CPU = "cross-encoder/ms-marco-TinyBERT-L-2-v2"


def _detect_device() -> str:
    env = os.environ.get("RERANKER_DEVICE", "")
    if env:
        return env
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class CrossEncoderReranker:
    """
    Cross-encoder reranker with automatic GPU/CPU selection.
    Uses best model for the available device.
    """

    def __init__(self, device: str = "auto"):
        self.device = _detect_device() if device == "auto" else device
        self._model = None
        model_env = os.environ.get("RERANKER_MODEL", "")
        if model_env:
            self._model_name = model_env
        else:
            self._model_name = (
                RERANKER_MODEL_GPU if self.device == "cuda"
                else RERANKER_MODEL_CPU
            )

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        log.info("Loading cross-encoder: %s on %s", self._model_name, self.device)
        self._model = CrossEncoder(
            self._model_name,
            device=self.device,
            max_length=512,
        )
        log.info("Cross-encoder ready.")

    def rerank(self, query: str, candidates: list, top_n: int = 5) -> list:
        """
        Score each (query, passage) pair and return top_n.
        GPU: full 512-token passages
        CPU: truncated to 200 words for speed
        """
        if not candidates:
            return []

        self._load()

        max_words = 512 if self.device == "cuda" else 200

        def _trunc(text: str) -> str:
            return " ".join(text.split()[:max_words])

        pairs  = [(query, _trunc(c["text"])) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_n]

    rerank_with_metadata = rerank