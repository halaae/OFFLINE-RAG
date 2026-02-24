"""
main.py — Interactive RAG query interface.

Run after build_index.py has populated the vector store:
    python main.py

The system will prompt:
    Enter your question:

For each question it will:
  1. Check the cache (L1 LRU + L2 disk)
  2. Hybrid retrieve top_k=10 candidates
  3. Cross-encoder rerank → top 5
  4. Compress context (40–60% reduction)
  5. Generate answer via local GGUF LLM
  6. Print answer + citations to terminal
  7. Append answer + sources to output.txt
  8. Log metrics

Special commands:
  metrics  — print runtime stats
  clear    — clear the screen
  quit / exit / q — exit

Press Ctrl+C to exit at any time.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — terminal + file
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Fix Windows CP1252 terminal encoding — use UTF-8 stream
import io
_safe_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") \
    if hasattr(sys.stdout, "buffer") else sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(_safe_stdout),
        logging.FileHandler("rag_system.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")

OUTPUT_FILE = Path("output.txt")


# ---------------------------------------------------------------------------
# Import pipeline components
# ---------------------------------------------------------------------------

from cache       import HybridCache
from compression import ContextCompressor
from llm         import LocalLLM
from reranker    import CrossEncoderReranker
from retriever   import HybridRetriever


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class Metrics:
    """Lightweight in-process metrics collector."""

    __slots__ = (
        "query_count", "cache_hits",
        "total_latency", "retrieval_ms", "rerank_ms", "compress_ms", "llm_ms",
    )

    def __init__(self):
        self.query_count   = 0
        self.cache_hits    = 0
        self.total_latency = 0.0
        self.retrieval_ms  = 0.0
        self.rerank_ms     = 0.0
        self.compress_ms   = 0.0
        self.llm_ms        = 0.0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, getattr(self, k) + v)

    def report(self) -> dict:
        n = max(self.query_count, 1)
        return {
            "total_queries":    self.query_count,
            "cache_hits":       self.cache_hits,
            "avg_total_ms":     round(self.total_latency / n, 1),
            "avg_retrieval_ms": round(self.retrieval_ms  / n, 1),
            "avg_rerank_ms":    round(self.rerank_ms     / n, 1),
            "avg_compress_ms":  round(self.compress_ms   / n, 1),
            "avg_llm_ms":       round(self.llm_ms        / n, 1),
        }


_metrics = Metrics()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_SEP  = "─" * 70
_SEP2 = "═" * 70


def format_answer(
    question:   str,
    answer:     str,
    sources:    list[dict],
    latency_ms: float,
    cached:     bool = False,
) -> str:
    """Produce a formatted answer block for terminal + file output."""
    lines = [
        "",
        _SEP,
        f"  Q: {question}",
        _SEP,
        "",
        answer,
        "",
        "  Sources:",
    ]
    for i, src in enumerate(sources, 1):
        score = src.get("rerank_score", src.get("rrf_score", 0.0))
        date  = f"  {src.get('filing_date', '')}" if src.get("filing_date") else ""
        lines.append(
            f"  [{i}] {src['filename']}{date}  "
            f"(chunk #{src['chunk_index']}, score={score:.4f})"
        )
    cached_tag = "  [CACHED]" if cached else ""
    lines += ["", f"  Latency: {latency_ms:.0f} ms{cached_tag}", _SEP, ""]
    return "\n".join(lines)


def save_to_file(text: str, path: Path = OUTPUT_FILE):
    """Append a result block to output.txt with timestamp header."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n=== {datetime.now().isoformat()} ===\n")
        f.write(text)
        f.write("\n")


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Orchestrates retrieval → reranking → compression → generation.

    All heavy components are loaded lazily on the first query so that
    startup is instant when doing component testing.
    Set eager=True to pre-load everything at init.
    """

    def __init__(
        self,
        top_k:        int  = 10,
        rerank_top_n: int  = 5,
        device:       str  = "cpu",
        eager:        bool = False,
    ):
        self.top_k        = top_k
        self.rerank_top_n = rerank_top_n
        self.device       = device

        self.cache      = HybridCache(cache_dir="cache/disk")
        self.retriever  = None
        self.reranker   = None
        self.compressor = None
        self.llm        = None

        if eager:
            self._load_components()

    def _load_components(self):
        """Initialise all heavy components (called lazily)."""
        if self.retriever is None:
            self.retriever = HybridRetriever(device=self.device)

        if self.reranker is None:
            self.reranker = CrossEncoderReranker(device=self.device)

        if self.compressor is None:
            # Share the embedding model — avoids a duplicate 90 MB RAM load.
            self.compressor = ContextCompressor(
                embed_model       = self.retriever.embed_model,
                max_tokens        = 2048,
                compression_ratio = 0.50,
            )

        if self.llm is None:
            self.llm = LocalLLM()

    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """
        Execute the full RAG pipeline for *question*.

        Returns a dict with:
            answer, sources, latency_ms, retrieval_ms, rerank_ms,
            compress_ms, llm_ms, compression_ratio, cached
        """
        t_total = time.perf_counter()
        _metrics.query_count += 1

        # ── 1. Cache check ─────────────────────────────────────────────
        cached = self.cache.get(question)
        if cached is not None:
            log.info("Cache HIT: %s", question[:60])
            _metrics.cache_hits += 1
            cached["cached"] = True
            return cached

        # ── 2. Lazy-load components ────────────────────────────────────
        self._load_components()

        # ── 3. Hybrid retrieval ────────────────────────────────────────
        t0 = time.perf_counter()
        candidates, q_emb = self.retriever.retrieve(question, top_k=self.top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        _metrics.add(retrieval_ms=retrieval_ms)
        log.info("Retrieved %d candidates in %.0f ms", len(candidates), retrieval_ms)

        if not candidates:
            elapsed = (time.perf_counter() - t_total) * 1000
            return {
                "answer": (
                    "No relevant documents found. "
                    "Add documents to data/ and re-run build_index.py."
                ),
                "sources":           [],
                "latency_ms":        elapsed,
                "retrieval_ms":      retrieval_ms,
                "rerank_ms":         0.0,
                "compress_ms":       0.0,
                "llm_ms":            0.0,
                "compression_ratio": 0.0,
                "cached":            False,
            }

        # ── 4. Cross-encoder reranking ────────────────────────────────
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(question, candidates, top_n=self.rerank_top_n)
        rerank_ms = (time.perf_counter() - t0) * 1000
        _metrics.add(rerank_ms=rerank_ms)
        log.info("Reranked to top %d in %.0f ms", len(reranked), rerank_ms)

        # ── 5. Context compression ────────────────────────────────────
        t0 = time.perf_counter()
        chunks = [r["text"] for r in reranked]
        compressed_ctx, compression_ratio = self.compressor.compress(
            question, chunks, query_embedding=q_emb
        )
        compress_ms = (time.perf_counter() - t0) * 1000
        _metrics.add(compress_ms=compress_ms)
        log.info(
            "Context compressed %.1f%% in %.0f ms",
            compression_ratio * 100, compress_ms,
        )

        # ── 6. LLM generation ─────────────────────────────────────────
        llm_result = self.llm.generate(
            question = question,
            context  = compressed_ctx,
            sources  = reranked,
        )
        llm_ms = llm_result["latency_ms"]
        _metrics.add(llm_ms=llm_ms)

        # ── 7. Assemble result ─────────────────────────────────────────
        total_ms = (time.perf_counter() - t_total) * 1000
        _metrics.add(total_latency=total_ms)

        result = {
            "answer":            llm_result["answer"],
            "sources":           reranked,
            "latency_ms":        total_ms,
            "retrieval_ms":      retrieval_ms,
            "rerank_ms":         rerank_ms,
            "compress_ms":       compress_ms,
            "llm_ms":            llm_ms,
            "compression_ratio": compression_ratio,
            "cached":            False,
        }

        # ── 8. Cache result ────────────────────────────────────────────
        self.cache.set(question, result)

        return result

    def close(self):
        self.cache.close()
        if self.llm:
            self.llm.unload()


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def main():
    print()
    print(_SEP2)
    print("  Offline RAG System — Production Grade")
    print("  Commands: 'metrics' | 'clear' | 'quit'")
    print(_SEP2)

    OUTPUT_FILE.touch(exist_ok=True)
    Path("cache/disk").mkdir(parents=True, exist_ok=True)

    pipeline = RAGPipeline(top_k=10, rerank_top_n=5)

    try:
        while True:
            try:
                print()
                question = input("  Enter your question: ").strip()
            except EOFError:
                break

            if not question:
                continue

            if question.lower() in {"quit", "exit", "q"}:
                break

            if question.lower() == "metrics":
                stats = _metrics.report() | pipeline.cache.stats()
                print(json.dumps(stats, indent=2))
                continue

            if question.lower() == "clear":
                print("\033[2J\033[H", end="")
                continue

            print("\n  ⏳ Processing...\n")

            try:
                result = pipeline.query(question)
            except FileNotFoundError as exc:
                print(f"\n  ❌ {exc}\n")
                continue
            except Exception as exc:
                log.exception("Pipeline error")
                print(f"\n  ❌ Unexpected error: {exc}\n")
                continue

            formatted = format_answer(
                question   = question,
                answer     = result["answer"],
                sources    = result["sources"],
                latency_ms = result["latency_ms"],
                cached     = result["cached"],
            )

            print(formatted)
            save_to_file(formatted)
            print(f"  (Answer saved to {OUTPUT_FILE})")

    except KeyboardInterrupt:
        print("\n\n  Interrupted. Goodbye!\n")

    finally:
        pipeline.close()
        log.info("Final metrics: %s", json.dumps(_metrics.report()))
        print("\n  Session metrics logged to rag_system.log")


if __name__ == "__main__":
    main()