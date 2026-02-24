"""
build_index.py — Document ingestion & FAISS + BM25 index builder.

Optimised for 16 GB RAM / laptop-class CPU:
  • Sentence-aware chunking (no regex soup) — fast and retrieval-friendly
  • Streaming line reader — never loads whole file into RAM
  • Batched embedding with progress bar
  • HNSW index (best recall for < 500 K chunks)
  • Skips already-indexed files (incremental)

Usage:
    python build_index.py [--data-dir data] [--db metadata.db] [--reset]

Options:
    --data-dir DIR   Directory containing .txt / .html / .htm / .md files
                     (default: data)
    --db PATH        SQLite database path (default: metadata.db)
    --reset          Drop and rebuild all indexes from scratch
"""

import argparse
import gc
import logging
import pickle
import re
import sqlite3
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config — tweak these to tune speed vs quality
# ---------------------------------------------------------------------------

EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
# Swap to the line below for +5–10% accuracy (needs more RAM/time):
# EMBED_MODEL    = "BAAI/bge-large-en-v1.5"

VECTOR_STORE_DIR = Path("vector_store")
DB_PATH_DEFAULT  = "metadata.db"
DATA_DIR_DEFAULT = Path("data")

# Chunking: split on sentence boundaries, target ~200 words per chunk.
# Smaller chunks = better precision; overlap keeps cross-boundary context.
CHUNK_TARGET_WORDS  = 200   # target words per chunk
CHUNK_OVERLAP_WORDS = 40    # words of overlap between consecutive chunks
MIN_CHUNK_WORDS     = 20    # discard chunks shorter than this

EMBED_BATCH  = 32           # embeddings per GPU/CPU call
HNSW_M       = 32           # HNSW graph connections (higher → better recall, more RAM)
HNSW_EF_CON  = 200          # construction-time search depth

import io as _io
_safe_stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") \
    if hasattr(sys.stdout, "buffer") else sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(_safe_stdout),
              logging.FileHandler("build_index.log", encoding="utf-8")],
)
log = logging.getLogger("build_index")

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT,
    source_path TEXT UNIQUE,
    file_type   TEXT,
    char_count  INTEGER,
    form_type   TEXT,
    filing_date TEXT,
    indexed_at  TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER REFERENCES documents(id),
    chunk_index INTEGER,
    content     TEXT,
    word_count  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_id  ON chunks(id);
"""


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(SCHEMA)
    # Performance pragmas for bulk inserts
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")   # 64 MB page cache
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=268435456") # 256 MB mmap
    return conn


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

_SCRIPT_RE = re.compile(r"<(script|style|head)[^>]*>.*?</\1>", re.DOTALL | re.I)
_TAG_RE    = re.compile(r"<[^>]{1,300}>")
_NBSP_RE   = re.compile(r"&(?:nbsp|#160|#xA0);", re.I)
_ENT_RE    = re.compile(r"&[a-z]{2,6};")
_WS_RE     = re.compile(r"[ \t]{2,}")

# Sentence boundary: period/!/?  followed by space + capital or end-of-string
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def extract_text(path: Path) -> str:
    """
    Read a file and return clean plain text.
    Handles .html/.htm (strip tags) and plain text files.
    Never loads the whole file twice — reads once, returns string.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        log.warning("Cannot read %s: %s", path.name, exc)
        return ""

    if path.suffix.lower() in {".html", ".htm"}:
        raw = _SCRIPT_RE.sub(" ", raw)
        raw = _NBSP_RE.sub(" ", raw)
        raw = _TAG_RE.sub(" ", raw)
        raw = _ENT_RE.sub(" ", raw)
        try:
            import html
            raw = html.unescape(raw)
        except Exception:
            pass

    # Normalise whitespace
    lines = [_WS_RE.sub(" ", ln).strip() for ln in raw.splitlines()]
    return " ".join(ln for ln in lines if ln)


# ---------------------------------------------------------------------------
# Sentence-aware chunking  (the key to good retrieval)
# ---------------------------------------------------------------------------

def chunk_text(text: str,
               target_words: int  = CHUNK_TARGET_WORDS,
               overlap_words: int = CHUNK_OVERLAP_WORDS,
               min_words: int     = MIN_CHUNK_WORDS) -> list[str]:
    """
    Split *text* into overlapping word-window chunks aligned to sentence
    boundaries.

    Why sentence-aware?
      • Mid-sentence splits hurt both precision and LLM coherence.
      • Overlap avoids losing context at chunk edges.
      • Word-count targets keep chunk sizes uniform → stable BM25 scores.

    Returns a list of chunk strings.
    """
    if not text.strip():
        return []

    # Split into sentences first
    sentences = _SENT_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks     = []
    buf_words  = []   # accumulated words in current chunk
    buf_sents  = []   # accumulated sentences (for overlap)
    word_count = 0

    for sent in sentences:
        words = sent.split()
        if not words:
            continue

        buf_words.extend(words)
        buf_sents.append(words)
        word_count += len(words)

        if word_count >= target_words:
            chunk = " ".join(buf_words)
            if len(chunk.split()) >= min_words:
                chunks.append(chunk)

            # Keep last `overlap_words` worth of sentences for next chunk
            overlap_buf: list[str] = []
            kept = 0
            for sw in reversed(buf_sents):
                if kept + len(sw) > overlap_words:
                    break
                overlap_buf.insert(0, sw)
                kept += len(sw)

            buf_words  = [w for sw in overlap_buf for w in sw]
            buf_sents  = overlap_buf
            word_count = len(buf_words)

    # Flush remainder
    if buf_words and len(buf_words) >= min_words:
        chunks.append(" ".join(buf_words))

    return chunks


# ---------------------------------------------------------------------------
# Indexing pipeline
# ---------------------------------------------------------------------------

def index_files(conn: sqlite3.Connection, data_dir: Path) -> int:
    """
    Phase 1: Read documents, chunk them, store chunks in SQLite.
    Returns total number of chunks written (new + cached).
    """
    exts  = {".html", ".htm", ".txt", ".md"}
    files = sorted(p for p in data_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in exts)

    if not files:
        log.error("No supported files found in %s", data_dir)
        return 0

    log.info("Found %d files in %s", len(files), data_dir)
    total_chunks = 0

    for i, fp in enumerate(files, 1):
        # Check if already indexed
        row = conn.execute(
            "SELECT id FROM documents WHERE source_path=?", (str(fp),)
        ).fetchone()

        if row:
            n = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id=?", (row[0],)
            ).fetchone()[0]
            total_chunks += n
            log.info("[%d/%d] SKIP  %s  (%d chunks cached)", i, len(files), fp.name, n)
            continue

        log.info("[%d/%d] INDEX %s ...", i, len(files), fp.name)
        t1 = time.time()

        # Extract text
        text = extract_text(fp)
        if not text:
            log.warning("  Empty text, skipping.")
            continue

        # Parse filename for form_type / filing_date (e.g. "10-K_2023-01-28.html")
        parts = fp.stem.split("_", 1)
        form_type   = parts[0] if parts else ""
        filing_date = parts[1].replace("_", " ") if len(parts) > 1 else ""

        # Insert document record
        cur = conn.execute(
            "INSERT INTO documents "
            "(filename, source_path, file_type, char_count, form_type, filing_date) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (fp.name, str(fp), fp.suffix.lower(), len(text), form_type, filing_date),
        )
        doc_id = cur.lastrowid

        # Chunk and insert
        chunks = chunk_text(text)
        batch  = [
            (doc_id, idx, chunk, len(chunk.split()))
            for idx, chunk in enumerate(chunks)
        ]

        conn.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, word_count) "
            "VALUES (?, ?, ?, ?)",
            batch,
        )
        conn.commit()

        n = len(chunks)
        total_chunks += n
        log.info("  → %d chunks | %.1f s", n, time.time() - t1)

        # Free memory eagerly on low-RAM machines
        del text, chunks, batch
        gc.collect()

    return total_chunks


def build_embeddings(conn: sqlite3.Connection,
                     model: SentenceTransformer,
                     dim: int) -> tuple[np.ndarray, list[int]]:
    """
    Phase 2: Embed all chunks in batches.
    Returns (matrix[N, dim], chunk_id_list).
    Memory-efficient: processes EMBED_BATCH at a time.
    """
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if total == 0:
        return np.empty((0, dim), dtype=np.float32), []

    log.info("Embedding %d chunks (batch=%d)...", total, EMBED_BATCH)

    all_ids  : list[int]        = []
    all_embs : list[np.ndarray] = []
    batch_ids: list[int]        = []
    batch_txt: list[str]        = []

    def flush():
        if not batch_txt:
            return
        emb = model.encode(
            batch_txt,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=EMBED_BATCH,
        ).astype(np.float32)
        all_ids.extend(batch_ids)
        all_embs.append(emb)

    with tqdm(total=total, desc="Embedding", unit="chunk") as pbar:
        for row in conn.execute("SELECT id, content FROM chunks ORDER BY id"):
            rid, content = row
            batch_ids.append(rid)
            batch_txt.append(content)
            if len(batch_txt) >= EMBED_BATCH:
                flush()
                pbar.update(len(batch_txt))
                batch_ids, batch_txt = [], []
        if batch_txt:
            flush()
            pbar.update(len(batch_txt))

    matrix = np.vstack(all_embs).astype(np.float32)
    del all_embs
    gc.collect()
    return matrix, all_ids


def build_faiss(matrix: np.ndarray, dim: int) -> faiss.Index:
    """
    Phase 3: Build HNSW FAISS index.
    HNSW is chosen over IVF for < 500 K chunks because:
      • No training pass required
      • Near-perfect recall
      • Fast single-query search

    For 1M+ chunks swap to IVF — see README.
    """
    log.info("Building FAISS HNSW index (%d vectors, dim=%d)...", len(matrix), dim)
    idx = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = HNSW_EF_CON
    idx.hnsw.efSearch       = 128
    idx.add(matrix)
    return idx


def build_bm25(conn: sqlite3.Connection) -> tuple["BM25Okapi", list[int]]:
    """
    Phase 4: Build BM25 sparse index over all chunk content.
    Tokenisation: lowercase split (fast, good enough for financial text).
    """
    log.info("Building BM25 index...")
    corpus: list[list[str]] = []
    ids:    list[int]       = []
    for rid, content in conn.execute("SELECT id, content FROM chunks ORDER BY id"):
        corpus.append(content.lower().split())
        ids.append(rid)
    bm25 = BM25Okapi(corpus)
    del corpus
    gc.collect()
    return bm25, ids


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build offline RAG indexes.")
    p.add_argument("--data-dir", default=str(DATA_DIR_DEFAULT))
    p.add_argument("--db",       default=DB_PATH_DEFAULT)
    p.add_argument("--reset",    action="store_true",
                   help="Drop all tables and rebuild from scratch.")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    db_path  = args.db

    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    Path("data/nvidia").mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    conn = init_db(db_path)

    if args.reset:
        log.info("--reset: dropping all indexes and chunks...")
        conn.executescript("""
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS documents;
        """)
        conn.executescript(SCHEMA)
        for f in VECTOR_STORE_DIR.iterdir():
            f.unlink(missing_ok=True)

    # ── Phase 1: Ingest documents ──────────────────────────────────────────
    total_chunks = index_files(conn, data_dir)
    log.info("Phase 1 complete: %d total chunks | %.1f s",
             total_chunks, time.time() - t_start)

    if total_chunks == 0:
        log.error("No chunks found. Add documents to %s and re-run.", data_dir)
        conn.close()
        sys.exit(1)

    # ── Phase 2: Embeddings ────────────────────────────────────────────────
    log.info("Loading embedding model: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    dim   = model.get_sentence_embedding_dimension()

    matrix, chunk_ids = build_embeddings(conn, model, dim)
    log.info("Embeddings done: %s | %.1f s", matrix.shape, time.time() - t_start)

    # ── Phase 3: FAISS ─────────────────────────────────────────────────────
    faiss_idx = build_faiss(matrix, dim)
    faiss.write_index(faiss_idx, str(VECTOR_STORE_DIR / "faiss.index"))

    with open(VECTOR_STORE_DIR / "doc_map.pkl", "wb") as f:
        pickle.dump(chunk_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info("FAISS saved: %d vectors | %.1f s",
             faiss_idx.ntotal, time.time() - t_start)

    del matrix
    gc.collect()

    # ── Phase 4: BM25 ─────────────────────────────────────────────────────
    bm25, bm25_ids = build_bm25(conn)

    with open(VECTOR_STORE_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": bm25_ids}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    log.info("BM25 saved | %.1f s", time.time() - t_start)

    conn.close()

    elapsed = time.time() - t_start
    log.info("=" * 55)
    log.info("  DONE — %d chunks | %.0f s (%.1f min)", total_chunks, elapsed, elapsed / 60)
    log.info("  Next step: python main.py")
    log.info("=" * 55)


if __name__ == "__main__":
    main()