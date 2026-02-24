# OFFLINE-RAG — Production Grade AI Question Answering System

A fully offline AI-powered question-answering system for NVIDIA SEC filings.  
**No API calls. No cloud. Runs entirely on local hardware.**  
GPU-optimised with automatic CPU fallback.

---

## What This Does

You ask a question like:
```
What was NVIDIA's revenue in fiscal year 2024?
```
The system searches through hundreds of NVIDIA SEC filings, finds the most relevant passages, and gives you a precise answer with source citations — all running locally on your machine.

---

## Architecture

```
Your Question
      │
      ├─► Dense Search  (FAISS + SentenceTransformer embeddings) ─┐
      │                                                             ├─► RRF Fusion ─► Top 10
      └─► Sparse Search (BM25 keyword matching)                  ─┘
                                                                         │
                                                                         ▼
                                                             Cross-Encoder Reranker
                                                             (picks best 5 chunks)
                                                                         │
                                                                         ▼
                                                             Context Compressor
                                                             (removes duplicates, 50% reduction)
                                                                         │
                                                                         ▼
                                                             Local LLM (GGUF — runs on your GPU/CPU)
                                                                         │
                                                                         ▼
                                                         Answer + Source Citations
                                                         Saved to terminal + output.txt
```

---

## Performance

| Component | GPU (RTX 3060+) | CPU (i7/i5) |
|---|---|---|
| Embedding + Retrieval | ~10 ms | ~80 ms |
| Reranking (10 pairs) | ~50 ms | ~300 ms |
| Context Compression | ~30 ms | ~15 ms |
| LLM — Mistral 7B | ~2–4 sec | ~60–120 sec |
| LLM — TinyLlama 1.1B | ~0.5 sec | ~5–8 sec |
| **Total (GPU)** | **~3–5 sec** | — |
| **Total (CPU)** | — | **~10–15 sec** |

**Recommendation:** Use Mistral 7B on GPU. Use TinyLlama on CPU-only machines.

---

## File Structure

```
OFFLINE-RAG/
├── main.py               — Interactive query CLI (run this to ask questions)
├── build_index.py        — Builds FAISS + BM25 indexes from your documents
├── download_data.py      — Downloads NVIDIA SEC filings from SEC EDGAR
├── retriever.py          — Hybrid dense + sparse retrieval with RRF fusion
├── compression.py        — Query-aware context compression
├── reranker.py           — Cross-encoder reranking
├── llm.py                — Local GGUF LLM inference (auto GPU/CPU)
├── cache.py              — Two-level cache (RAM + disk)
├── encryption.py         — Optional AES-256 document encryption
├── requirements.txt      — Python dependencies
│
├── data/                 — Your documents go here (created after download)
│   └── nvidia/
│       └── raw_html/     — NVIDIA SEC filings (HTML)
├── models/               — Your .gguf model file goes here
├── vector_store/         — Auto-generated indexes (after build_index.py)
├── metadata.db           — Auto-generated SQLite database
├── output.txt            — All answers saved here automatically
└── rag_system.log        — Runtime logs
```

> **Note:** `data/`, `models/`, `vector_store/`, and `metadata.db` are NOT included in this repo.  
> You generate them yourself using the steps below. This keeps the repo small and clean.

---

## Installation

### Step 1 — Clone the repo
```bash
git clone https://github.com/halaae/OFFLINE-RAG.git
cd OFFLINE-RAG
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

**For GPU (NVIDIA CUDA) — recommended:**
```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install FAISS GPU
pip install faiss-gpu

# Install llama-cpp with CUDA support
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install remaining dependencies
pip install sentence-transformers rank-bm25 numpy tqdm huggingface-hub
```

**For CPU only:**
```bash
pip install -r requirements.txt
pip install llama-cpp-python
```

---

## Step 4 — Download the LLM Model

Create the `models/` folder and download one model into it.

**GPU (8GB+ VRAM) — Mistral 7B — Best quality:**
```bash
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models/
```

**CPU only — TinyLlama — Fastest on CPU:**
```bash
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir models/
```

**GPU (24GB+ VRAM) — LLaMA 3 8B — Best overall:**
```bash
huggingface-cli download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --local-dir models/
```

The system **auto-detects** which model is in `models/` — no config needed.

---

## Step 5 — Download NVIDIA SEC Data

This downloads ~80 NVIDIA SEC filings (10-K, 10-Q, 8-K) directly from the  
US government's public SEC EDGAR database. Completely free and legal.

```bash
python download_data.py
```

- Takes 2–3 minutes
- Downloads ~150 MB of HTML filings
- Saves to `data/nvidia/raw_html/`

---

## Step 6 — Build the Index

```bash
python build_index.py
```

This will:
1. Parse and chunk all documents (~2–5 minutes for 80 files)
2. Generate embeddings for all chunks
3. Build the FAISS vector index
4. Build the BM25 keyword index
5. Save everything to `vector_store/` and `metadata.db`

To rebuild from scratch:
```bash
python build_index.py --reset
```

---

## Step 7 — Run and Ask Questions

```bash
python main.py
```

Then type your question:
```
Enter your question: What was NVIDIA's revenue in fiscal year 2024?
```

The answer appears in the terminal and is saved to `output.txt`.

**Special commands:**
| Command | Action |
|---|---|
| `metrics` | Show latency and cache statistics |
| `clear` | Clear the screen |
| `quit` | Exit |

---

## Sample Questions

```
What was NVIDIA's total revenue in fiscal year 2024?
What are NVIDIA's main business segments?
How much did the Data Center segment grow year over year?
What risks does NVIDIA mention about China export restrictions?
What was NVIDIA's net income last year?
What does NVIDIA say about its AI chip strategy?
What guidance did NVIDIA provide for next quarter?
What supply chain risks does NVIDIA disclose?
How did NVIDIA's gross margin change?
What is NVIDIA's automotive business revenue?
```

---

## Environment Variables (Optional Tuning)

Set these before running `main.py` to customise behaviour.

**Windows PowerShell:**
```powershell
$env:LLM_GPU_LAYERS="-1"
$env:LLM_MAX_TOKENS="512"
$env:LLM_CONTEXT_WORDS="800"
python main.py
```

**Linux / macOS:**
```bash
LLM_GPU_LAYERS=-1 LLM_MAX_TOKENS=512 python main.py
```

| Variable | Default | Description |
|---|---|---|
| `LLM_GPU_LAYERS` | auto | `-1` = all layers on GPU, `0` = CPU only |
| `LLM_N_CTX` | 4096 | LLM context window size |
| `LLM_MAX_TOKENS` | 512 | Maximum tokens in answer |
| `LLM_CONTEXT_WORDS` | 800 | Max words fed to LLM (reduce for speed) |
| `LLM_THREADS` | auto | CPU threads (set to physical core count) |
| `EMBED_MODEL` | MiniLM-L6-v2 | Swap to `BAAI/bge-large-en-v1.5` for +10% accuracy |
| `RERANKER_DEVICE` | auto | `cuda` or `cpu` |

---

## Hardware Requirements

| | Minimum | Recommended |
|---|---|---|
| **RAM** | 8 GB | 16–32 GB |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | Not required | NVIDIA RTX 3060 12GB+ |
| **Storage** | 10 GB free | 50 GB NVMe SSD |
| **Python** | 3.10+ | 3.10 or 3.11 |

---

## Troubleshooting

**`No .gguf model found in models/`**  
→ Run the model download command in Step 4.

**`FAISS index not found`**  
→ Run `python build_index.py` first.

**`No relevant documents found`**  
→ Run `python download_data.py` then `python build_index.py`.

**Very slow on CPU**  
→ Switch to TinyLlama (Step 4, CPU option).  
→ Set `LLM_CONTEXT_WORDS=200` to reduce context size.

**CUDA out of memory**  
→ Set `$env:LLM_GPU_LAYERS="20"` to partially offload (not all layers).

**llama-cpp-python build error on Windows**  
→ Install pre-built wheel:  
`pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`

---

## Data Source

All documents are downloaded from the publicly available  
**US SEC EDGAR database**: https://www.sec.gov/edgar

**Company:** NVIDIA Corporation  
**CIK:** 0001045810  
**Filing types:** 10-K (Annual), 10-Q (Quarterly), 8-K (Material Events)  

No private or proprietary data is used. Everything is public regulatory information.

---

## Tech Stack

| Component | Technology |
|---|---|
| Vector search | FAISS (HNSW index) |
| Keyword search | BM25 (rank-bm25) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco (auto GPU/CPU) |
| LLM inference | llama-cpp-python (GGUF format) |
| Metadata storage | SQLite |
| Caching | LRU (RAM) + shelve (disk) |
| Language | Python 3.10+ |

---

## Built By

Hala — VTU Computer Science  
Project: Production-grade offline RAG system for financial document intelligence
