"""
llm.py - Local GGUF model inference via llama-cpp-python.

GPU-optimised defaults. Auto-detects CUDA and offloads all layers.
Falls back to CPU gracefully if no GPU is available.

Environment variables (all optional):
    LLM_MODEL_PATH   path to .gguf file (auto-detect in models/)
    LLM_N_CTX        context window     (default: 4096)
    LLM_GPU_LAYERS   layers to offload  (default: -1 = all layers auto)
    LLM_THREADS      CPU threads        (default: physical cores)
    LLM_MAX_TOKENS   max answer tokens  (default: 512)
    LLM_CONTEXT_WORDS max context words (default: 800)
"""

import logging
import os
import time
import warnings
from pathlib import Path

log = logging.getLogger("llm")

MODELS_DIR    = Path(os.environ.get("LLM_MODEL_DIR",  "models"))
MODEL_PATH    = os.environ.get("LLM_MODEL_PATH",  "")
N_CTX         = int(os.environ.get("LLM_N_CTX",       4096))
MAX_TOKENS    = int(os.environ.get("LLM_MAX_TOKENS",    512))
CONTEXT_WORDS = int(os.environ.get("LLM_CONTEXT_WORDS", 800))

import os as _os
_physical_cores = max(1, (_os.cpu_count() or 4) // 2)
THREADS = int(os.environ.get("LLM_THREADS", _physical_cores))


def _detect_gpu_layers() -> int:
    """
    Auto-detect best GPU layer count.
    Returns -1 (all layers) if CUDA is available, 0 for CPU-only.
    Can be overridden with LLM_GPU_LAYERS env var.
    """
    env_val = os.environ.get("LLM_GPU_LAYERS", "")
    if env_val:
        return int(env_val)
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info("GPU detected: %s (%.1f GB VRAM)",
                     torch.cuda.get_device_name(0), vram_gb)
            # -1 = offload all layers (llama.cpp handles it)
            return -1
    except Exception:
        pass
    log.info("No GPU detected — running on CPU.")
    return 0


# ---------------------------------------------------------------------------
# Prompt templates — best quality for each model family
# ---------------------------------------------------------------------------

MISTRAL_TEMPLATE = """\
<s>[INST] You are a precise financial analyst assistant.
Answer the question using ONLY the provided context.
Be specific — include numbers, dates, and figures when available.
If the answer is not in the context, say exactly: "Not found in the provided documents."
Do not guess or hallucinate figures.

Context:
{context}

Question: {question} [/INST]
"""

LLAMA3_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise financial analyst assistant.
Answer the question using ONLY the provided context.
Be specific — include numbers, dates, and figures when available.
If the answer is not in the context, say exactly: "Not found in the provided documents."
Do not guess or hallucinate figures.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Context:
{context}

Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

TINYLLAMA_TEMPLATE = """\
<|system|>
You are a financial analyst. Answer using ONLY the context below.
Include specific numbers and dates. If not in context say "Not found in documents."</s>
<|user|>
Context:
{context}

Question: {question}</s>
<|assistant|>
"""

PHI3_TEMPLATE = """\
<|user|>
You are a financial analyst. Answer using ONLY the context below.
Include specific numbers and dates. If not in context say "Not found in documents."

Context:
{context}

Question: {question}<|end|>
<|assistant|>
"""

GEMMA_TEMPLATE = """\
<start_of_turn>user
You are a financial analyst. Answer using ONLY the context below.
Include specific numbers and dates. If not in context say "Not found in documents."

Context:
{context}

Question: {question}<end_of_turn>
<start_of_turn>model
"""


def _pick_template(name: str) -> tuple:
    """Return (template, stop_tokens) based on model filename."""
    n = name.lower()
    if "tinyllama" in n or "tiny" in n:
        return TINYLLAMA_TEMPLATE, ["</s>", "<|user|>", "<|system|>"]
    if "llama-3" in n or "llama3" in n or "meta-llama-3" in n:
        return LLAMA3_TEMPLATE, ["<|eot_id|>", "<|end_of_text|>"]
    if "phi-3" in n or "phi3" in n:
        return PHI3_TEMPLATE, ["<|end|>", "<|user|>"]
    if "gemma" in n:
        return GEMMA_TEMPLATE, ["<end_of_turn>"]
    if "mistral" in n or "mixtral" in n:
        return MISTRAL_TEMPLATE, ["[INST]", "</s>"]
    # Fallback to Mistral template — works for most instruction-tuned models
    return MISTRAL_TEMPLATE, ["[INST]", "</s>"]


def _find_model() -> Path:
    if MODEL_PATH:
        p = Path(MODEL_PATH)
        if not p.exists():
            raise FileNotFoundError(f"LLM_MODEL_PATH='{MODEL_PATH}' not found.")
        return p
    MODELS_DIR.mkdir(exist_ok=True)
    gguf_files = sorted(MODELS_DIR.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(
            f"No .gguf model found in {MODELS_DIR}/\n\n"
            "Recommended models (place in models/ folder):\n"
            "  GPU 8GB+  : Mistral-7B-Instruct-v0.2.Q4_K_M.gguf\n"
            "  GPU 24GB+ : Mixtral-8x7B or LLaMA-3-70B\n"
            "  CPU only  : tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf\n\n"
            "Download with:\n"
            "  huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF "
            "mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models/"
        )
    # Prefer larger/better models when multiple are present
    # Priority: mistral > llama3 > phi > tinyllama
    priority = ["mistral", "llama-3", "llama3", "phi-3", "phi3", "gemma", "tinyllama"]
    for pref in priority:
        for f in gguf_files:
            if pref in f.stem.lower():
                return f
    return gguf_files[-1]  # fallback: last alphabetically (usually largest)


class LocalLLM:
    """
    GPU-optimised local LLM wrapper.
    Loads lazily on first generate() call.
    Auto-detects GPU and offloads all layers.
    """

    def __init__(self):
        self._llm      = None
        self._template = None
        self._stop     = None

    def _load(self):
        if self._llm is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "CPU:  pip install llama-cpp-python\n"
                "CUDA: CMAKE_ARGS='-DLLAMA_CUBLAS=on' "
                "pip install llama-cpp-python --force-reinstall --no-cache-dir"
            )

        model_path = _find_model()
        gpu_layers = _detect_gpu_layers()

        log.info("Loading LLM: %s", model_path.name)
        log.info("  n_ctx=%d | gpu_layers=%s | threads=%d",
                 N_CTX,
                 "ALL" if gpu_layers == -1 else gpu_layers,
                 THREADS)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._llm = Llama(
                model_path      = str(model_path),
                n_ctx           = N_CTX,
                n_gpu_layers    = gpu_layers,
                n_threads       = THREADS,
                n_threads_batch = THREADS,
                n_batch         = 512,    # larger batch = faster GPU prefill
                use_mmap        = True,   # memory-mapped loading = faster startup
                use_mlock       = False,  # don't lock RAM (important on 16GB systems)
                verbose         = False,
            )

        self._template, self._stop = _pick_template(model_path.stem)
        log.info("LLM ready. Template: %s", model_path.stem)

    def generate(self, question: str, context: str, sources: list) -> dict:
        """
        Generate an answer for question given context.
        Truncates context to CONTEXT_WORDS to control latency.
        """
        self._load()

        # Truncate context to word budget
        ctx_words = context.split()
        if len(ctx_words) > CONTEXT_WORDS:
            context = " ".join(ctx_words[:CONTEXT_WORDS])
            log.debug("Context truncated to %d words", CONTEXT_WORDS)

        prompt = self._template.format(
            context  = context.strip(),
            question = question.strip(),
        )

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = self._llm(
                prompt,
                max_tokens     = MAX_TOKENS,
                temperature    = 0.0,     # deterministic
                top_p          = 1.0,
                repeat_penalty = 1.1,
                stop           = self._stop,
                echo           = False,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = output["choices"][0]["text"].strip()

        if not answer:
            answer = "The model did not return an answer. Try rephrasing your question."

        if sources:
            refs = "\n".join(
                f"  [{i+1}] {s['filename']}  (chunk #{s['chunk_index']}, "
                f"score={s.get('rerank_score', s.get('rrf_score', 0)):.4f})"
                for i, s in enumerate(sources)
            )
            answer = f"{answer}\n\nSources:\n{refs}"

        log.info("LLM generated in %.0f ms", latency_ms)
        return {"answer": answer, "latency_ms": latency_ms}

    def unload(self):
        if self._llm:
            del self._llm
            self._llm = None
            log.info("LLM unloaded.")