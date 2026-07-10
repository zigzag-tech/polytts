"""PolyTTS FastAPI server — multi-engine, dual MLX / manager backend.

Voices are uploaded via POST /voices (content-hashed, deduplicated) and
referenced by voice_id in the POST /tts endpoint.

MLX path: a single persistent model with pre-computed voice embeddings and an
incremental PCM streaming endpoint backed by a disk L2 cache.  Manager path
(non-MLX): a multi-engine ModelManager (qwen + voxcpm) that keeps one model in
VRAM at a time and evicts it after idle so a co-resident workload can reclaim
the GPU.
"""

import os
import gc
import io
import json
import hashlib
import asyncio
import concurrent.futures
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

import cache
import pcm_cache
from engines import QwenEngine, VoxcpmEngine, CosyvoiceEngine, pcm16

RUNTIME = os.environ.get("POLYTTS_RUNTIME", "mlx").lower()

# Non-MLX runtimes use the multi-engine manager (one model in VRAM at a time,
# evicted on engine-switch and after idle so the shared GPU is freed). MLX uses
# a separate native-MLX inference path (NOT engines.py, which is the
# PyTorch/CUDA backend) — so the manager path is PyTorch-only. Making MLX a
# livestack node requires wrapping the native-MLX load/unload in ManagedUnits,
# not flipping this flag. See docs/ — deferred.
_MANAGER_PATH = RUNTIME != "mlx"
IDLE_EVICT_SECONDS = int(os.environ.get("POLYTTS_IDLE_EVICT_SECONDS", "120"))
DEFAULT_ENGINE = os.environ.get("POLYTTS_DEFAULT_ENGINE", "qwen")

# Cap in-memory voice caches so that registering thousands of unique voices
# cannot grow memory without bound.  Evicted voices remain on disk and are
# reloaded on next use.  Default 100 is generous for normal usage.
_MAX_VOICES_IN_MEMORY = int(os.environ.get("POLYTTS_MAX_VOICES_MEM", "100"))
VOICES_DIR = Path(os.environ.get(
    "POLYTTS_VOICES_DIR",
    str(Path(__file__).resolve().parent / "voices"),
))
MODELS_DIR = Path(__file__).resolve().parent / "models"
PORT = int(os.environ.get("POLYTTS_PORT", "8100"))
TTS_TIMEOUT = int(os.environ.get("POLYTTS_TIMEOUT_SECONDS", "600"))
# Audio seconds per streamed chunk.  Lower = faster first-audio, more overhead.
# 0.5 s gives sub-second time-to-first-chunk while keeping per-chunk work small.
STREAM_INTERVAL = float(os.environ.get("POLYTTS_STREAM_INTERVAL", "0.5"))

# MLX voxcpm engine (optional second model on the MLX runtime). When voices
# registered under engine="voxcpm" are present, the MLX path lazily loads this
# model and serves them alongside qwen. Voice ids are hashed exactly like the
# manager path, so a voxcpm voice has the SAME voice_id on the mac (MLX) and the
# CUDA nodes — the caller can hit either without re-registering.
MLX_VOXCPM_MODEL_ID = os.environ.get("POLYTTS_MLX_VOXCPM_MODEL", "mlx-community/VoxCPM2-8bit")
MLX_VOXCPM_CFG = float(os.environ.get("POLYTTS_MLX_VOXCPM_CFG", "2.0"))
MLX_VOXCPM_STEPS = int(os.environ.get("POLYTTS_MLX_VOXCPM_STEPS", "10"))

# Per-engine cap on chars (and sentences) per SINGLE model generation. VoxCPM
# quality degrades on long continuous generation (~>3 sentences / ~80 CJK
# chars): long input is split into multiple generations and the audio
# concatenated into one seamless response. Engines not listed here have NO
# server-side cap (qwen etc. are robust to long text). A client may override
# per-request via max_chars_per_gen / max_sentences_per_gen (0 = unlimited),
# at its own risk.
MAX_CHARS_PER_GEN = {
    "voxcpm": int(os.environ.get("POLYTTS_VOXCPM_MAX_CHARS", "80")),
}
MAX_SENTENCES_PER_GEN = {
    "voxcpm": int(os.environ.get("POLYTTS_VOXCPM_MAX_SENTENCES", "3")),
}

app = FastAPI(title="PolyTTS Server")

# Filled at startup
model = None
model_name = None
voice_meta: OrderedDict[str, dict] = OrderedDict()           # MLX:     voice_id -> {ref_text}

# MLX voice prompt cache: voice_id -> {speaker_embed, ref_codes, ref_text, audio}
# Pre-computed at registration time so that generation avoids redundant
# speaker-encoder and speech-tokenizer work on every cache-miss TTS call.
# The "audio" field is a minimal stub — the full waveform is not retained
# because the monkey-patched methods ignore their input.
_mlx_prompt_cache: OrderedDict[str, dict] = OrderedDict()

# Manager path (non-MLX): the one-model-in-VRAM manager + a GPU-free registry of
# voices (voice_id -> {engine, dir, meta}) scanned from disk at startup.
manager = None        # polycore.ModelManager, set by attach() below
residence = None      # LivestackCoordinator (None in standalone / mlx mode)
voice_registry: "OrderedDict[str, dict]" = OrderedDict()

# MLX runtime, voxcpm engine: a lazily-loaded second model + a GPU-free registry
# of voxcpm voices (voice_id -> {dir, ref_text, meta}). The model loads on the
# first voxcpm request (or at startup if such voices exist) and stays resident —
# the MLX path never evicts, so there is no per-request cold start.
_voxcpm_mlx = None
_mlx_voxcpm_voices: "OrderedDict[str, dict]" = OrderedDict()

# Single-thread executor — keeps all GPU work on ONE thread to respect
# Metal thread affinity (MLX) and MPS requirements (PyTorch).
_gpu_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="gpu",
)

# --- polycore + livestack residency (same wiring as polyasr) ----------------
# Build polycore ManagedUnits — the default engine is HARD_PIN (benchday needs TTS
# hot at all times) — then attach() builds the manager+coordinator and mounts
# /livestack. Exclusive (coload=False): one engine in VRAM at a time, the other
# evicted on switch. Without livestack, polycore's LocalCoordinator reproduces the
# standalone one-model-in-VRAM + idle-evict behaviour. No hard dependency.
HOST_ID = os.environ.get("POLYTTS_HOST_ID", os.environ.get("HOST_ID", "zz-tower0"))
if _MANAGER_PATH:
    from livestack_node import ManagedUnit, ResidencyPolicy, free_cuda

    # VRAM footprints (bytes; estimates from nvidia-smi, refine with measure_footprint).
    _FOOTPRINTS = {"qwen": 8_700_000_000, "voxcpm": 5_000_000_000, "cosyvoice": 3_000_000_000}

    def _engine_unit(name, engine, pin):
        def loader():
            engine.load()
            return engine          # ensure() returns the Engine, as callers expect
        def freer():
            engine.unload()
            free_cuda()
        # Default engine is SOFT_PIN: preferred-warm but PREEMPTIBLE under GPU
        # pressure (e.g. an ASR/align burst) — the residence planner restores it
        # when pressure settles. The other engine is pure demand (UNPINNED).
        return ManagedUnit(name, loader, freer,
                           footprint=_FOOTPRINTS.get(name, 0),
                           residency_policy=(ResidencyPolicy.SOFT_PIN if pin
                                             else ResidencyPolicy.UNPINNED))

    _ENGINES = {"qwen": QwenEngine(MODELS_DIR), "voxcpm": VoxcpmEngine(), "cosyvoice": CosyvoiceEngine()}
    _UNITS = {n: _engine_unit(n, e, n == DEFAULT_ENGINE) for n, e in _ENGINES.items()}

    def _gpu_call(fn):
        """Run a thunk on the single GPU executor (warm/evict from /livestack)."""
        return _gpu_executor.submit(fn).result()

    try:
        from livestack_node import attach
        manager, residence = attach(app, host_id=HOST_ID, kind="polytts", units=_UNITS,
                                    idle_seconds=IDLE_EVICT_SECONDS, coload=False,
                                    gpu_call=_gpu_call)
    except ImportError:
        from livestack_node import ModelManager
        manager = ModelManager(_UNITS, IDLE_EVICT_SECONDS, coload=False)

# MLX runtime: the native-MLX models (qwen `model`, voxcpm `_voxcpm_mlx`) are held
# as globals and served by a separate native path (NOT engines.py, the PyTorch
# backend). Wrap their load/free in ManagedUnits so polytts becomes a livestack
# node: a host-broker can SEE its Metal footprint and EVICT the heavy voxcpm
# engine when idle to relieve Metal pressure (e.g. an ASR align-chunk spike).
# qwen is HARD_PIN (its `model` global is read pervasively in the serving path, so
# it is never evicted); voxcpm is SOFT_PIN + reached only through the
# _load_voxcpm_mlx() accessor above, so it reloads safely on next use.
elif RUNTIME == "mlx":
    from livestack_node import ManagedUnit, ResidencyPolicy

    def _qwen_unit_load():
        _load_mlx()
        return model

    def _qwen_unit_free():
        global model
        model = None
        import mlx.core as mx
        mx.clear_cache()
        gc.collect()

    # Deferred wrappers: the voxcpm raw load/free are defined later in the file.
    def _voxcpm_unit_load():
        return _load_voxcpm_mlx_raw()

    def _voxcpm_unit_free():
        _free_voxcpm_mlx()

    _MLX_UNITS = {
        "qwen": ManagedUnit("qwen", _qwen_unit_load, _qwen_unit_free,
                            footprint=3_000_000_000,
                            residency_policy=ResidencyPolicy.HARD_PIN),
        "voxcpm": ManagedUnit("voxcpm", _voxcpm_unit_load, _voxcpm_unit_free,
                              footprint=8_000_000_000,
                              residency_policy=ResidencyPolicy.SOFT_PIN),
    }

    def _gpu_call(fn):
        """Run a thunk on the single MLX GPU executor (facade warm/evict)."""
        return _gpu_executor.submit(fn).result()

    try:
        from livestack_node import attach
        manager, residence = attach(app, host_id=HOST_ID, kind="polytts",
                                    units=_MLX_UNITS, idle_seconds=IDLE_EVICT_SECONDS,
                                    coload=True, gpu_call=_gpu_call)
    except ImportError:
        manager = None
        residence = None

# ---------------------------------------------------------------------------
# Admission control — bound concurrent in-flight streaming syntheses so a burst
# cannot pile up unbounded on the single GPU thread.  Under cap: no behavior
# change; over cap: honest 429 + Retry-After.  (Per-account fairness needs the
# account threaded from the hub capability; this is the global guard.)
#
# On the manager path the high-level generate_voice_clone returns whole clips,
# so /tts/stream streams at sentence granularity: synthesize each sentence and
# emit its PCM as soon as it is ready -> first-audio latency is one sentence,
# not the whole digest.
# ---------------------------------------------------------------------------
_MAX_INFLIGHT = max(1, int(os.environ.get("POLYTTS_MAX_INFLIGHT", "3")))
_RETRY_AFTER_SECONDS = max(1, int(os.environ.get("POLYTTS_RETRY_AFTER", "2")))
_inflight = 0  # single-threaded event loop -> a plain counter is race-free
_SENTENCE_ENDERS = ".!?\n。！？"


def _try_admit() -> bool:
    """Reserve an in-flight slot if under cap; caller must _release() when done."""
    global _inflight
    if _inflight >= _MAX_INFLIGHT:
        return False
    _inflight += 1
    return True


def _release() -> None:
    global _inflight
    if _inflight > 0:
        _inflight -= 1


def _split_sentences(text):
    """Split into sentence chunks (ASCII + CJK enders + newline). No regex so
    there are no unicode-escape pitfalls."""
    text = (text or "").strip()
    if not text:
        return []
    out, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in _SENTENCE_ENDERS:
            seg = "".join(buf).strip()
            if seg:
                out.append(seg)
            buf = []
    seg = "".join(buf).strip()
    if seg:
        out.append(seg)
    return out or [text]


# Clause-level punctuation used to hard-split a single over-long sentence.
_CLAUSE_ENDERS = ",;，；、、"


def _hard_split_clause(sentence, max_chars):
    """Split one over-long sentence into pieces each <= max_chars, breaking on
    clause punctuation first, then hard on character count as a last resort."""
    if not max_chars or len(sentence) <= max_chars:
        return [sentence]
    pieces, buf = [], ""
    for ch in sentence:
        buf += ch
        if ch in _CLAUSE_ENDERS and len(buf) >= max_chars:
            pieces.append(buf)
            buf = ""
    if buf:
        pieces.append(buf)
    out = []
    for p in pieces:
        if len(p) <= max_chars:
            out.append(p)
        else:
            for i in range(0, len(p), max_chars):
                out.append(p[i:i + max_chars])
    return out or [sentence]


def _chunk_text(text, max_chars, max_sentences):
    """Greedily pack sentences into generation chunks, each <= max_chars chars
    and <= max_sentences sentences, so a single model generation never runs
    long enough for quality to degrade. max_chars=0 / max_sentences=0 disables
    that limit; both 0 -> one chunk (no chunking). Over-long single sentences
    are hard-split on clause boundaries."""
    text = (text or "").strip()
    if not text:
        return []
    no_char_cap = not max_chars
    no_sent_cap = not max_sentences
    if no_char_cap and no_sent_cap:
        return [text]
    sentences = _split_sentences(text)
    if (no_char_cap or len(text) <= max_chars) and (no_sent_cap or len(sentences) <= max_sentences):
        return [text]
    chunks, cur, cur_sents = [], "", 0
    for s in sentences:
        if max_chars and len(s) > max_chars:
            if cur:
                chunks.append(cur)
                cur, cur_sents = "", 0
            for piece in _hard_split_clause(s, max_chars):
                chunks.append(piece)
            continue
        candidate = (cur + s) if cur else s
        overflow_chars = max_chars and len(candidate) > max_chars
        overflow_sents = max_sentences and cur_sents + 1 > max_sentences
        if cur and (overflow_chars or overflow_sents):
            chunks.append(cur)
            cur, cur_sents = s, 1
        else:
            cur, cur_sents = candidate, cur_sents + 1
    if cur:
        chunks.append(cur)
    return chunks or [text]


_LANG_NAMES = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian",
}


def _lang_name(code):
    """Base ISO code (what the phone sends) -> the model's expected language
    name. Pass through an already-named value; default English for terminal
    content (the model's own default is Chinese, wrong for code)."""
    if not code:
        return "English"
    c = str(code).strip()
    return _LANG_NAMES.get(c.lower(), c)

# ---------------------------------------------------------------------------
# MLX backend
# ---------------------------------------------------------------------------

# Default: 1.7B-8bit for best voice quality (~2.9 GB: 2.25 GB backbone + 651 MB codec).
# Override with POLYTTS_MLX_MODEL for lower memory at reduced quality:
#   0.6B-4bit: ~1.63 GB (mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit)
#   0.6B-6bit: ~1.65 GB (mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit)
MLX_MODEL_ID = os.environ.get(
    "POLYTTS_MLX_MODEL",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
)

# How much freed Metal memory the MLX allocator may hoard for reuse.
# Lower = tighter steady-state footprint; higher = fewer re-allocations.
# 0 disables the cache entirely.  Default: 256 MB.
_MLX_CACHE_LIMIT = int(os.environ.get("POLYTTS_MLX_CACHE_MB", "256")) * 1024 * 1024


def _trim_voice_caches() -> None:
    """Evict oldest voices from in-memory caches when over capacity.

    Oldest = least-recently registered/loaded.  Evicted voices remain on disk
    and are reloaded automatically on next use.
    """
    while len(voice_meta) > _MAX_VOICES_IN_MEMORY:
        oldest, _ = voice_meta.popitem(last=False)
        _mlx_prompt_cache.pop(oldest, None)


def _load_mlx():
    global model, model_name
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    model_name = MLX_MODEL_ID.split("/")[-1]
    print(f"Loading MLX model {model_name} …")
    model = load_model(MLX_MODEL_ID)

    # Cap the Metal allocator's free-buffer cache so freed memory returns
    # to the OS instead of being held indefinitely for potential reuse.
    mx.set_cache_limit(_MLX_CACHE_LIMIT)

    active_mb = mx.get_active_memory() / 1024 / 1024
    print(f"MLX model loaded.  Active Metal memory: {active_mb:.0f} MB  "
          f"(cache limit: {_MLX_CACHE_LIMIT // 1024 // 1024} MB)")
    mx.reset_peak_memory()


def _precompute_voice_mlx(wav_path: str, ref_text: str) -> dict:
    """Extract and return voice-specific artifacts that are reused across
    every TTS call for the same voice.

    Returns dict with:
      speaker_embed: mx.array — x-vector from the speaker encoder
      ref_codes:     mx.array or None — speech tokenizer codes (ICL mode)
      ref_text:      str
      audio:         mx.array — minimal stub (monkey-patched methods ignore it)
    """
    import mlx.core as mx
    from mlx_audio.utils import load_audio

    audio = load_audio(wav_path, sample_rate=model.sample_rate)

    # Speaker embedding (x-vector)
    speaker_embed = None
    if getattr(model, "speaker_encoder", None) is not None:
        speaker_embed = model.extract_speaker_embedding(audio)
        mx.eval(speaker_embed)

    # Reference codec codes (for ICL voice cloning)
    ref_codes = None
    st = getattr(model, "speech_tokenizer", None)
    if st is not None and getattr(st, "has_encoder", False):
        audio_enc = audio[None, None, :] if audio.ndim == 1 else audio[None, :]
        ref_codes = st.encode(audio_enc)
        mx.eval(ref_codes)

    # Replace full waveform with a tiny stub.  The generate() pipeline
    # requires ref_audio to be a non-None mx.array with ndim >= 1, but
    # both extract_speaker_embedding and speech_tokenizer.encode are
    # monkey-patched to return cached values — so the stub is never
    # actually processed.  This avoids retaining ~0.5-1 MB per voice.
    audio_stub = mx.zeros((model.sample_rate,))

    # Release the full waveform from Metal memory
    del audio
    mx.clear_cache()
    return {
        "speaker_embed": speaker_embed,
        "ref_codes": ref_codes,
        "ref_text": ref_text,
        "audio": audio_stub,
    }


def _prompt_cache_dir(voice_dir: Path) -> Path:
    """Return the model-specific prompt cache directory for a voice.

    Layout: voices/<voice_id>/prompts/<model_key>/
    Each model gets its own namespace so switching models (e.g. 0.6B <-> 1.7B)
    never requires recomputation — both caches coexist on disk.
    """
    model_key = hashlib.sha256(MLX_MODEL_ID.encode()).hexdigest()[:12]
    return voice_dir / "prompts" / model_key


def _save_voice_cache_mlx(voice_dir: Path, prompt: dict):
    """Persist pre-computed voice embeddings as .npy so they survive restarts."""
    cache_dir = _prompt_cache_dir(voice_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if prompt["speaker_embed"] is not None:
        np.save(str(cache_dir / "speaker_embed.npy"), np.array(prompt["speaker_embed"]))
    if prompt["ref_codes"] is not None:
        np.save(str(cache_dir / "ref_codes.npy"), np.array(prompt["ref_codes"]))


def _load_voice_cache_mlx(voice_dir: Path, ref_text: str) -> dict | None:
    """Load persisted voice embeddings from disk for the current model.

    Returns None if no cache exists for this model — a different model's
    cache is simply ignored, not deleted."""
    import mlx.core as mx

    cache_dir = _prompt_cache_dir(voice_dir)
    embed_path = cache_dir / "speaker_embed.npy"
    if not embed_path.exists():
        return None

    codes_path = cache_dir / "ref_codes.npy"
    speaker_embed = mx.array(np.load(str(embed_path)))
    ref_codes = mx.array(np.load(str(codes_path))) if codes_path.exists() else None

    # Minimal stub — see _precompute_voice_mlx for rationale
    audio_stub = mx.zeros((model.sample_rate,))

    return {
        "speaker_embed": speaker_embed,
        "ref_codes": ref_codes,
        "ref_text": ref_text,
        "audio": audio_stub,
    }


def _load_voices_mlx():
    """Load previously-uploaded voices from disk (MLX backend).

    Tries to load pre-computed embeddings (.npy) first; falls back to full
    recomputation from the WAV file if cached embeddings are missing.
    """
    if not VOICES_DIR.exists():
        return
    for voice_dir in sorted(VOICES_DIR.iterdir()):
        if not voice_dir.is_dir():
            continue
        meta_path = voice_dir / "meta.json"
        wav_path = voice_dir / "voice.wav"
        if not meta_path.exists() or not wav_path.exists():
            continue
        voice_id = voice_dir.name
        try:
            meta = json.loads(meta_path.read_text())
            ref_text = meta["ref_text"]

            # voxcpm voices are served by the MLX voxcpm model, not the qwen
            # speaker-embedding cache — register and skip the qwen precompute.
            if meta.get("engine") == "voxcpm":
                _mlx_voxcpm_voices[voice_id] = {
                    "dir": voice_dir, "ref_text": ref_text, "meta": meta}
                print(f"  registered voxcpm voice: {voice_id}")
                continue

            # Try cached embeddings first (fast path — no model inference)
            prompt = _load_voice_cache_mlx(voice_dir, ref_text)
            if prompt is None:
                # Cold start: compute from WAV and persist for next time
                print(f"  computing voice prompt: {voice_id}")
                prompt = _precompute_voice_mlx(str(wav_path), ref_text)
                _save_voice_cache_mlx(voice_dir, prompt)
            else:
                print(f"  loaded cached voice: {voice_id}")

            voice_meta[voice_id] = {"ref_text": ref_text}
            _mlx_prompt_cache[voice_id] = prompt
        except Exception as e:
            print(f"  WARNING: skipping voice {voice_id}: {e}")
    _trim_voice_caches()
    print(f"Loaded {len(voice_meta)} voices")


def _register_voice_mlx(voice_id: str, wav_path: str, ref_text: str):
    """Register a new voice — pre-compute and cache embeddings.

    Must be called from the GPU executor thread.
    """
    prompt = _precompute_voice_mlx(wav_path, ref_text)
    _save_voice_cache_mlx(VOICES_DIR / voice_id, prompt)
    voice_meta[voice_id] = {"ref_text": ref_text}
    voice_meta.move_to_end(voice_id)
    _mlx_prompt_cache[voice_id] = prompt
    _mlx_prompt_cache.move_to_end(voice_id)
    _trim_voice_caches()


@contextmanager
def _mlx_voice_context(text: str, voice_id: str | None, extra_kwargs: dict | None = None):
    """Yield generate() kwargs with cached voice embeddings injected.

    When a voice_id has pre-computed data, we temporarily replace the model's
    extract_speaker_embedding and speech_tokenizer.encode with lambdas that
    return the cached values, restoring the originals on exit.  This is safe
    because the GPU executor is single-threaded — only one generation runs at
    a time.  Must be entered on the GPU executor thread.
    """
    kwargs: dict = {"text": text}
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    original_extract = None
    original_encode = None

    if voice_id and voice_id in _mlx_prompt_cache:
        cached = _mlx_prompt_cache[voice_id]
        kwargs["ref_audio"] = cached["audio"]
        kwargs["ref_text"] = cached["ref_text"]

        # Inject cached speaker embedding — skip mel spectrogram + encoder CNN
        if cached["speaker_embed"] is not None:
            original_extract = model.extract_speaker_embedding
            _embed = cached["speaker_embed"]
            model.extract_speaker_embedding = lambda *_a, **_kw: _embed

        # Inject cached ref codes — skip speech tokenizer encoding
        if cached["ref_codes"] is not None:
            st = model.speech_tokenizer
            original_encode = st.encode
            _codes = cached["ref_codes"]
            st.encode = lambda *_a, **_kw: _codes

    try:
        yield kwargs
    finally:
        # Always restore originals, even on error
        if original_extract is not None:
            model.extract_speaker_embedding = original_extract
        if original_encode is not None:
            model.speech_tokenizer.encode = original_encode


def _generate_mlx(text: str, voice_id: str | None) -> bytes:
    """Generate one complete WAV, injecting cached voice embeddings."""
    import mlx.core as mx

    with _mlx_voice_context(text, voice_id) as kwargs:
        chunks = []
        sample_rate = None
        for result in model.generate(**kwargs):
            chunks.append(np.array(result.audio))
            if sample_rate is None:
                sample_rate = result.sample_rate

        audio = np.concatenate(chunks)
        del chunks
        mx.clear_cache()
        gc.collect()

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
        return buf.getvalue()


# Sentinel pushed onto the stream queue to signal clean end-of-generation.
_STREAM_DONE = object()


def _float_to_pcm16(audio_arr) -> bytes:
    """Convert a float waveform in [-1, 1] to little-endian signed 16-bit PCM."""
    a = np.asarray(audio_arr, dtype=np.float32).reshape(-1)
    np.clip(a, -1.0, 1.0, out=a)
    return (a * 32767.0).astype("<i2").tobytes()


def _generate_mlx_stream(text, voice_id, queue, loop):
    """Stream PCM chunks as they are produced.

    Runs on the single GPU executor thread.  Pushes raw s16le PCM byte chunks
    onto `queue` via the event loop, then a `_STREAM_DONE` sentinel.  A raised
    exception is pushed instead so the consumer can stop cleanly.
    """
    import mlx.core as mx

    def push(item):
        loop.call_soon_threadsafe(queue.put_nowait, item)

    try:
        stream_flags = {"stream": True, "streaming_interval": STREAM_INTERVAL}
        with _mlx_voice_context(text, voice_id, stream_flags) as kwargs:
            for result in model.generate(**kwargs):
                pcm = _float_to_pcm16(result.audio)
                if pcm:
                    push(pcm)
        mx.clear_cache()
        gc.collect()
        push(_STREAM_DONE)
    except Exception as exc:  # surfaced to the consumer; never crashes the loop
        push(exc)


# ---------------------------------------------------------------------------
# MLX voxcpm engine (second model on the MLX runtime)
# ---------------------------------------------------------------------------
def _load_voxcpm_mlx_raw():
    """Actually load the MLX voxcpm model into the global (the ManagedUnit
    loader). Must run on the GPU executor thread."""
    global _voxcpm_mlx
    if _voxcpm_mlx is None:
        from mlx_audio.tts.utils import load_model
        print(f"Loading MLX voxcpm model {MLX_VOXCPM_MODEL_ID} …", flush=True)
        _voxcpm_mlx = load_model(MLX_VOXCPM_MODEL_ID)
        print(f"MLX voxcpm loaded. sr={getattr(_voxcpm_mlx, 'sample_rate', '?')}", flush=True)
    return _voxcpm_mlx


def _free_voxcpm_mlx():
    """Drop the MLX voxcpm model and return its Metal memory to the OS (the
    ManagedUnit freer). Runs when the broker evicts voxcpm under Metal pressure."""
    global _voxcpm_mlx
    _voxcpm_mlx = None
    import mlx.core as mx
    mx.clear_cache()
    gc.collect()


def _load_voxcpm_mlx():
    """Lazily load the MLX voxcpm model. Must run on the GPU executor thread.
    Routed through the residence manager when attached, so a host-broker can
    evict the heavy (~8 GB) voxcpm engine under Metal pressure and it reloads on
    the next use (every voxcpm path goes through here). Falls back to a direct
    load when no manager is wired."""
    if manager is not None:
        return manager.ensure("voxcpm")
    return _load_voxcpm_mlx_raw()


def _voxcpm_mlx_sr() -> int:
    """Output sample rate for the voxcpm header. VoxCPM2 is 48 kHz; falls back to
    that until the model is resident (the audio is 48 kHz regardless)."""
    return int(getattr(_voxcpm_mlx, "sample_rate", 48000)) if _voxcpm_mlx is not None else 48000


def _voxcpm_mlx_kwargs(voice: dict) -> dict:
    return dict(
        ref_audio=str(voice["dir"] / "voice.wav"),
        ref_text=voice["ref_text"],
        inference_timesteps=MLX_VOXCPM_STEPS,
        cfg_value=MLX_VOXCPM_CFG,
    )


def _generate_voxcpm_mlx(text: str, voice: dict, chunks=None) -> bytes:
    """One complete WAV from the MLX voxcpm model (GPU executor thread).

    Long text is synthesized one chunk at a time (see _chunk_text) and the
    audio concatenated, so a single generation never runs past the VoxCPM cap."""
    import mlx.core as mx

    m = _load_voxcpm_mlx()
    for_chunks = chunks if chunks else [text]
    parts, sample_rate = [], None
    for ch in for_chunks:
        for result in m.generate(text=ch, **_voxcpm_mlx_kwargs(voice)):
            parts.append(np.array(result.audio))
            if sample_rate is None:
                sample_rate = result.sample_rate
    audio = np.concatenate(parts)
    del parts
    mx.clear_cache()
    gc.collect()
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _generate_voxcpm_mlx_stream(text, voice, queue, loop, chunks=None):
    """Push s16le PCM from the MLX voxcpm model onto `queue` (GPU thread).

    VoxCPM2 decodes a clip per generate() call (one yield), so /tts/stream emits
    it at sentence granularity — same shape as the manager path, but with the
    model resident (no cold start). Long input is pre-split into cap-bounded
    chunks (one generation each) so quality does not degrade on long text."""
    import mlx.core as mx

    def push(item):
        loop.call_soon_threadsafe(queue.put_nowait, item)

    try:
        m = _load_voxcpm_mlx()
        for_chunks = chunks if chunks else [text]
        for ch in for_chunks:
            for result in m.generate(text=ch, **_voxcpm_mlx_kwargs(voice)):
                pcm = _float_to_pcm16(result.audio)
                if pcm:
                    push(pcm)
        mx.clear_cache()
        gc.collect()
        push(_STREAM_DONE)
    except Exception as exc:
        push(exc)


# ---------------------------------------------------------------------------
# Manager path (non-MLX): GPU-free voice registry + one-model-in-VRAM manager
# ---------------------------------------------------------------------------
def _scan_voice_registry():
    """Populate voice_registry from disk WITHOUT any GPU work (no model load)."""
    voice_registry.clear()
    if not VOICES_DIR.exists():
        return
    for d in sorted(VOICES_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path, wav_path = d / "meta.json", d / "voice.wav"
        if not meta_path.exists() or not wav_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            print(f"  skipping voice {d.name}: {e}")
            continue
        # Legacy voices (pre-engine field) are Qwen.
        voice_registry[d.name] = {"engine": meta.get("engine", "qwen"), "dir": d, "meta": meta}
    print(f"Registered {len(voice_registry)} voices (models load lazily)")


def _resolve_engine(req_engine, voice_id):
    """(engine_name, registry_entry) for a request, or (None, None) if unknown."""
    entry = voice_registry.get(voice_id)
    if entry is None:
        return None, None
    return (req_engine or entry["engine"]), entry


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    global manager
    if RUNTIME == "mlx":
        # Load the HARD_PIN qwen unit through the manager (registers residency
        # with the broker); direct load if no manager is wired.
        if manager is not None:
            manager.ensure("qwen")
        else:
            _load_mlx()
        _load_voices_mlx()
        # Pre-load the voxcpm model when voxcpm voices exist so the first
        # request isn't slow. Never fail startup if it can't load.
        if _mlx_voxcpm_voices:
            try:
                _load_voxcpm_mlx()
            except Exception as e:
                print(f"[voxcpm-mlx] preload failed (voxcpm voices will 503): {e}")
        return

    # Manager path: manager + residency were wired by attach() at import. Preload
    # the pinned default engine (benchday needs it hot) and start the idle sweep,
    # which calls manager.maybe_evict() -> coordinator.idle_sweep().
    _scan_voice_registry()
    _gpu_executor.submit(manager.ensure, DEFAULT_ENGINE).result()

    async def _idle_loop():
        loop = asyncio.get_running_loop()
        interval = max(5, IDLE_EVICT_SECONDS // 4)
        while True:
            await asyncio.sleep(interval)
            try:
                await loop.run_in_executor(_gpu_executor, manager.maybe_evict)
            except Exception as e:
                print(f"[idle] {e}")

    asyncio.create_task(_idle_loop())
    print(f"Manager ready (units={list(manager.units)}, default={DEFAULT_ENGINE} "
          f"(pinned), livestack={residence is not None}, idle_evict={IDLE_EVICT_SECONDS}s)")


def _hash_audio(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


@app.post("/voices")
async def upload_voice(
    audio: UploadFile = File(...),
    ref_text: str = Form(...),
    x_vector_only_mode: bool = Form(False),
    engine: str = Form(DEFAULT_ENGINE),
    seed_audio: UploadFile | None = File(None),
    seed_text: str | None = Form(None),
):
    """Register a voice.

    Qwen voices = reference clip (timbre). VoxCPM voices may additionally carry
    a tone *seed* (``seed_audio`` + ``seed_text``) so every generation continues
    that locked tone. ``engine``/``seed_*`` are ignored on the MLX path.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    # ----- Legacy MLX path (unchanged) -----
    if not _MANAGER_PATH:
        # voxcpm on the MLX runtime: hash the voice id EXACTLY like the manager
        # path so the same reference produces the same voice_id on every node.
        if engine == "voxcpm":
            seed_bytes = await seed_audio.read() if seed_audio is not None else b""
            h = audio_bytes + b"|" + engine.encode()
            if x_vector_only_mode:
                h += b"|xvec"
            if seed_bytes:
                h += b"|seed:" + hashlib.sha256(seed_bytes).digest()
            if seed_text:
                h += b"|st:" + seed_text.encode()
            voice_id = _hash_audio(h)
            if voice_id in _mlx_voxcpm_voices:
                return {"voice_id": voice_id}
            voice_dir = VOICES_DIR / voice_id
            voice_dir.mkdir(parents=True, exist_ok=True)
            (voice_dir / "voice.wav").write_bytes(audio_bytes)
            meta = {"engine": engine, "ref_text": ref_text, "x_vector_only_mode": x_vector_only_mode}
            if seed_bytes and seed_text:
                (voice_dir / "seed.wav").write_bytes(seed_bytes)
                meta["seed_text"] = seed_text
            (voice_dir / "meta.json").write_text(json.dumps(meta))
            _mlx_voxcpm_voices[voice_id] = {
                "dir": voice_dir, "ref_text": ref_text, "meta": meta}
            print(f"Registered voxcpm voice {voice_id} (mlx)")
            return {"voice_id": voice_id}

        voice_id = _hash_audio(audio_bytes + b"|xvec") if x_vector_only_mode else _hash_audio(audio_bytes)
        if voice_id in _mlx_prompt_cache:
            return {"voice_id": voice_id}
        voice_dir = VOICES_DIR / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        (voice_dir / "voice.wav").write_bytes(audio_bytes)
        (voice_dir / "meta.json").write_text(json.dumps(
            {"ref_text": ref_text, "x_vector_only_mode": x_vector_only_mode}))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _gpu_executor, _register_voice_mlx, voice_id, str(voice_dir / "voice.wav"), ref_text)
        print(f"Registered voice {voice_id}")
        return {"voice_id": voice_id}

    # ----- Manager path -----
    if engine not in ("qwen", "voxcpm", "cosyvoice"):
        raise HTTPException(400, f"Unknown engine: {engine}")
    seed_bytes = await seed_audio.read() if seed_audio is not None else b""

    # Content-hash so identical (audio, engine, mode, seed) dedupes to one id.
    h = audio_bytes + b"|" + engine.encode()
    if x_vector_only_mode:
        h += b"|xvec"
    if seed_bytes:
        h += b"|seed:" + hashlib.sha256(seed_bytes).digest()
    if seed_text:
        h += b"|st:" + seed_text.encode()
    voice_id = _hash_audio(h)

    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / "voice.wav").write_bytes(audio_bytes)
    meta = {"engine": engine, "ref_text": ref_text, "x_vector_only_mode": x_vector_only_mode}
    if seed_bytes and seed_text:
        (voice_dir / "seed.wav").write_bytes(seed_bytes)
        meta["seed_text"] = seed_text
    (voice_dir / "meta.json").write_text(json.dumps(meta))

    # No GPU work here — voice artifacts are built lazily on first /tts when the
    # owning engine is resident, so registration never forces a model load.
    voice_registry[voice_id] = {"engine": engine, "dir": voice_dir, "meta": meta}
    print(f"Registered {engine} voice {voice_id}")
    return {"voice_id": voice_id}


@app.get("/voices")
def list_voices():
    if not _MANAGER_PATH:
        return {"voices": sorted(set(voice_meta.keys()) | set(_mlx_voxcpm_voices.keys()))}
    return {"voices": sorted(voice_registry.keys())}


class TTSRequest(BaseModel):
    text: str
    voice_id: str | None = None
    language: str | None = None
    engine: str | None = None          # manager path: override the voice's engine
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    subtalker_temperature: float | None = None
    subtalker_top_k: int | None = None
    subtalker_top_p: float | None = None
    max_new_tokens: int | None = None
    non_streaming_mode: bool | None = None
    # Server-side generation chunking override (at the caller's own risk).
    # None = use the engine's default cap; 0 = unlimited (one generation).
    max_chars_per_gen: int | None = None
    max_sentences_per_gen: int | None = None
    # VoxCPM generation-quality overrides (engine default when None):
    #   cfg_value = classifier-free-guidance / clone strength (lower = more natural)
    #   inference_timesteps = denoising steps (higher = cleaner, slower)
    #   denoise = denoise the reference/seed audio before cloning
    cfg_value: float | None = None
    inference_timesteps: int | None = None
    denoise: bool | None = None
    # CosyVoice instruct (emotion/style) — e.g. "请用温暖的语气" or "用沉稳严肃的语调".
    # Only the cosyvoice engine honors this; ignored by qwen/voxcpm.
    instruct: str | None = None


def _generation_kwargs(req: TTSRequest) -> dict:
    fields = (
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "subtalker_temperature",
        "subtalker_top_k",
        "subtalker_top_p",
        "max_new_tokens",
        "cfg_value",
        "inference_timesteps",
        "denoise",
        "instruct",
    )
    return {key: getattr(req, key) for key in fields if getattr(req, key) is not None}


def _gen_chunks(engine_name, req):
    """Split req.text into per-generation chunks bounded by the engine's
    server-side cap (overridable per-request). Returns [] only for empty text."""
    mc = req.max_chars_per_gen
    if mc is None:
        mc = MAX_CHARS_PER_GEN.get(engine_name, 0)
    ms = req.max_sentences_per_gen
    if ms is None:
        ms = MAX_SENTENCES_PER_GEN.get(engine_name, 0)
    return _chunk_text(req.text, mc, ms)


@app.post("/tts")
async def tts(req: TTSRequest):
    language = req.language or "Chinese"
    gen_kwargs = _generation_kwargs(req)
    non_streaming_mode = req.non_streaming_mode if req.non_streaming_mode is not None else False
    loop = asyncio.get_running_loop()

    if not _MANAGER_PATH:
        # voxcpm voice → MLX voxcpm model (its own cache namespace + 48 kHz).
        vox = _mlx_voxcpm_voices.get(req.voice_id) if req.voice_id else None
        if vox is not None:
            vox_key = ("voxcpm", req.text, req.voice_id, language,
                       req.max_chars_per_gen, req.max_sentences_per_gen)
            cached = cache.get(vox_key)
            if cached is not None:
                return Response(content=cached, media_type="audio/wav")
            vox_chunks = _gen_chunks("voxcpm", req)
            try:
                wav_bytes = await asyncio.wait_for(
                    loop.run_in_executor(_gpu_executor, _generate_voxcpm_mlx, req.text, vox, vox_chunks),
                    timeout=TTS_TIMEOUT)
            except asyncio.TimeoutError:
                raise HTTPException(504, "TTS generation timed out")
            cache.put(vox_key, wav_bytes)
            return Response(content=wav_bytes, media_type="audio/wav")

        cache_key = (req.text, req.voice_id, language, non_streaming_mode,
                     json.dumps(gen_kwargs, sort_keys=True))
        cached = cache.get(cache_key)
        if cached is not None:
            return Response(content=cached, media_type="audio/wav")
        if req.voice_id and req.voice_id not in _mlx_prompt_cache:
            raise HTTPException(404, f"Unknown voice_id: {req.voice_id}")
        try:
            wav_bytes = await asyncio.wait_for(
                loop.run_in_executor(_gpu_executor, _generate_mlx, req.text, req.voice_id),
                timeout=TTS_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(504, "TTS generation timed out")
        cache.put(cache_key, wav_bytes)
        return Response(content=wav_bytes, media_type="audio/wav")

    # ----- Manager path -----
    if not req.voice_id:
        raise HTTPException(400, "voice_id is required")
    engine_name, entry = _resolve_engine(req.engine, req.voice_id)
    if entry is None:
        raise HTTPException(404, f"Unknown voice_id: {req.voice_id}")

    cache_key = (engine_name, req.text, req.voice_id, language, non_streaming_mode,
                 json.dumps(gen_kwargs, sort_keys=True),
                 req.max_chars_per_gen, req.max_sentences_per_gen)
    cached = cache.get(cache_key)
    if cached is not None:
        return Response(content=cached, media_type="audio/wav")

    def job():
        eng = manager.ensure(engine_name)
        # VoxCPM etc. degrade on long continuous generation: synthesize one
        # chunk at a time and concatenate, so a single model call never runs
        # past the engine's cap.
        chunks = _gen_chunks(engine_name, req)
        parts, sr = [], None
        for ch in chunks:
            audio, _sr = eng.generate(
                ch, req.voice_id, entry["dir"], entry["meta"], language,
                {**gen_kwargs, "non_streaming_mode": non_streaming_mode})
            parts.append(np.asarray(audio, dtype=np.float32))
            if sr is None:
                sr = _sr
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, sr or eng.sample_rate, subtype="PCM_16", format="WAV")
        return buf.getvalue()

    try:
        wav_bytes = await asyncio.wait_for(
            loop.run_in_executor(_gpu_executor, job), timeout=TTS_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(504, "TTS generation timed out")
    cache.put(cache_key, wav_bytes)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    """Stream raw PCM as the model produces it.

    Body: little-endian signed 16-bit mono PCM, chunked as generated.
    Headers advertise the format so the caller can wrap/transcode (e.g. to
    Opus).  No WAV header is emitted — the length is unknown mid-stream.

    MLX path: the generation runs on the single GPU executor thread; chunks
    flow out through an asyncio queue.  First-audio latency ≈ model
    time-to-first-chunk (tuned by POLYTTS_STREAM_INTERVAL), not the full clip
    duration.  Identical (text, voice_id) requests are served from a disk L2
    PCM cache, skipping the GPU entirely.

    Manager path: sentence-chunked across engines; within a sentence, engines
    with native streaming (VoxCPM) emit sub-sentence chunks for lower
    first-audio latency.  Qwen emits one chunk per sentence.
    """
    if not _MANAGER_PATH:
        vox = _mlx_voxcpm_voices.get(req.voice_id) if req.voice_id else None
        if req.voice_id and vox is None and req.voice_id not in _mlx_prompt_cache:
            raise HTTPException(404, f"Unknown voice_id: {req.voice_id}")

        sr = _voxcpm_mlx_sr() if vox is not None else int(getattr(model, "sample_rate", 24000))
        base_headers = {
            "X-Sample-Rate": str(sr),
            "X-Audio-Format": "s16le",
            "X-Channels": "1",
            "Cache-Control": "no-store",
        }

        # L2 cache: identical (text, voice_id) ⇒ identical audio. Serve from disk and
        # skip the GPU entirely (no admission slot needed — no synthesis runs).
        cached = pcm_cache.get(req.text, req.voice_id)
        if cached is not None:
            cached_sr, cached_bytes = cached

            async def cached_body():
                chunk = 32768
                for i in range(0, len(cached_bytes), chunk):
                    yield cached_bytes[i:i + chunk]

            return StreamingResponse(
                cached_body(),
                media_type="application/octet-stream",
                headers={**base_headers, "X-Sample-Rate": str(cached_sr), "X-Cache": "hit"},
            )

        if not _try_admit():
            raise HTTPException(
                429, "TTS server busy",
                headers={"Retry-After": str(_RETRY_AFTER_SECONDS)},
            )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        # Kick off generation on the GPU thread; do NOT await — chunks arrive via
        # the queue.  The future is intentionally not held: if the client
        # disconnects, generation runs to completion and its chunks are dropped.
        vox_chunks = _gen_chunks("voxcpm", req) if vox is not None else None
        if vox is not None:
            loop.run_in_executor(
                _gpu_executor, _generate_voxcpm_mlx_stream, req.text, vox, queue, loop, vox_chunks,
            )
        else:
            loop.run_in_executor(
                _gpu_executor, _generate_mlx_stream, req.text, req.voice_id, queue, loop,
            )

        async def body():
            chunks: list[bytes] = []
            completed = False
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=TTS_TIMEOUT)
                    except asyncio.TimeoutError:
                        print("[tts/stream] timed out waiting for audio")
                        return
                    if item is _STREAM_DONE:
                        completed = True
                        return
                    if isinstance(item, Exception):
                        # Mid-stream failure: end the response.  Bytes already sent
                        # are valid PCM; the caller treats a short stream as a soft
                        # failure.
                        print(f"[tts/stream] generation error: {item}")
                        return
                    chunks.append(item)
                    yield item
            finally:
                _release()
                # Only cache a CLEANLY completed synthesis — never a partial stream
                # (client disconnect / timeout / error), which would poison the cache
                # with truncated audio.
                if completed and chunks:
                    try:
                        pcm_cache.put(req.text, req.voice_id, sr, b"".join(chunks))
                    except Exception as e:  # noqa: BLE001
                        print(f"[pcm_cache] put failed: {e}")

        return StreamingResponse(
            body(),
            media_type="application/octet-stream",
            headers={**base_headers, "X-Cache": "miss"},
        )

    # ----- Manager path -----
    if not req.voice_id:
        raise HTTPException(400, "voice_id is required")
    engine_name, entry = _resolve_engine(req.engine, req.voice_id)
    if entry is None:
        raise HTTPException(404, f"Unknown voice_id: {req.voice_id}")

    if not _try_admit():
        raise HTTPException(
            429, "TTS server busy",
            headers={"Retry-After": str(_RETRY_AFTER_SECONDS)},
        )

    try:
        language = _lang_name(req.language)
        gen_kwargs = _generation_kwargs(req)
        non_streaming_mode = req.non_streaming_mode if req.non_streaming_mode is not None else False
        gen_chunks = _gen_chunks(engine_name, req)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        # Make the engine resident first so the response sample-rate header is
        # correct (cold start may load the model here).
        eng = await loop.run_in_executor(_gpu_executor, manager.ensure, engine_name)
        sample_rate = eng.sample_rate

        def worker():
            try:
                e = manager.ensure(engine_name)
                for gch in gen_chunks:
                    for chunk in e.stream(
                        gch, req.voice_id, entry["dir"], entry["meta"], language,
                        {**gen_kwargs, "non_streaming_mode": non_streaming_mode},
                    ):
                        pcm = pcm16(chunk)
                        if pcm:
                            loop.call_soon_threadsafe(queue.put_nowait, pcm)
                loop.call_soon_threadsafe(queue.put_nowait, _STREAM_DONE)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        loop.run_in_executor(_gpu_executor, worker)

        async def body():
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=TTS_TIMEOUT)
                    except asyncio.TimeoutError:
                        print("[tts/stream] timed out waiting for audio")
                        return
                    if item is _STREAM_DONE:
                        return
                    if isinstance(item, Exception):
                        print(f"[tts/stream] generation error: {item}")
                        return
                    yield item
            finally:
                _release()

        headers = {
            "X-Sample-Rate": str(sample_rate),
            "X-Audio-Format": "s16le",
            "X-Channels": "1",
            "X-Engine": engine_name,
            "Cache-Control": "no-store",
        }
        return StreamingResponse(body(), media_type="application/octet-stream", headers=headers)
    except Exception:
        _release()
        raise


@app.post("/model/unload")
async def model_unload():
    """Gracefully evict the resident model from VRAM (and return freed heap to
    the OS) without stopping the server. Lets a co-resident workload — e.g. the
    local renderer — reclaim GPU+RAM; the model reloads lazily on the next /tts.
    No-op on the MLX path (single persistent model)."""
    if not _MANAGER_PATH or manager is None:
        raise HTTPException(400, "unload requires the manager (non-MLX) runtime")
    loop = asyncio.get_running_loop()
    evicted = await loop.run_in_executor(_gpu_executor, manager.unload_now)
    return {"unloaded": evicted, "manager": manager.status()}


@app.get("/health")
def health():
    info = {
        "model": model_name,
        "runtime": RUNTIME,
        "tts_timeout_seconds": TTS_TIMEOUT,
        "inflight": _inflight,
        "max_inflight": _MAX_INFLIGHT,
    }

    if not _MANAGER_PATH:
        import mlx.core as mx
        info["device"] = "apple-silicon-mlx"
        info["voices"] = sorted(voice_meta.keys())
        info["memory_mb"] = {
            "active": round(mx.get_active_memory() / 1024 / 1024),
            "peak": round(mx.get_peak_memory() / 1024 / 1024),
            "cache": round(mx.get_cache_memory() / 1024 / 1024),
        }
        info["wav_cache_entries"] = len(cache._store)
    else:
        import torch
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["vram_mb"] = {
                "allocated": round(torch.cuda.memory_allocated() / 1e6),
                "reserved": round(torch.cuda.memory_reserved() / 1e6),
            }
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["device"] = "mps"
        else:
            info["device"] = "cpu"
        info["manager"] = manager.status() if manager else None
        info["model"] = sorted(manager.resident) if manager else None
        info["voices"] = sorted(voice_registry.keys())
        info["wav_cache_entries"] = len(cache._store)

    return info


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
