"""Multi-engine TTS with one-model-in-VRAM-at-a-time eviction.

The CUDA box is shared with other GPU workloads, so this module keeps at most
ONE engine's model resident in VRAM. Switching engines (or going idle) unloads
the current model and returns its VRAM to the driver before loading the next.

All load / unload / generate calls MUST run on the server's single GPU executor
thread (Metal/MPS/CUDA thread affinity + serialization). The ModelManager does
no locking of its own beyond a light guard because the executor already
serializes every GPU job, including the idle-eviction sweep.

Engines:
  - QwenEngine   : qwen_tts (existing pytorch backend), GPU voice prompts.
  - VoxcpmEngine : VoxCPM2, voices = reference clip (timbre) + optional tone
                   seed (prompt). Voice state is just file paths -> no GPU
                   tensors to juggle across eviction.
  - DotsEngine   : dots.tts, voices = reference clip (+ ref_text in continuation
                   mode). No tone seed. Also file-path-only voice state.
  - CosyvoiceEngine : CosyVoice 3 via a sidecar process (own torch pin).
"""
import os
import gc
import io
import hashlib
import time
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import requests


def pcm16(audio_f32) -> bytes:
    """float32 [-1,1] mono -> little-endian s16le PCM bytes."""
    a = np.clip(np.asarray(audio_f32, dtype=np.float32), -1.0, 1.0)
    return (a * 32767.0).astype("<i2").tobytes()


def _free_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _trim_ram():
    """Return freed heap pages to the OS. torch/glibc hold freed allocations in
    the process arena by default; malloc_trim hands them back so a co-resident
    renderer doesn't get OOM-killed while we sit idle with no model loaded."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Engine base
# ---------------------------------------------------------------------------
class Engine:
    name = "base"
    sample_rate = 24000

    def load(self):
        raise NotImplementedError

    def unload(self):
        """Free this engine's VRAM. Drops GPU-side voice state too; voices are
        rebuilt lazily (from disk/paths) the next time the engine is resident."""
        raise NotImplementedError

    @property
    def loaded(self) -> bool:
        raise NotImplementedError

    def prepare_voice(self, voice_id: str, voice_dir: Path, meta: dict):
        """Ensure any per-voice GPU artifacts exist (model is resident)."""

    def generate(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        """Return (audio_float32_1d, sample_rate)."""
        raise NotImplementedError

    def stream(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        """Yield audio_float32_1d chunks. Default: one chunk via generate()."""
        audio, _sr = self.generate(text, voice_id, voice_dir, meta, language, gen_kwargs)
        yield audio


# ---------------------------------------------------------------------------
# Qwen3-TTS engine (pytorch)
# ---------------------------------------------------------------------------
class QwenEngine(Engine):
    name = "qwen"
    sample_rate = 24000

    AVAILABLE_MODELS = {
        "0.6B": "Qwen3-TTS-12Hz-0.6B-Base",
        "1.7B": "Qwen3-TTS-12Hz-1.7B-Base",
    }

    def __init__(self, models_dir: Path):
        self._models_dir = models_dir
        self._model = None
        self.model_name = None
        self._prompts: dict[str, list] = {}   # voice_id -> List[VoiceClonePromptItem] (GPU)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        import torch
        from qwen_tts import Qwen3TTSModel

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu")
        key = os.environ.get("POLYTTS_MODEL", "1.7B")
        if key not in self.AVAILABLE_MODELS:
            key = "1.7B"
        self.model_name = self.AVAILABLE_MODELS[key]
        path = self._models_dir / self.model_name
        if not path.exists():
            raise RuntimeError(f"Qwen model not found: {path}")
        print(f"[qwen] loading {self.model_name} on {device} …", flush=True)
        self._model = Qwen3TTSModel.from_pretrained(
            str(path), device_map=device, dtype=torch.float32, attn_implementation="sdpa")
        print("[qwen] loaded.", flush=True)

    def unload(self):
        # GPU voice prompts are tied to the model device; drop them and rebuild
        # lazily on next use rather than shuttling tensors CPU<->GPU.
        self._prompts.clear()
        self._model = None
        gc.collect()
        _free_cuda()
        _trim_ram()
        print("[qwen] unloaded.", flush=True)

    def prepare_voice(self, voice_id, voice_dir, meta):
        if voice_id in self._prompts:
            return
        wav_path = voice_dir / "voice.wav"
        self._prompts[voice_id] = self._model.create_voice_clone_prompt(
            ref_audio=str(wav_path),
            ref_text=meta["ref_text"],
            x_vector_only_mode=bool(meta.get("x_vector_only_mode", False)),
        )

    def generate(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        self.prepare_voice(voice_id, voice_dir, meta)
        # Strip VoxCPM/CosyVoice-only knobs before forwarding to the qwen model.
        gk = {k: v for k, v in gen_kwargs.items()
              if k not in ("cfg_value", "inference_timesteps", "denoise", "instruct")}
        wavs, sr = self._model.generate_voice_clone(
            text=text, language=language or "Chinese",
            voice_clone_prompt=self._prompts[voice_id], **gk)
        return np.asarray(wavs[0], dtype=np.float32), sr


# ---------------------------------------------------------------------------
# VoxCPM2 engine
# ---------------------------------------------------------------------------
class VoxcpmEngine(Engine):
    name = "voxcpm"
    sample_rate = 48000

    def __init__(self):
        self._model = None
        self.model_name = os.environ.get("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
        self._cfg = float(os.environ.get("VOXCPM_CFG_VALUE", "3.3"))
        self._steps = int(os.environ.get("VOXCPM_TIMESTEPS", "10"))

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        from voxcpm import VoxCPM
        print(f"[voxcpm] loading {self.model_name} …", flush=True)
        self._model = VoxCPM.from_pretrained(self.model_name, load_denoiser=False)
        self.sample_rate = self._model.tts_model.sample_rate
        print(f"[voxcpm] loaded. sr={self.sample_rate}", flush=True)

    def unload(self):
        # Voice state is file paths only -> nothing GPU-resident to drop.
        self._model = None
        gc.collect()
        _free_cuda()
        _trim_ram()
        print("[voxcpm] unloaded.", flush=True)

    def _clone_kwargs(self, voice_dir: Path, meta: dict, gen_kwargs: dict | None = None) -> dict:
        """reference clip = timbre; optional seed clip = locked tone.

        Per-request overrides (cfg_value / inference_timesteps / denoise) are
        read from gen_kwargs; absent -> engine env defaults."""
        gk = gen_kwargs or {}
        kw = dict(reference_wav_path=str(voice_dir / "voice.wav"),
                  cfg_value=gk.get("cfg_value", self._cfg),
                  inference_timesteps=gk.get("inference_timesteps", self._steps),
                  normalize=True)
        seed = voice_dir / "seed.wav"
        if seed.exists() and meta.get("seed_text"):
            kw["prompt_wav_path"] = str(seed)
            kw["prompt_text"] = meta["seed_text"]
        if gk.get("denoise"):
            kw["denoise"] = True
        return kw

    def generate(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        audio = self._model.generate(text=text, **self._clone_kwargs(voice_dir, meta, gen_kwargs))
        return np.asarray(audio, dtype=np.float32), self.sample_rate

    def stream(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        ck = self._clone_kwargs(voice_dir, meta, gen_kwargs)
        for chunk in self._model.generate_streaming(text=text, **ck):
            yield np.asarray(chunk, dtype=np.float32)


# ---------------------------------------------------------------------------
# dots.tts engine
# ---------------------------------------------------------------------------
class DotsEngine(Engine):
    """dots.tts — fully-continuous AR TTS (Qwen2.5-1.5B backbone + a flow-matching
    DiT head over a 48 kHz AudioVAE, with a frozen CAM++ speaker x-vector as a
    side input). Voice state is file paths only, like VoxCPM.

    Two cloning paths, and which one a voice gets is DECLARED by its meta, never
    inferred: ``x_vector_only_mode`` -> timbre embedding alone (prompt audio, no
    transcript); otherwise *continuation*, which additionally feeds ``ref_text``
    as ``prompt_text``. The two measure materially differently on our own voices
    (0.838 vs 0.762 speaker similarity on a VoxAlert pack), so this engine must
    not quietly substitute one for the other — a silent downgrade surfaces months
    later as "the voice sounds different now". See
    openspec/changes/add-dots-tts-engine/design.md.
    """
    name = "dots"
    sample_rate = 48000

    def __init__(self):
        self._rt = None
        self.model_name = os.environ.get("POLYTTS_DOTS_MODEL", "dots-studio/dots.tts-mf")
        # NFE 4 + optimize=True are LOAD-BEARING, not tuning. The MeanFlow
        # checkpoint ships no `sampling` block, so the library falls back to
        # (euler, 10 steps, cfg 1.2) — at which these same weights measured
        # RTF 1.81-2.02 on an RTX 3090, i.e. SLOWER THAN REAL TIME, against
        # 0.23-0.37 at 4 steps with optimize=True. Do not "simplify" these
        # constants away: at the library defaults this engine reads as a
        # regression and the result is buried.
        self._steps = int(os.environ.get("POLYTTS_DOTS_NUM_STEPS", "4"))
        self._optimize = os.environ.get("POLYTTS_DOTS_OPTIMIZE", "1").lower() not in ("0", "false", "no")

    @property
    def loaded(self) -> bool:
        return self._rt is not None

    def load(self):
        from dots_tts.runtime import DotsTtsRuntime
        print(f"[dots] loading {self.model_name} (steps={self._steps}, "
              f"optimize={self._optimize}) …", flush=True)
        self._rt = DotsTtsRuntime.from_pretrained(
            self.model_name, precision="bfloat16", optimize=self._optimize)
        self.sample_rate = int(self._rt.sample_rate)
        # from_pretrained stages the checkpoint through host memory before the
        # warm model lands on CUDA; hand those pages back without unloading the
        # GPU-resident model (same reclaim as voxcpm).
        gc.collect()
        _trim_ram()
        print(f"[dots] loaded. sr={self.sample_rate}", flush=True)

    def unload(self):
        # Voice state is file paths only -> nothing GPU-resident to drop.
        self._rt = None
        gc.collect()
        _free_cuda()
        _trim_ram()
        print("[dots] unloaded.", flush=True)

    def _clone_kwargs(self, voice_dir: Path, meta: dict, language) -> dict:
        """reference clip = timbre; ref_text = the transcript the model continues
        from, in continuation mode only.

        VoxCPM/Qwen per-request knobs (cfg_value, inference_timesteps,
        temperature, ...) are deliberately NOT forwarded. They are differently
        scaled here — VoxCPM's cfg_value 3.3 against dots' guidance_scale 1.2 —
        so passing them through would change quality under a name that means
        something else on this engine.
        """
        kw = dict(prompt_audio_path=str(voice_dir / "voice.wav"),
                  num_steps=self._steps)
        if not meta.get("x_vector_only_mode", False):
            kw["prompt_text"] = meta["ref_text"]
        if language:
            # normalize_language_code() accepts our qwen-style names ("Chinese",
            # "English") via langcodes and returns None for anything it cannot
            # resolve, so an unknown language degrades to auto-detect.
            kw["language"] = language
        return kw

    # A generation is a COLLAPSE when the model emits EOS almost immediately and
    # returns a fraction of a second of audio for a whole sentence. English runs
    # ~14 characters of text per second of speech; under this fraction of that
    # estimate is not a short reading, it is a failure.
    #
    # DETECTION ONLY — there is deliberately no retry. Measured on the reference
    # that produces it (hl-hev-suit, 30 s of Half-Life HEV announcer audio): the
    # collapse is reproducible across three different RNG seeds, across
    # prompt_text truncated to 289/200/120/60 characters, and across four
    # rewordings of the target line. The seed cannot move it because the seed
    # feeds the flow-matching noise while the EOS decision is the AR backbone's,
    # which runs deterministically. A redraw costs a full generation and returns
    # the same bytes, so this logs and moves on rather than pretending to recover.
    #
    # Rate on this host: 2 of 60 generations (3.3%) over ten voice packs and six
    # realistic narration lines — and BOTH were hl-hev-suit (2 of its 6 lines).
    # The other nine packs were clean across 54 generations. It is a property of
    # particular (reference, text) pairs, not a background flake rate.
    _COLLAPSE_FLOOR_RATIO = 0.25
    _COLLAPSE_MIN_S = 0.35

    def _seed(self, voice_id: str, text: str):
        """dots.tts draws its flow-matching noise from the GLOBAL torch RNG and
        exposes no seed argument, so a repeat of the same (text, voice) would
        otherwise differ byte-for-byte — which the /tts/stream disk PCM cache
        assumes it does not. Pin the RNG per (voice, text) before every generation.

        The text is IN the seed deliberately. Seeding from the voice alone makes
        every phrase in that voice start from one RNG state, so a voice that draws
        badly draws badly across the board; per-pair seeding keeps one bad draw
        from correlating with the next.

        This is an RNG seed, not a tone lock: the engine has no tone-seed input, so
        VoxCPM's `seed.wav` / `seed_text` have no meaning here and are ignored.
        """
        import torch
        h = hashlib.sha256(f"{voice_id}\x00{text}".encode()).hexdigest()[:8]
        torch.manual_seed(int(h, 16))

    def _collapsed(self, text: str, samples: int) -> bool:
        """Did this generation fail rather than merely run short?

        Reported so the pair is identifiable. A caller cannot tell 0.16 s of audio
        from a very short line, and the /tts/stream disk cache will store the short
        body and serve it forever after, so without this the failure is invisible
        in the logs and permanent in the cache.
        """
        floor = max(self._COLLAPSE_MIN_S,
                    (len(text) / 14.0) * self._COLLAPSE_FLOOR_RATIO)
        return (samples / float(self.sample_rate)) < floor

    def _warn_if_collapsed(self, text, voice_id, samples):
        if self._collapsed(text, samples):
            print(f"[dots] COLLAPSED generation: {samples / self.sample_rate:.2f}s of "
                  f"audio for {len(text)} chars, voice={voice_id}. This pair is "
                  f"deterministic — it will not clear on retry, and the PCM cache "
                  f"will keep it. Re-cut the reference or use a different voice.",
                  flush=True)

    def generate(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        self._seed(voice_id, text)
        out = self._rt.generate(text=text, **self._clone_kwargs(voice_dir, meta, language))
        audio = out["audio"].detach().float().cpu().numpy().reshape(-1)
        self._warn_if_collapsed(text, voice_id, audio.size)
        return np.asarray(audio, dtype=np.float32), int(out.get("sample_rate", self.sample_rate))

    def stream(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        self._seed(voice_id, text)
        n = 0
        for chunk in self._rt.generate_stream(
                text=text, **self._clone_kwargs(voice_dir, meta, language)):
            a = chunk.detach().float().cpu().numpy().reshape(-1)
            n += a.size
            yield a
        self._warn_if_collapsed(text, voice_id, n)


# ---------------------------------------------------------------------------
# CosyVoice 3 engine (via an isolated sidecar process)
# ---------------------------------------------------------------------------
class CosyvoiceEngine(Engine):
    """CosyVoice 3 — zero-shot voice clone + instruct (emotion/style) control,
    the one real TONE lever (VoxCPM cannot vary tone). CosyVoice pins torch 2.3.1
    which conflicts with this venv (torch 2.12 for qwen/voxcpm), so it runs in a
    separate `cosyvoice` conda env as a sidecar HTTP service (cosyvoice_worker.py
    on 127.0.0.1:8101). This engine is a thin client; PolyTTS's ModelManager still
    orchestrates VRAM — load()/unload() ask the sidecar to load/free the model
    (evicting voxcpm first since the manager keeps one engine in VRAM)."""
    name = "cosyvoice"
    sample_rate = 24000

    def __init__(self, base_url="http://127.0.0.1:8101"):
        self.base_url = base_url
        self._loaded = False
        self.model_name = "Fun-CosyVoice3-0.5B"

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self):
        try:
            r = requests.post(f"{self.base_url}/load", timeout=600)
            r.raise_for_status()
            self.sample_rate = int(r.json().get("sample_rate", self.sample_rate))
            self._loaded = True
            print(f"[cosyvoice] sidecar model loaded (sr={self.sample_rate})", flush=True)
        except Exception as e:
            raise RuntimeError(
                f"cosyvoice sidecar load failed — is cosyvoice_worker.py running on "
                f"{self.base_url}? (conda run -n cosyvoice python cosyvoice_worker.py): {e}")

    def unload(self):
        try:
            requests.post(f"{self.base_url}/unload", timeout=120)
        except Exception:
            pass
        self._loaded = False
        print("[cosyvoice] sidecar model unloaded", flush=True)

    def _synth(self, text, voice_dir, meta, gen_kwargs):
        instruct = gen_kwargs.get("instruct")
        body = {"text": text,
                "voice_wav_path": str(voice_dir / "voice.wav"),
                "ref_text": meta.get("ref_text", "")}
        if instruct:
            body["instruct"] = instruct
        r = requests.post(f"{self.base_url}/tts", json=body, timeout=900)
        r.raise_for_status()
        sr = int(r.headers.get("X-Sample-Rate", self.sample_rate))
        audio, _ = sf.read(io.BytesIO(r.content), dtype="float32")
        return np.asarray(audio, dtype=np.float32), sr

    def generate(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        return self._synth(text, voice_dir, meta, gen_kwargs)

    def stream(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        # CosyVoice streams per-sentence internally; default to one chunk here.
        audio, sr = self._synth(text, voice_dir, meta, gen_kwargs)
        self.sample_rate = sr
        yield audio


# ---------------------------------------------------------------------------
# One-model-in-VRAM manager
# ---------------------------------------------------------------------------
class ModelManager:
    def __init__(self, engines: dict[str, Engine], idle_seconds: int):
        self.engines = engines
        self.idle_seconds = idle_seconds
        self.resident: str | None = None
        self.last_used = time.monotonic()
        self._guard = threading.Lock()

    def ensure(self, name: str) -> Engine:
        """Make `name` the resident engine, evicting any other. GPU-thread only."""
        if name not in self.engines:
            raise KeyError(f"unknown engine: {name}")
        with self._guard:
            if self.resident != name:
                if self.resident is not None:
                    self.engines[self.resident].unload()
                    self.resident = None
                self.engines[name].load()
                self.resident = name
            self.last_used = time.monotonic()
            return self.engines[name]

    def unload_now(self) -> str | None:
        """Force-evict the resident model now, regardless of idle time. Returns
        the name of the engine that was unloaded (or None if already empty).
        GPU-thread only. Used to hand VRAM/RAM to a co-resident workload (e.g.
        the local renderer) without killing the server process."""
        with self._guard:
            evicted = self.resident
            if evicted is not None:
                self.engines[evicted].unload()
                self.resident = None
                print(f"[manager] force-unloaded {evicted}", flush=True)
            else:
                _trim_ram()
            return evicted

    def maybe_evict(self) -> bool:
        """Evict the resident model if idle past the timeout. GPU-thread only."""
        with self._guard:
            if self.resident and (time.monotonic() - self.last_used) > self.idle_seconds:
                evicted = self.resident
                self.engines[evicted].unload()
                self.resident = None
                print(f"[manager] idle-evicted {evicted}", flush=True)
                return True
        return False

    def status(self) -> dict:
        return {
            "resident": self.resident,
            "idle_seconds": self.idle_seconds,
            "idle_for": round(time.monotonic() - self.last_used, 1) if self.resident else None,
            "engines": list(self.engines.keys()),
        }
