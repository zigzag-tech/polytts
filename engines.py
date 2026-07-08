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
"""
import os
import gc
import io
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
