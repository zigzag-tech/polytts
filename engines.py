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


class IndexTts25Engine(Engine):
    """Pinned IndexTTS 2.5 native backend; model residency belongs to PolyTTS."""
    name = "indextts25"
    sample_rate = 22050

    def __init__(self):
        self._model = None
        self.model_name = "IndexTeam/IndexTTS-2.5"

    @property
    def loaded(self):
        return self._model is not None

    def load(self):
        import sys
        import torch
        source = Path(os.environ["POLYTTS_INDEXTTS25_SOURCE"]).resolve(strict=True)
        models = Path(os.environ["POLYTTS_INDEXTTS25_MODELS"]).resolve(strict=True)
        if not torch.cuda.is_available():
            raise RuntimeError("IndexTTS 2.5 requires the configured CUDA device")
        budget = int(os.environ.get("POLYTTS_INDEXTTS25_BUDGET_BYTES", str(16 * 1024**3)))
        torch.cuda.set_per_process_memory_fraction(budget / torch.cuda.get_device_properties(0).total_memory, 0)
        sys.path.insert(0, str(source))
        from indextts.infer_v2_5 import IndexTTS2
        self._model = IndexTTS2(cfg_path=str(models / "config.yaml"), model_dir=str(models),
            device="cuda:0", use_bf16=True, use_cuda_kernel=False, use_deepspeed=False,
            use_accel=False, use_torch_compile=False, use_qwen_emo=False)
        torch.cuda.synchronize()

    def unload(self):
        model, self._model = self._model, None
        del model
        gc.collect()
        _free_cuda()

    def generate_native(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        import random
        import warnings
        import torch
        from livestack_node.tts.protocol import validate_request
        controls = {
            "durationFactor": gen_kwargs.get("duration_factor", 1.0),
            "emotionAlpha": gen_kwargs.get("emotion_alpha", 1.0),
            "intervalSilenceMs": gen_kwargs.get("interval_silence_ms", 200),
            "maxTextTokensPerSegment": gen_kwargs.get("max_text_tokens_per_segment", 120),
            "seed": gen_kwargs.get("seed", 0),
        }
        if "emotion_vector" in gen_kwargs:
            controls["emotionVector"] = gen_kwargs["emotion_vector"]
        validate_request(dict(jobId="validation", ownerId="polytts", text=text,
            language=language or "en", voiceId=voice_id, referenceSha256="validation", controls=controls))
        random.seed(controls["seed"])
        np.random.seed(controls["seed"])
        torch.manual_seed(controls["seed"])
        torch.cuda.manual_seed_all(controls["seed"])
        with warnings.catch_warnings(record=True) as observed:
            result = self._model.infer(spk_audio_prompt=str(voice_dir / "voice.wav"), text=text,
                output_path=None, lang=language or "en", duration_factor=controls["durationFactor"],
                emo_vector=controls.get("emotionVector"), emo_alpha=controls["emotionAlpha"],
                interval_silence=controls["intervalSilenceMs"],
                max_text_tokens_per_segment=controls["maxTextTokensPerSegment"],
                use_random=False, use_emo_text=False, stream_return=False, text_normalization=True)
        torch.cuda.synchronize()
        messages = [str(item.message) for item in observed]
        if any("max_mel_tokens" in item or "generation stopped" in item for item in messages):
            raise RuntimeError("Truncated IndexTTS 2.5 output refused")
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("IndexTTS 2.5 returned no native audio")
        sr, audio = result
        audio = np.asarray(audio)
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio[:, 0]
        if sr != self.sample_rate or audio.ndim != 1 or audio.dtype != np.int16 or not len(audio):
            raise RuntimeError("IndexTTS 2.5 native mono PCM16 clock refused")
        return audio, sr, messages

    def generate(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        audio, sr, _ = self.generate_native(text, voice_id, voice_dir, meta, language, gen_kwargs)
        return audio.astype(np.float32) / 32768.0, sr


class Drama3Engine(Engine):
    """Exact Fish Audio Drama3 via the upstream audio.cpp community GGML port."""
    name = "drama3"
    sample_rate = 44100

    def __init__(self):
        self._session = None
        self.model_name = "fishaudio/drama-3"
        self._artifacts = []

    @property
    def loaded(self):
        return self._session is not None

    def load(self):
        import torch
        from audio_cpp import NativeAudioSession
        if not torch.cuda.is_available():
            raise RuntimeError("The configured Drama3 backend requires CUDA")
        model = Path(os.environ["POLYTTS_DRAMA3_MODEL"]).resolve(strict=True)
        library = Path(os.environ["POLYTTS_DRAMA3_LIBRARY"]).resolve(strict=True)
        for path, expected in [(model, os.environ["POLYTTS_DRAMA3_MODEL_SHA256"]),
                               (library, os.environ["POLYTTS_DRAMA3_LIBRARY_SHA256"])]:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise RuntimeError("Pinned native Drama3 model/library bytes changed")
        cpu_layers = int(os.environ.get("POLYTTS_DRAMA3_CPU_EXPERT_LAYERS", "8"))
        context = int(os.environ.get("POLYTTS_DRAMA3_CONTEXT_LENGTH", "2048"))
        if not 0 <= cpu_layers <= 40 or context not in (2048, 4096, 8192):
            raise RuntimeError("Closed Drama3 session profile refused")
        self._session = NativeAudioSession(library, model, family="fish_drama3",
            session_options={"fish_drama3.cpu_expert_layers": cpu_layers,
                             "fish_drama3.context_length": context}, threads=4)
        torch.cuda.synchronize()

    def unload(self):
        if self._session is not None:
            self._session.close()
            self._session = None
        self._artifacts = []
        gc.collect()
        _free_cuda()

    @staticmethod
    def validate_controls(controls):
        import math
        required = {"temperature", "topP", "topK", "maxTokens", "seed", "textChunkSize"}
        if not isinstance(controls, dict) or not required <= controls.keys() or controls.keys() - required - {"previousTakes"}:
            raise ValueError("Closed explicit Drama3 controls required")
        for key, low, high in [("temperature", .1, 2), ("topP", .01, 1)]:
            value = controls[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError("Native Drama3 sampling bound refused")
        for key, low, high in [("topK", 1, 4097), ("maxTokens", 32, 2048), ("seed", 0, 2**32-1), ("textChunkSize", 32, 4096)]:
            if type(controls[key]) is not int or not low <= controls[key] <= high:
                raise ValueError("Native Drama3 integer control refused")
        previous = controls.get("previousTakes", [])
        if not isinstance(previous, list) or len(previous) > 4:
            raise ValueError("Bounded explicit Drama3 take history required")
        for take in previous:
            if not isinstance(take, dict) or set(take) != {"text", "vqCodes"} or not isinstance(take["text"], str) or not 1 <= len(take["text"]) <= 4000:
                raise ValueError("Drama3 history accepts exact text and VQ codes, never server paths")
            grid = take["vqCodes"]
            if not isinstance(grid, list) or len(grid) != 10 or not isinstance(grid[0], list) or not 1 <= len(grid[0]) <= 2048:
                raise ValueError("Ten bounded native codebooks required")
            if any(not isinstance(row, list) or len(row) != len(grid[0]) or any(type(value) is not int or not 0 <= value < 2**31 for value in row) for row in grid):
                raise ValueError("Malformed native codebook history")
        return controls

    def generate_native(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        import json
        import torch
        controls = self.validate_controls(gen_kwargs.get("drama_controls", gen_kwargs))
        reference, rate = sf.read(voice_dir / "voice.wav", dtype="float32")
        options = {"temperature": controls["temperature"], "top_p": controls["topP"],
            "top_k": controls["topK"], "max_tokens": controls["maxTokens"], "seed": controls["seed"],
            "text_chunk_size": controls["textChunkSize"], "text_chunk_mode": "tag_aware",
            "reference_text": meta["ref_text"]}
        if "previousTakes" in controls:
            options["previous_takes"] = json.dumps([{"text": take["text"], "vq_codes": take["vqCodes"]} for take in controls["previousTakes"]], separators=(",", ":"), ensure_ascii=False)
        audio, rate, artifacts = self._session.run(text, language or "en", reference, rate, options)
        torch.cuda.synchronize()
        self._artifacts = artifacts
        return audio, rate, []

    def take_artifacts(self):
        result, self._artifacts = self._artifacts, []
        return result

    def generate(self, text, voice_id, voice_dir, meta, language, gen_kwargs):
        audio, rate, _ = self.generate_native(text, voice_id, voice_dir, meta, language, gen_kwargs)
        return audio, rate


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
        # from_pretrained stages the full checkpoint through CPU tensors before
        # moving the warm model to CUDA. Reclaim those now-unused host buffers
        # without unloading the GPU-resident model.
        gc.collect()
        _trim_ram()
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
